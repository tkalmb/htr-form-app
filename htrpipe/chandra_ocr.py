"""
chandra_ocr.py -- Chandra 2 runner for the form-HTR baseline.

Chandra 2 is a Qwen3.5-based VLM, so unlike DeepSeek-OCR-2 it loads through the
ordinary `AutoModelForImageTextToText` + `AutoProcessor` path -- the same one the
Qwen condition uses. The awkward part is elsewhere: the prompt the model was
trained on lives inside the `chandra-ocr` package, not in the model card, and the
raw output carries layout markup that the package converts to plain markdown.

So this module supports two backends and lets you check they agree:

  "transformers"  our own apply_chat_template + generate. Same code shape as the
                  Qwen condition, full control over decoding, therefore fully
                  reportable. The prompt is READ OUT of the installed chandra
                  package at run time -- never guessed.
  "vendor"        chandra.model.hf.generate_hf. Least risk of invoking the model
                  wrongly, least control over decoding.

`cross_check()` runs one page through both and reports whether they produced the
same string. Do that once on val before trusting the "transformers" backend.

Nothing here reads ground truth. Field extraction is in formparse.py, shared with
the DeepSeek condition.

Verified from the model card (huggingface.co/datalab-to/chandra-ocr-2): the
loader classes, `BatchInputItem(image=..., prompt_type="ocr_layout")`,
`generate_hf(batch, model)`, `result.raw`, and `parse_markdown`. NOT verified,
and therefore discovered defensively at run time rather than hard-coded: where
`PROMPT_MAPPING` lives inside the package, and the exact prompt strings.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

MODEL_ID = "datalab-to/chandra-ocr-2"

# The two prompt types the package documents. "ocr_layout" is what the model card
# uses and what the benchmarks were run with; "ocr" drops the layout markup.
PROMPT_TYPES = ("ocr_layout", "ocr")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def check_environment(strict: bool = True, require_vendor: bool = False) -> dict:
    """Check transformers knows the architecture, and see what chandra offers.

    Chandra 2's config declares model_type "qwen3_5". Rather than guess which
    transformers release added it -- a version number I would be inventing --
    this asks the installed transformers directly whether it can map that type.
    A capability check is both more honest and more robust than a pin.
    """
    import torch
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    known = "qwen3_5" in CONFIG_MAPPING_NAMES
    if not known:
        msg = (
            f"transformers {transformers.__version__} does not recognise model_type "
            f"'qwen3_5', which Chandra 2 declares. Upgrade transformers, or the load "
            f"will fail with an 'unrecognized architecture' error. (This condition "
            f"needs a different venv from the DeepSeek one, which is pinned to 4.46.x.)"
        )
        if strict:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    vendor = probe_vendor_package()
    if require_vendor and not vendor["importable"]:
        raise RuntimeError(
            "the `chandra` package is not importable (`pip install chandra-ocr[hf]`). "
            "It is needed for the vendor prompt text and for parse_markdown."
        )

    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "qwen3_5_supported": known,
        "chandra_package": vendor,
    }


def _try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception:
        return None


def probe_vendor_package() -> dict:
    """Report what the installed chandra package exposes, without assuming."""
    root = _try_import("chandra")
    info: dict[str, Any] = {
        "importable": root is not None,
        "version": getattr(root, "__version__", None) if root else None,
        "prompt_mapping_at": None,
        "prompt_types": None,
        "has_generate_hf": False,
        "has_parse_markdown": False,
        "has_batch_input_item": False,
        "has_scale_to_fit": False,
    }
    if root is None:
        return info

    for path in ("chandra.model.prompts", "chandra.prompts", "chandra.model"):
        mod = _try_import(path)
        mapping = getattr(mod, "PROMPT_MAPPING", None) if mod else None
        if isinstance(mapping, dict) and mapping:
            info["prompt_mapping_at"] = f"{path}.PROMPT_MAPPING"
            info["prompt_types"] = sorted(mapping)
            break

    info["has_generate_hf"] = hasattr(_try_import("chandra.model.hf") or object(),
                                      "generate_hf")
    info["has_parse_markdown"] = hasattr(_try_import("chandra.output") or object(),
                                         "parse_markdown")
    info["has_batch_input_item"] = hasattr(_try_import("chandra.model.schema") or object(),
                                           "BatchInputItem")
    for path in ("chandra.input", "chandra.model.hf", "chandra.image"):
        if hasattr(_try_import(path) or object(), "scale_to_fit"):
            info["has_scale_to_fit"] = path + ".scale_to_fit"
            break
    return info


def resolve_vendor_prompt(prompt_type: str = "ocr_layout") -> tuple[str, str]:
    """Return (prompt_text, where_it_came_from).

    Reads the prompt out of the installed package. Guessing it would be the
    single worst thing to do here: the model is trained on a specific string, an
    approximation would degrade it, and the degradation would be invisible and
    unattributable in the results.
    """
    for path in ("chandra.model.prompts", "chandra.prompts", "chandra.model"):
        mod = _try_import(path)
        mapping = getattr(mod, "PROMPT_MAPPING", None) if mod else None
        if isinstance(mapping, dict) and prompt_type in mapping:
            value = mapping[prompt_type]
            text = value if isinstance(value, str) else getattr(value, "prompt", str(value))
            return text, f"{path}.PROMPT_MAPPING[{prompt_type!r}]"

    raise RuntimeError(
        f"could not find PROMPT_MAPPING[{prompt_type!r}] in the installed chandra "
        f"package. Probe result: {probe_vendor_package()}. Either install "
        f"`chandra-ocr[hf]`, or use BACKEND='vendor' so the package supplies the "
        f"prompt itself. Do not hand-write the prompt."
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

class ChandraOCR:
    """Wrapper over Chandra 2 with a vendor backend and a transformers backend."""

    def __init__(self, model, processor, prompt_type: str = "ocr_layout",
                 prompt_text: str | None = None, prompt_source: str | None = None):
        self.model = model
        self.processor = processor
        self.prompt_type = prompt_type
        self.prompt_text = prompt_text
        self.prompt_source = prompt_source
        self.max_new_tokens = 2048
        self.do_sample = False
        self.repetition_penalty = 1.0

    @classmethod
    def load(cls, model_path: str = MODEL_ID, prompt_type: str = "ocr_layout",
             attn_implementation: str = "flash_attention_2",
             min_pixels: int | None = None, max_pixels: int | None = None,
             need_prompt_text: bool = True):
        """Load the checkpoint. Returns (runner, meta) where meta goes in the manifest."""
        import torch
        from transformers import AutoProcessor

        model_cls = _resolve_model_class()

        proc_kwargs = {}
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        try:
            processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
            pixel_kwargs_applied = bool(proc_kwargs)
        except TypeError:
            # Not every processor accepts the Qwen pixel-budget kwargs. Failing
            # over silently would leave the manifest claiming a budget that was
            # never applied, so record that it was not.
            processor = AutoProcessor.from_pretrained(model_path)
            pixel_kwargs_applied = False

        # The vendor path expects left padding; harmless for single-image batches
        # but set anyway so both backends see the same processor state.
        try:
            processor.tokenizer.padding_side = "left"
        except AttributeError:
            pass

        used = attn_implementation
        try:
            model = model_cls.from_pretrained(
                model_path, dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
        except Exception as exc:
            print(f"{attn_implementation} unavailable ({type(exc).__name__}); "
                  f"falling back to sdpa.")
            used = "sdpa"
            model = model_cls.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation="sdpa")

        # Explicit .to("cuda"), not device_map="auto": one GPU is pinned, and
        # device_map can silently offload to CPU when memory is tight, turning a
        # memory problem into a mysterious slowdown.
        model = model.to("cuda").eval()
        model.processor = processor          # the vendor backend reads it off the model

        gc = model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None

        # Always try to resolve the prompt, whichever backend is selected. Two
        # reasons: the manifest has to record which string produced the numbers
        # (with need_prompt_text gating this, a vendor-backend run recorded
        # prompt.text = null), and the cross-check needs both paths available.
        prompt_text, prompt_source = None, None
        try:
            prompt_text, prompt_source = resolve_vendor_prompt(prompt_type)
        except RuntimeError as exc:
            if need_prompt_text:
                raise
            print(f"NOTE: prompt text unavailable ({exc.__class__.__name__}); the vendor "
                  f"backend supplies it internally, but it will be missing from the "
                  f"manifest. Install chandra-ocr[hf] to record it.")

        meta = {
            "model_path": str(model_path),
            "model_class": model.__class__.__name__,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "attn_implementation": used,
            "dtype": "bfloat16",
            "min_pixels": min_pixels, "max_pixels": max_pixels,
            "pixel_kwargs_applied": pixel_kwargs_applied,
        }
        return cls(model, processor, prompt_type, prompt_text, prompt_source), meta

    # -- backends -----------------------------------------------------------
    def _transcribe_transformers(self, image):
        import torch

        if self.prompt_text is None:
            raise RuntimeError("no prompt text resolved; use BACKEND='vendor'")

        # The vendor path resizes the image first (28x28 grid alignment). Skipping
        # it was the visible cause of the cross-check mismatch: bounding boxes
        # drifted by ~1px, i.e. the two backends were not seeing the same pixels.
        image = _apply_scale_to_fit(image)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt_text},
        ]}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image],
                                return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _transcribe_vendor(self, image):
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem

        batch = [BatchInputItem(image=image, prompt_type=self.prompt_type)]
        result = generate_hf(batch, self.model)[0]
        return getattr(result, "raw", None) or getattr(result, "markdown", "")

    def transcribe(self, image_path, backend: str = "transformers"):
        """One page -> (raw_text, seconds). Greedy, so repeated calls are identical."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        if backend == "transformers":
            text = self._transcribe_transformers(image)
        elif backend == "vendor":
            text = self._transcribe_vendor(image)
        else:
            raise ValueError(f"unknown backend: {backend!r}")
        return text, time.perf_counter() - t0

    def cross_check(self, image_path) -> dict:
        """Run one page through both backends and compare. Do this once, on val.

        If they disagree, the vendor backend is the one to trust -- it is the
        reference invocation -- and the difference is a bug in this module, not a
        property of the model.
        """
        a, sec_a = self.transcribe(image_path, "transformers")
        b, sec_b = self.transcribe(image_path, "vendor")
        return {
            "identical": a == b,
            "len_transformers": len(a), "len_vendor": len(b),
            "seconds_transformers": round(sec_a, 2), "seconds_vendor": round(sec_b, 2),
            "transformers": a, "vendor": b,
        }

    # -- output -------------------------------------------------------------
    def to_markdown(self, raw: str) -> tuple[str, str]:
        """(markdown, how). Uses the vendor converter when available.

        `ocr_layout` output carries layout markup; `parse_markdown` is what turns
        it into plain markdown. Falling back to the raw string is survivable --
        the label anchors usually still match -- but it is recorded, because a
        run parsed one way is not comparable to a run parsed the other way.
        """
        mod = _try_import("chandra.output")
        fn = getattr(mod, "parse_markdown", None) if mod else None
        if fn is None:
            return raw, "raw (chandra.output.parse_markdown unavailable)"
        try:
            return fn(raw), "chandra.output.parse_markdown"
        except Exception as exc:
            return raw, f"raw (parse_markdown raised {type(exc).__name__})"

    def unload(self) -> None:
        import gc as _gc
        import torch
        del self.model
        _gc.collect()
        torch.cuda.empty_cache()


def _apply_scale_to_fit(image):
    """Use the package's own image resize when it is available.

    Returns the image untouched if not, which is survivable but means the
    transformers backend is not seeing the pixels the vendor backend sees.
    """
    for path in ("chandra.model.hf", "chandra.input", "chandra.image"):
        fn = getattr(_try_import(path) or object(), "scale_to_fit", None)
        if fn is not None:
            try:
                return fn(image)
            except Exception as exc:
                print(f"scale_to_fit raised {type(exc).__name__}; using the raw image")
                return image
    return image


def _resolve_model_class():
    """The model card shows two loader classes in two places; accept either.

    The Chandra usage example uses AutoModelForImageTextToText; the auto-generated
    "Load model directly" snippet on the same page uses AutoModelForMultimodalLM.
    Which one exists depends on the transformers release, so try in that order.
    """
    import transformers
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModel"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no suitable Auto model class found in transformers")


if __name__ == "__main__":
    # No GPU needed: reports what the environment can actually do.
    import json as _json
    print(_json.dumps(probe_vendor_package(), indent=2))
