"""Qwen page-level extraction: zero-shot and few-shot, with field confidence.

PROVENANCE -- read before editing
---------------------------------
The prompt, decoding settings and control flow are the evaluation
implementation that produced the thesis results, taken over with module-level
settings turned into function parameters (defaulting to the evaluated values
below). Three additions sit on top, each a declared deviation recorded in the
run manifest:

* **Few-shot (in-context examples).** The prompt additionally carries k
  ``(form image, correct JSON)`` pairs, assembled by
  ``htrpipe.fewshot.build_messages``. Few-shot output is parsed with an
  echo-aware parser (below), because a model shown example answers sometimes
  repeats them before its own.
* **Field confidence.** ``generate`` additionally returns the raw logits, from
  which each field's ``conf_min`` -- the probability of its least certain
  token -- is computed. Greedy decoding is unchanged, so the predictions are
  identical to a run without confidence.
* **English retry instruction.** Pages whose first output does not parse are
  generated once more with an appended instruction. The thesis runs used a
  German wording; the application uses English for all instructions. In the
  evaluated zero-shot run no page needed a retry.

JSON parsing for zero-shot is ``htrpipe.formparse.parse_json_fields``, the
packaged parser the evaluation settled on.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Evaluated settings. These are the defaults everywhere below; the app
# records any deviation in the run manifest.
# ---------------------------------------------------------------------------

MAX_PIXELS = 4096 * 28 * 28   # ~4096 visual tokens; the vision budget lever
MIN_PIXELS = 256 * 28 * 28
MAX_NEW_TOKENS = 768
DO_SAMPLE = False             # greedy == reproducible
REPETITION_PENALTY = 1.0
SEED = 42
MAX_PARSE_RETRIES = 1

#: Visual budget per demonstration image (few-shot only): a quarter of the
#: query page's tokens, about half its linear resolution. A demonstration only
#: has to convey the layout and the page-to-JSON mapping, not fine strokes.
DEMO_MAX_PIXELS = 1024 * 28 * 28

#: Printed labels of the thesis form. Used only to pre-fill
#: the field table in the UI; the run uses whatever the user confirmed there.
DEFAULT_FIELD_LABELS: Dict[str, str] = {
    "last_name":      "Name",
    "first_name":     "Vorname",
    "monthly_salary": "Monatliches Einkommen",
    "birthday":       "Geburtsdatum",
    "email":          "E-Mail-Adresse",
    "phone":          "Telefonnummer",
    "studies":        "Studiengang",
    "street":         "Stra\u00dfe",
    "house_number":   "Hausnummer",
    "postal_code":    "PLZ",
    "city":           "Ort",
    "comment":        "Kommentar",
    "signature":      "Unterschrift",
}

# ---------------------------------------------------------------------------
# Prompt (evaluated, verbatim)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a transcription system for handwritten German administrative "
    "forms. You return JSON only."
)


def build_prompt(target_fields, checkbox_fields, labels, line_join=" "):
    lines = []
    for f in target_fields:
        label = labels[f.name]
        if isinstance(label, (list, tuple)):
            # The Qwen prompt names one printed label per field; if the user
            # entered variants (a Chandra feature), the first one is used.
            label = label[0]
        if f.name in checkbox_fields:
            opts = " | ".join(checkbox_fields[f.name])
            how = f'checked option, exactly one of these values: {opts}'
        elif f.type == "long_text":
            how = (f'entire handwritten text; join multiple lines with '
                f'"{line_join}"')
        else:
            how = "handwritten entry, transcribed character by character"
        lines.append(f'  "{f.name}"  -> Feld "{label}": {how}')
    schema = "\n".join(lines)

    keys = ", ".join(f'"{f.name}"' for f in target_fields)

    return f"""Transcribe the handwritten entries in this form.

Fields (JSON key -> printed label on the form):
{schema}

Rules:
1. Return exactly these keys, in this order: {keys}
2. ALL values are strings in double quotes -- including pure numbers.
   Write "01234", not 01234 and not 1234. Leading zeros are preserved.
3. Transcribe character by character. Correct NOTHING: no spelling, no date
   or number formats, no umlauts, no names, no place names. Reproduce
   spelling and punctuation exactly as written.
4. Transcribe only the HANDWRITING. Printed labels and the form number in
   the top right corner are not part of it.
5. If a field is empty or illegible, return "". Never guess.
6. Output only the JSON object. No prose, no explanation, no markdown code
   blocks."""


#: Appended to the prompt when the first output does not parse as JSON.
#: English translation of the evaluated (German) wording.
RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your last answer was not valid JSON. Now output ONLY the "
    "JSON object -- starting with { and ending with }. All values as strings."
)

#: Few-shot variant: additionally forbids repeating the examples, which is
#: the usual reason a few-shot output fails to parse.
FEWSHOT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your last answer was not valid JSON. Now output ONLY ONE "
    "JSON object -- starting with { and ending with }. All values as strings. "
    "Do NOT repeat the examples."
)


# ---------------------------------------------------------------------------
# Model loading (evaluated settings; module globals became parameters)
# ---------------------------------------------------------------------------

def load_model(model_path: str,
               min_pixels: int = MIN_PIXELS,
               max_pixels: int = MAX_PIXELS,
               seed: int = SEED):
    """Load the Qwen VL model and processor exactly as the evaluation did.

    Returns ``(model, processor, meta)`` where ``meta`` records the resolved
    attention implementation, parameter count and model type for the run
    manifest. flash_attention_2 is tried first and falls back to sdpa, and the
    manifest says which one actually ran -- resolved once, held constant.
    """
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_cfg_path = pathlib.Path(model_path) / "config.json"
    model_type = (json.loads(model_cfg_path.read_text())["model_type"]
                  if model_cfg_path.exists() else "?")

    if model_type.startswith("qwen3_vl") and Version(transformers.__version__) < Version("4.57.0"):
        raise RuntimeError(
            f"{model_path} is a {model_type} checkpoint, which needs transformers >= 4.57.0 "
            f"(installed: {transformers.__version__}).\n"
            f"Either `pip install -U 'transformers>=4.57.0'` or point the model path at a "
            f"Qwen2.5-VL checkpoint."
        )

    torch.manual_seed(seed)

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels,
    )

    # flash_attention_2 cuts peak memory noticeably on large images, but is an
    # optional dependency -- fall back rather than making the run depend on it.
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        )
        attn_impl = "flash_attention_2"
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        attn_impl = "sdpa"

    # Explicit .to("cuda") rather than device_map="auto": one GPU is pinned,
    # and device_map can silently offload layers to CPU when memory is tight,
    # which turns a memory problem into a mysterious slowdown.
    model = model.to("cuda").eval()

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "model_path": model_path,
        "model_type": model_type,
        "model_class": model.__class__.__name__,
        "n_params": int(n_params),
        "attn_implementation": attn_impl,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "seed": seed,
    }
    return model, processor, meta


# ---------------------------------------------------------------------------
# Few-shot output parsing (echo-aware)
# ---------------------------------------------------------------------------
# Taken over from the few-shot evaluation implementation; the only change is
# that the line-join separator became a parameter.

_BARE_VALUE_RE = re.compile(r'(:\s*)(?!")([^",\{\}\[\]\s][^",\{\}\[\]]*?)(\s*[,\}])')


def _extract_json_span(text):
    """Outermost {...} span, tolerating fences and preamble."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in output")
    return text[start:end + 1]


def _balanced_spans(text):
    """Yield every top-level balanced {...} span, in order of appearance."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _quote_bare_values(blob):
    """Wrap unquoted scalar values in quotes so 01234 survives as a string."""
    prev = None
    while prev != blob:
        prev = blob
        blob = _BARE_VALUE_RE.sub(
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}', blob)
    return blob


def _load_object(blob):
    try:
        return json.loads(blob, parse_int=str, parse_float=str)
    except json.JSONDecodeError:
        return json.loads(_quote_bare_values(blob), parse_int=str, parse_float=str)


def parse_fields_fewshot(text, columns, line_join=" "):
    """Model text -> ({field: str}, extra_keys, n_objects).

    Tries the outermost span first, so single-object output parses exactly as
    in the zero-shot condition. Only if that fails does it fall back to
    per-object scanning, which is where echoed demonstrations get handled.
    """
    spans = list(_balanced_spans(text))
    n_objects = len(spans)

    try:
        obj = _load_object(_extract_json_span(text))
        if not isinstance(obj, dict) or not (set(obj) & set(columns)):
            raise ValueError("outermost span is not the target schema")
    except Exception:
        obj = None
        # LAST matching object, not the first. When the model echoes, it emits
        # the demonstrations and *then* the answer, so the first object is a
        # demonstration -- taking it would store the demo's values as this
        # page's prediction.
        for blob in reversed(spans):
            try:
                candidate = _load_object(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and (set(candidate) & set(columns)):
                obj = candidate
                break
        if obj is None:
            raise ValueError(
                f"no parseable object matching the schema in {n_objects} "
                f"balanced span(s)"
            )

    row, extra = {}, sorted(set(obj) - set(columns))
    for col in columns:
        val = obj.get(col, "")
        if val is None:
            val = ""
        elif isinstance(val, (list, tuple)):
            val = line_join.join(str(v) for v in val)
        elif isinstance(val, dict):
            val = ""
        row[col] = str(val).strip()
    return row, extra, n_objects


# ---------------------------------------------------------------------------
# Per-field confidence (conf_min)
# ---------------------------------------------------------------------------
# Taken over from the confidence evaluation implementation; the line-join
# separator became a parameter. Only conf_min is reported by the application.
#
# Token -> character span: each token's span is found by decoding the growing
# prefix ids[:i+1] -- quadratic, but well under a second for <= 768 tokens and
# exact for Qwen's byte-level tokenizer. Field -> tokens: the value of each key
# is located in the raw JSON text; if a key occurs twice, the LAST occurrence
# is used (the same one json.loads keeps, and the answer when a few-shot model
# echoes). Empty values are scored by the tokens that emitted the "" quotes.
# NaN when the key is missing, the page never parsed, or the incremental
# decode disagrees with the batch decode.

def token_char_spans(ids, tokenizer):
    """[(start, end)] character span of every token in the decoded text."""
    spans, prev = [], ""
    for i in range(len(ids)):
        cur = tokenizer.decode(ids[:i + 1], skip_special_tokens=True)
        start = len(prev)
        spans.append((start, max(start, len(cur))))
        prev = cur
    return spans, prev


def find_value_span(text, key):
    """Locate the value of `key` in (possibly imperfect) JSON text."""
    pat = re.compile(
        r'"' + re.escape(key) + r'"\s*:\s*'
        r'(?:"((?:[^"\\]|\\.)*)"'        # 1: quoted string
        r'|(\[[^\]]*\])'                 # 2: array (comment split into lines)
        r'|([^,}\n]*))'                  # 3: bare value (e.g. 01234)
    )
    matches = list(pat.finditer(text))
    if not matches:
        return None
    m = matches[-1]                      # json.loads keeps the last duplicate
    g = next(k for k in (1, 2, 3) if m.group(k) is not None)
    s, e = m.span(g)
    raw = m.group(g)
    if g == 3:                           # trim whitespace around a bare value
        e = s + len(raw.rstrip())
        s = s + (len(raw) - len(raw.lstrip()))
        raw = raw.strip()
    kind = {1: "string", 2: "array", 3: "bare"}[g]
    outer = (s - 1, e + 1) if kind == "string" else (s, e)
    return {"kind": kind, "content": (s, e), "outer": outer, "raw": raw}


def _value_as_parsed(v, line_join=" "):
    """What the parser would have stored for this span (best effort)."""
    try:
        if v["kind"] == "string":
            return str(json.loads('"' + v["raw"] + '"')).strip()
        if v["kind"] == "array":
            return line_join.join(str(x) for x in json.loads(v["raw"])).strip()
    except (json.JSONDecodeError, TypeError):
        return None
    return v["raw"].strip()


def _empty_rec(span):
    import numpy as np
    return {"n_tokens": 0, "conf_min": np.nan, "conf_first": np.nan,
            "conf_mean": np.nan, "span": span, "span_matches_value": False}


def field_confidences(text, gen, columns, row, tokenizer, line_join=" "):
    """{field: {n_tokens, conf_min, conf_first, conf_mean, span, span_matches_value}}"""
    import numpy as np

    spans, dec_text = token_char_spans(gen["ids"], tokenizer)
    if dec_text != text:
        return {c: _empty_rec("decode_mismatch") for c in columns}

    logp = np.asarray(gen["logp"], dtype=np.float64)
    result = {}
    for col in columns:
        v = find_value_span(text, col)
        if v is None:
            result[col] = _empty_rec("missing")
            continue
        s, e = v["content"]
        if e <= s:                       # empty value -> score the "" tokens
            s, e = v["outer"]
        idx = [i for i, (a, b) in enumerate(spans) if a < e and b > s]
        if not idx:
            result[col] = _empty_rec(v["kind"])
            continue
        lp = logp[idx]
        result[col] = {
            "n_tokens": len(idx),
            "conf_min": float(np.exp(lp.min())),     # weakest token
            "conf_first": float(np.exp(lp[0])),      # first value token
            "conf_mean": float(np.exp(lp.mean())),   # geometric mean probability
            "span": v["kind"],
            "span_matches_value": _value_as_parsed(v, line_join) == row[col],
        }
    return result


# ---------------------------------------------------------------------------
# Single-page transcription
# ---------------------------------------------------------------------------

def build_page_messages(image, prompt: str,
                        demos: Optional[Sequence] = None,
                        columns: Optional[Sequence[str]] = None,
                        system_prompt: str = SYSTEM_PROMPT):
    """``(messages, images)`` for one page, zero-shot or few-shot.

    Zero-shot: exactly the evaluated message layout (image, then prompt).
    Few-shot: ``htrpipe.fewshot.build_messages`` -- task prompt, the k
    examples, the English query preamble, the page, the closing instruction.
    Messages and image list always come from ONE function, so they cannot
    disagree about which image fills which placeholder.
    """
    if demos:
        from htrpipe import fewshot
        return fewshot.build_messages(
            query_image=image, task_prompt=prompt, demos=demos,
            columns=list(columns), system_prompt=system_prompt,
            query_preamble=fewshot.QUERY_PREAMBLE_EN)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": prompt}]},
    ]
    return messages, [image]


def transcribe(model, processor, image_path, prompt: str,
               demos: Optional[Sequence] = None,
               columns: Optional[Sequence[str]] = None,
               system_prompt: str = SYSTEM_PROMPT,
               max_new_tokens: int = MAX_NEW_TOKENS,
               do_sample: bool = DO_SAMPLE,
               repetition_penalty: float = REPETITION_PENALTY):
    """One page -> (raw model text, n input tokens, generation info).

    Greedy, so repeated calls are identical. ``output_logits=True`` returns the
    RAW logits of every generated token (before any logits processor), from
    which the log-probability of each chosen token is taken; it does not
    change which tokens are chosen.
    """
    import torch
    from PIL import Image

    with torch.inference_mode():
        image = Image.open(image_path).convert("RGB")
        messages, images = build_page_messages(image, prompt, demos, columns,
                                               system_prompt)
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=images, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            output_logits=True,             # raw logits, one (batch, vocab) tensor per step
            return_dict_in_generate=True,
        )
        n_in = int(inputs["input_ids"].shape[1])
        gen_ids = out.sequences[0, n_in:]
        assert len(out.logits) == len(gen_ids), "logits/token count mismatch"

        # log p(chosen token) per step. float32 for a numerically safe
        # log_softmax; step by step so the full (steps x vocab) tensor is
        # never materialised.
        token_logp = torch.stack([
            torch.log_softmax(step[0].float(), dim=-1)[tok]
            for step, tok in zip(out.logits, gen_ids)
        ]).cpu().numpy()

        text = processor.batch_decode(gen_ids[None], skip_special_tokens=True)[0]
        gen = {"ids": gen_ids.tolist(), "logp": token_logp}
        del out
        return text, n_in, gen


# ---------------------------------------------------------------------------
# Batch loop (evaluated control flow; console output became a callback)
# ---------------------------------------------------------------------------

def run_batch(forms,
              model, processor,
              user_prompt: str,
              output_columns: List[str],
              demos: Optional[Sequence] = None,
              line_join: str = " ",
              max_parse_retries: int = MAX_PARSE_RETRIES,
              max_new_tokens: int = MAX_NEW_TOKENS,
              progress: Optional[Callable[[int, int, str], None]] = None):
    """Process every form; one stricter retry if the output fails to parse.

    ``forms`` is a sequence of ``(doc_id, image_path)`` pairs. With ``demos``
    the few-shot prompt, parser and retry wording are used. Failures never
    abort the batch: a failed page yields an all-empty row, an entry in
    ``failures`` and NaN confidence for every field.

    Returns ``(rows, runlog_df, failures, conf)`` where ``conf`` is
    ``{doc_id: {field: conf_min}}`` (NaN where no score exists).
    """
    import numpy as np
    import pandas as pd

    from htrpipe.formparse import parse_json_fields

    fewshot_mode = bool(demos)
    retry_suffix = FEWSHOT_RETRY_SUFFIX if fewshot_mode else RETRY_SUFFIX

    rows, records, failures, conf = {}, [], [], {}

    for i, (doc_id, path) in enumerate(forms, 1):
        t0 = time.perf_counter()
        text, in_tokens, gen, row, err, attempts = None, None, None, None, None, 0
        extra: List[str] = []
        n_objects = None

        for attempt in range(max_parse_retries + 1):
            attempts = attempt + 1
            prompt = user_prompt if attempt == 0 else user_prompt + retry_suffix
            try:
                text, in_tokens, gen = transcribe(
                    model, processor, path, prompt, demos=demos,
                    columns=output_columns, max_new_tokens=max_new_tokens)
                if fewshot_mode:
                    row, extra, n_objects = parse_fields_fewshot(
                        text, output_columns, line_join)
                else:
                    row, extra = parse_json_fields(text, output_columns, line_join)
                err = None
                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

        dt = time.perf_counter() - t0

        if row is None:
            row = {c: "" for c in output_columns}
            extra = []
            failures.append({"doc_id": doc_id, "error": err, "raw": text})
            page_conf = {c: np.nan for c in output_columns}
        else:
            # Confidence belongs to the attempt that parsed (the last `gen`).
            scored = field_confidences(text, gen, output_columns, row,
                                       processor.tokenizer, line_join)
            page_conf = {c: scored[c]["conf_min"] for c in output_columns}

        rows[doc_id] = row
        conf[doc_id] = page_conf
        field_min = [v for v in page_conf.values() if not np.isnan(v)]
        record = {
            "doc_id": doc_id, "seconds": round(dt, 2), "input_tokens": in_tokens,
            "attempts": attempts, "parsed": err is None,
            "n_empty": sum(1 for v in row.values() if not v),
            "min_field_conf": min(field_min) if field_min else np.nan,
            "extra_keys": ",".join(extra), "error": err,
        }
        if fewshot_mode:
            # >1 means the model repeated an example before its answer.
            record["n_objects"] = n_objects
        records.append(record)

        if progress is not None:
            status = "ok" if err is None else "FAIL"
            echo = " ECHO" if fewshot_mode and (n_objects or 0) > 1 else ""
            progress(i, len(forms),
                     f"doc_id {doc_id} {status} {dt:.1f}s attempts={attempts}{echo}")

    runlog = pd.DataFrame(records).set_index("doc_id")
    return rows, runlog, failures, conf
