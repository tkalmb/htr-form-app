"""Text recognition engines behind one interface.

Every engine exposes ``recognize(images) -> list[str]``, so the pipeline does
not branch on which model is selected. Heavy dependencies (torch,
transformers, pytesseract, easyocr) are imported lazily inside each engine, so
selecting TrOCR does not require EasyOCR to be installed and vice versa.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .config import HTRConfig
from .preprocess import to_rgb


def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BaseEngine:
    """Interface every recognizer implements."""

    name = "base"

    #: True for models that read a multi-line crop directly.
    #:
    #: TrOCR is a *single-line* recognizer, so ``long_text`` fields are split
    #: into per-line crops before recognition and merged afterwards. The
    #: crop-level VLMs have no such constraint, and pre-splitting for them
    #: would impose TrOCR's limitation on a model that does not share it --
    #: and would drag the line segmenter in as a confound on the one field
    #: where segmentation quality is already a known issue.
    #:
    #: Read via :func:`engine_handles_multiline`, which asks the *class*: the
    #: crops are extracted before the model is loaded.
    handles_multiline = False

    def recognize(self, images: Sequence[np.ndarray]) -> List[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class TrOCREngine(BaseEngine):
    """Hugging Face ``VisionEncoderDecoderModel`` (TrOCR and its fine-tunes).

    The model and processor are loaded once at construction, not per call --
    reloading a checkpoint for every form was one of the slower parts of the
    original notebook.
    """

    def __init__(self, cfg: HTRConfig):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.cfg = cfg
        self.name = cfg.model_path
        self.device = resolve_device(cfg.device)
        self._torch = torch

        self.processor = TrOCRProcessor.from_pretrained(cfg.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(cfg.model_path).to(self.device)
        self.model.eval()
        self._check_special_tokens()

    def _check_special_tokens(self) -> None:
        """Warn on a mismatch between the model config and generation config.

        A ``decoder_start_token_id`` that differs between the two produces
        generations that are subtly wrong rather than obviously broken, and it
        is easy to introduce when saving a fine-tuned checkpoint.
        """
        import warnings

        model_start = getattr(self.model.config, "decoder_start_token_id", None)
        gen_cfg = getattr(self.model, "generation_config", None)
        gen_start = getattr(gen_cfg, "decoder_start_token_id", None) if gen_cfg else None

        if gen_start is not None and model_start is not None and gen_start != model_start:
            warnings.warn(
                f"{self.name}: generation_config.decoder_start_token_id "
                f"({gen_start}) differs from config.decoder_start_token_id "
                f"({model_start}). Decoding will not behave as trained.",
                stacklevel=2,
            )

    def recognize(self, images: Sequence[np.ndarray]) -> List[str]:
        from PIL import Image

        if not images:
            return []

        pil_images = [Image.fromarray(to_rgb(img)) for img in images]
        out: List[str] = []

        for start in range(0, len(pil_images), self.cfg.batch_size):
            batch = pil_images[start:start + self.cfg.batch_size]
            pixel_values = self.processor(
                images=batch, return_tensors="pt"
            ).pixel_values.to(self.device)

            with self._torch.no_grad():
                generated = self.model.generate(
                    pixel_values,
                    max_length=self.cfg.max_length,
                    num_beams=self.cfg.num_beams,
                    early_stopping=self.cfg.num_beams > 1,
                )
            out.extend(self.processor.batch_decode(generated, skip_special_tokens=True))

        return out


class TesseractEngine(BaseEngine):
    """pytesseract baseline. Included for comparison, not as the main model."""

    name = "tesseract"

    def __init__(self, cfg: HTRConfig, tesseract_cmd: Optional[str] = None,
                 tessdata_prefix: Optional[str] = None, lang: str = "deu"):
        import os

        import pytesseract

        if tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._pytesseract = pytesseract
        self.lang = lang

    def recognize(self, images: Sequence[np.ndarray]) -> List[str]:
        return [
            self._pytesseract.image_to_string(to_rgb(img), lang=self.lang).strip()
            for img in images
        ]


class EasyOCREngine(BaseEngine):
    """EasyOCR baseline (CRNN + CTC)."""

    name = "easyocr"

    def __init__(self, cfg: HTRConfig):
        import easyocr

        self.reader = easyocr.Reader(list(cfg.languages),
                                     gpu=resolve_device(cfg.device) == "cuda")

    def recognize(self, images: Sequence[np.ndarray]) -> List[str]:
        out: List[str] = []
        for img in images:
            detections = self.reader.readtext(to_rgb(img))
            # readtext returns (bbox, text, confidence) per detection; the crop
            # is one field, so join whatever it found rather than keeping only
            # the first detection (which silently dropped text in the original).
            out.append(" ".join(d[1] for d in detections).strip())
        return out


ENGINES = {
    "trocr": TrOCREngine,
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
}


def _all_engines() -> dict:
    """Line-level engines plus the crop-level VLMs, if importable.

    The import is guarded because the three VLM families pin mutually
    incompatible ``transformers`` versions and each lives in its own conda env.
    An env that can run TrOCR must still be able to ``import htrpipe`` without
    having chandra-ocr installed.
    """
    engines = dict(ENGINES)
    try:
        from .recognize_vlm import VLM_ENGINES
        engines.update(VLM_ENGINES)
        _all_engines.import_error = None
    except ImportError as exc:
        # Remembered rather than swallowed: without it, selecting "deepseek" in
        # an env that cannot import the module fails with "unknown engine
        # 'deepseek'", which points at a typo instead of at a missing package.
        _all_engines.import_error = exc
    return engines


_all_engines.import_error = None


def build_engine(cfg: HTRConfig, **kwargs) -> BaseEngine:
    """Instantiate the engine named by ``cfg.engine``.

    ``kwargs`` are forwarded to the engine constructor -- this is how
    engine-specific settings (DeepSeek's ``vision_mode``, Chandra's
    ``min_pixels``) are passed without putting them in ``HTRConfig``.
    """
    engines = _all_engines()
    key = cfg.engine.lower()
    if key not in engines:
        hint = ""
        if _all_engines.import_error is not None:
            hint = (f"\nhtrpipe.recognize_vlm could not be imported in this "
                    f"environment ({_all_engines.import_error}), so the "
                    f"crop-level VLM engines are unavailable here. Each VLM "
                    f"family needs its own conda env -- check you are on the "
                    f"right kernel.")
        raise ValueError(
            f"unknown engine {cfg.engine!r} (available: {sorted(engines)}){hint}"
        )
    return engines[key](cfg, **kwargs)


def engine_handles_multiline(cfg: HTRConfig) -> bool:
    """Whether the configured engine reads multi-line crops directly.

    Asks the class rather than an instance: field crops are extracted before
    the recognizer is constructed, and loading a 7B checkpoint to read one
    boolean is absurd.
    """
    return getattr(_all_engines().get(cfg.engine.lower(), BaseEngine),
                   "handles_multiline", False)


def recognize_forms(
    crops_per_form: Sequence[dict],
    engine: BaseEngine,
    skip_empty: bool = True,
) -> List[dict]:
    """Run ``engine`` over every crop of every form in one batched pass.

    All crops are flattened into a single list so the engine sees full batches
    regardless of how many fields each form has, then reassembled using an
    index map. This is why the return value is per-form dicts even though the
    engine call is flat.
    """
    flat: List[np.ndarray] = []
    index_map: List[tuple] = []

    for form_idx, crops in enumerate(crops_per_form):
        for label, crop in crops.items():
            if skip_empty and (crop is None or crop.size == 0):
                continue
            flat.append(crop)
            index_map.append((form_idx, label))

    texts = engine.recognize(flat)
    if len(texts) != len(flat):
        raise RuntimeError(
            f"engine returned {len(texts)} predictions for {len(flat)} crops -- "
            f"cannot align results to fields"
        )

    results: List[dict] = [
        {label: "" for label in crops} for crops in crops_per_form
    ]
    for (form_idx, label), text in zip(index_map, texts):
        results[form_idx][label] = text

    return results
