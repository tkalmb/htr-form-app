"""Named, end-to-end preprocessing presets.

The app deliberately does not expose individual preprocessing switches. It
offers a small set of *named* presets, each mapping to a complete
``htrpipe.PipelineConfig``. Two reasons:

1. Provenance. The thesis evaluation ran specific configurations; the app
   marks exactly those as "(evaluated)" and records in the run manifest
   whether the chosen configuration matches an evaluated one.
2. Coherence. Free switch-combinations allow states that are not merely
   untested but broken -- most obviously CLAHE applied twice (once on the
   page, once on the field crop), which flattens contrast. The preset lists
   below cannot express that state, and ``valid_field_presets`` enforces it.

The "(evaluated)" values are transcribed from the thesis run manifests
(run_A_TrOCR-scads-finetuned-100_test_..._pipeline.json for TrOCR; the VLM
manifests show the page-level runs consumed the raw scans, i.e. preset
"none"). Do not edit them without re-running the evaluation.
"""

from __future__ import annotations

from typing import Dict, List

from htrpipe import PipelineConfig

# ---------------------------------------------------------------------------
# Preset names (shown verbatim in the UI)
# ---------------------------------------------------------------------------

PAGE_NONE = "none"
PAGE_DESKEW = "deskew only"
PAGE_FULL = "light denoise + CLAHE + deskew"

FIELD_NONE = "none"
FIELD_LINES_CROP = "remove lines + crop"
FIELD_FULL = "remove lines + crop + denoise + CLAHE"

PAGE_PRESETS: List[str] = [PAGE_NONE, PAGE_DESKEW, PAGE_FULL]
FIELD_PRESETS: List[str] = [FIELD_NONE, FIELD_LINES_CROP, FIELD_FULL]

#: The standard configuration per model kind (author's decision): the
#: whole-page models consume the raw scans, exactly as their thesis runs did;
#: only the TrOCR pipeline deskews. Deskewing remains selectable for the
#: VLMs, but as a deviation from the standard, recorded in the manifest.
EVALUATED_PAGE_PRESET: Dict[str, str] = {
    "trocr": PAGE_DESKEW,
    "qwen": PAGE_NONE,
    "chandra": PAGE_NONE,
}
EVALUATED_FIELD_PRESET: Dict[str, str] = {
    "trocr": FIELD_LINES_CROP,  # roi.remove_lines=True, crop_to_text=True,
                                # denoise=False, clahe=False
}


def valid_field_presets(page_preset: str) -> List[str]:
    """Field presets that are coherent with the chosen page preset.

    The only incoherent combination is CLAHE at both levels: contrast
    equalisation applied to an already-equalised image. That combination is
    removed from the choices rather than warned about -- an invalid state
    should not be selectable at all.
    """
    if page_preset == PAGE_FULL:
        return [p for p in FIELD_PRESETS if p != FIELD_FULL]
    return list(FIELD_PRESETS)


def build_pipeline_config(page_preset: str, field_preset: str,
                          trocr_model_path: str) -> PipelineConfig:
    """Translate two preset names into a complete ``PipelineConfig``.

    Starts from ``PipelineConfig()`` defaults -- which already equal the
    evaluated settings for everything the presets do not touch (alignment,
    line segmentation, checkbox thresholds, ROI crop parameters) -- and then
    applies only the toggles the chosen presets control.

    The HTR block is set to the evaluated TrOCR decoding settings from the
    run manifest: batch_size=8, num_beams=1 (greedy, matching the VLMs),
    max_length=128. ``num_beams`` must be set explicitly because the htrpipe
    default is 4.
    """
    if page_preset not in PAGE_PRESETS:
        raise ValueError(f"unknown page preset: {page_preset!r}")
    if field_preset not in FIELD_PRESETS:
        raise ValueError(f"unknown field preset: {field_preset!r}")
    if field_preset not in valid_field_presets(page_preset):
        raise ValueError(
            f"field preset {field_preset!r} is not valid with page preset "
            f"{page_preset!r} (CLAHE would be applied twice)"
        )

    cfg = PipelineConfig()

    # ---- page level -----------------------------------------------------
    cfg.page.denoise = page_preset == PAGE_FULL
    cfg.page.clahe = page_preset == PAGE_FULL
    cfg.page.deskew = page_preset in (PAGE_DESKEW, PAGE_FULL)

    # ---- field-crop level (TrOCR path only) -----------------------------
    cfg.roi.remove_lines = field_preset in (FIELD_LINES_CROP, FIELD_FULL)
    cfg.roi.crop_to_text = field_preset in (FIELD_LINES_CROP, FIELD_FULL)
    cfg.roi.denoise = field_preset == FIELD_FULL
    cfg.roi.clahe = field_preset == FIELD_FULL

    # ---- recognition engine (evaluated settings) ------------------------
    cfg.htr.engine = "trocr"
    cfg.htr.model_path = trocr_model_path
    cfg.htr.batch_size = 8
    cfg.htr.num_beams = 1
    cfg.htr.max_length = 128
    cfg.htr.device = "auto"

    return cfg


def config_matches_evaluated(model_kind: str, page_preset: str,
                             field_preset: str = "") -> bool:
    """True when the chosen presets equal the thesis-evaluated configuration.

    Recorded in the run manifest so any exported result can be traced to
    whether it came from a measured configuration. For the VLMs the field
    preset does not apply and is ignored.
    """
    if EVALUATED_PAGE_PRESET.get(model_kind) != page_preset:
        return False
    if model_kind == "trocr":
        return EVALUATED_FIELD_PRESET["trocr"] == field_preset
    return True
