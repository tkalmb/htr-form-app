"""Configuration objects for the HTR pipeline.

Every stage of the pipeline takes its settings as an explicit config object.
This replaces the module-level globals (``DENOISE_1``, ``CLAHE_2``,
``REMOVE_LINES``, ...) that the original notebooks relied on: those could not
survive a move into importable modules, because a function reading a global
only works if the caller happens to have defined that name first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


def _require_odd(value: int, name: str) -> None:
    if value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer (OpenCV kernel), got {value}")


@dataclass
class PagePreprocessConfig:
    """Full-page preprocessing, applied once per scanned form.

    Note on colour: ``preprocess_page`` always returns a 2-D grayscale array.
    The original notebook had a ``convert_to_gray=False`` branch, but it was
    unreachable in practice -- with deskewing enabled it raised
    ``ValueError: too many values to unpack`` on the ``(h, w) = img.shape``
    line, and it double-converted already-RGB input to BGR. It has been
    removed rather than ported.
    """

    denoise: bool = False
    denoise_kernel: int = 3

    clahe: bool = False
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    deskew: bool = True
    #: Skip rotation below this absolute angle (degrees) -- avoids resampling
    #: the whole page for a rotation too small to matter.
    deskew_min_angle: float = 1.0
    #: Re-measure skew after rotating and keep the original if it got worse.
    #: Costs one extra angle estimate per page; makes the stage self-checking.
    deskew_verify: bool = True

    def __post_init__(self) -> None:
        _require_odd(self.denoise_kernel, "denoise_kernel")


@dataclass
class RoiPreprocessConfig:
    """Per-field-crop preprocessing, applied to every extracted ROI."""

    clahe: bool = False
    clahe_clip_limit: float = 1.5
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    remove_lines: bool = True
    line_horizontal_kernel: Tuple[int, int] = (40, 1)
    line_vertical_kernel: Tuple[int, int] = (1, 40)
    line_dilate_kernel: Tuple[int, int] = (3, 3)
    line_iterations: int = 2

    denoise: bool = False
    denoise_kernel: int = 3

    #: Upscale crops shorter than this many pixels before recognition.
    min_height: int = 32

    binarize: bool = False

    crop_to_text: bool = True
    crop_denoise_kernel: int = 5
    crop_padding: int = 5
    #: Fixed threshold used to find the text bounding box. 200 = "darker than
    #: light grey", chosen to catch ink but not scanner background noise.
    crop_threshold: int = 200

    def __post_init__(self) -> None:
        _require_odd(self.denoise_kernel, "denoise_kernel")
        _require_odd(self.crop_denoise_kernel, "crop_denoise_kernel")


@dataclass
class LineSegmentConfig:
    """Horizontal-projection line splitting, used for ``long_text`` fields."""

    min_line_height: int = 8
    gap_threshold: int = 5
    projection_threshold_frac: float = 0.035
    pad: int = 2
    #: How per-line predictions are re-joined into one field value. Must match
    #: whatever the normalization step assumes.
    join_separator: str = " "


@dataclass
class RoiAlignConfig:
    """ORB + RANSAC homography alignment of reference ROIs onto each page."""

    enabled: bool = True
    orb_features: int = 5000
    max_matches: int = 500
    ransac_threshold: float = 5.0
    #: Below this many RANSAC inliers the homography is treated as unreliable.
    #: The original code had no such check and would silently produce garbage
    #: coordinates on a failed match.
    min_inliers: int = 20
    #: ``"warn"`` keeps the (untrusted) result and records it in the
    #: diagnostics; ``"raise"`` stops the run; ``"fallback"`` uses the
    #: unaligned reference ROIs for that page.
    on_low_inliers: str = "warn"

    def __post_init__(self) -> None:
        allowed = {"warn", "raise", "fallback"}
        if self.on_low_inliers not in allowed:
            raise ValueError(f"on_low_inliers must be one of {sorted(allowed)}")


@dataclass
class CheckboxConfig:
    """Ink-ratio checkbox classification.

    ``select_marked_checkbox`` is an argmax over sibling options: it
    structurally assumes **exactly one** option is marked per group. A form
    with independent (multi-select) checkboxes would need a different rule.
    """

    #: Below this ink fraction on the best candidate, report ``no_mark``.
    min_ink: float = 0.02
    #: Relative margin between best and second-best; below it, ``ambiguous``.
    margin_threshold: float = 0.15


@dataclass
class HTRConfig:
    """Text-recognition engine settings."""

    #: Line-level engines: ``"trocr"``, ``"tesseract"``, ``"easyocr"``.
    #: Crop-level VLMs (see :mod:`htrpipe.recognize_vlm`): ``"deepseek"``,
    #: ``"chandra"``, ``"qwen"``.
    engine: str = "trocr"
    model_path: str = "microsoft/trocr-base-handwritten"
    batch_size: int = 8
    #: TrOCR only. Set to 1 if the thesis claims greedy decoding across all
    #: conditions -- the VLM engines are greedy, so 4 here makes that claim
    #: false.
    num_beams: int = 4
    max_length: int = 128
    #: ``"auto"`` -> cuda if available, else cpu.
    device: str = "auto"
    #: EasyOCR only.
    languages: Tuple[str, ...] = ("de", "en")
    #: VLM engines only: how a multi-line VLM output is flattened into one
    #: field value. Must equal ``LineSegmentConfig.join_separator``, or a
    #: multi-line field is joined one way for TrOCR and another for the VLMs
    #: and is charged edit distance no model actually incurred.
    join_separator: str = " "

    #: Engine-specific settings (DeepSeek's ``vision_mode``, Chandra's
    #: ``min_pixels``) are deliberately NOT here. They are constructor kwargs
    #: passed through ``build_engine(cfg.htr, **kwargs)``, so a Chandra run's
    #: ``describe()`` does not list DeepSeek's vision modes. They are recorded
    #: instead in ``engine.meta``, which goes into the run manifest.


@dataclass
class PostprocessConfig:
    """Thresholds for the correction rules.

    Data the rules need (lexicons, domain lists, the confusion map, the LLM)
    lives in :class:`htrpipe.postprocess.PostprocessResources`. This holds only
    the numbers, so a run's tuning is visible in one place and lands in the run
    manifest.

    The ``*_max_relative_distance`` values are edit distance allowed per
    character of the input: ``0.25`` means a 12-character value may be
    corrected to a candidate up to 3 edits away. Lower is more conservative.
    """

    #: Nearest-neighbour cutoff for the ``lexicon`` rule (city names).
    lexicon_max_relative_distance: float = 0.25
    #: Cutoff for correcting an email domain to a known one.
    email_max_relative_distance: float = 0.34
    #: Cutoff for correcting a trailing street-type word.
    street_max_relative_distance: float = 0.34
    #: Cutoff for correcting a spelled-out month name.
    month_max_relative_distance: float = 0.34

    # -- LLM guardrail -----------------------------------------------------
    #: Reject if any aligned word pair differs by more than this many edits.
    llm_max_word_edit_distance: int = 2
    #: Reject if the word count changes by more than this.
    llm_max_word_count_diff: int = 2
    #: Reject if more than this fraction of words was touched.
    llm_max_changed_word_fraction: float = 0.3
    #: Always allow at least this many changed words, regardless of the
    #: fraction. Matters because comments are often 2-4 words long, where one
    #: legitimate fix already exceeds any pure fraction.
    llm_min_changed_word_allowance: int = 2
    #: Accept character-preserving word splits ("wirmeine" -> "wir meine").
    #: Off by default: see the note in ``accept_llm_correction``.
    llm_allow_word_splits: bool = False
    #: Generation cap for the correction model.
    llm_max_new_tokens: int = 64

    def guardrail_kwargs(self) -> dict:
        """The subset of settings :func:`accept_llm_correction` takes."""
        return {
            "max_word_edit_distance": self.llm_max_word_edit_distance,
            "max_word_count_diff": self.llm_max_word_count_diff,
            "max_changed_word_fraction": self.llm_max_changed_word_fraction,
            "min_changed_word_allowance": self.llm_min_changed_word_allowance,
            "allow_word_splits": self.llm_allow_word_splits,
        }


@dataclass
class PipelineConfig:
    """Everything the pipeline needs, in one object."""

    page: PagePreprocessConfig = field(default_factory=PagePreprocessConfig)
    roi: RoiPreprocessConfig = field(default_factory=RoiPreprocessConfig)
    align: RoiAlignConfig = field(default_factory=RoiAlignConfig)
    lines: LineSegmentConfig = field(default_factory=LineSegmentConfig)
    checkbox: CheckboxConfig = field(default_factory=CheckboxConfig)
    htr: HTRConfig = field(default_factory=HTRConfig)
    post: PostprocessConfig = field(default_factory=PostprocessConfig)

    def describe(self) -> str:
        """Flat text summary -- useful to paste into a run log or the thesis
        appendix so a reported number can be traced back to its settings."""
        lines_out = []
        for section, cfg in (
            ("page", self.page), ("roi", self.roi), ("align", self.align),
            ("lines", self.lines), ("checkbox", self.checkbox), ("htr", self.htr),
            ("post", self.post),
        ):
            for key, value in vars(cfg).items():
                lines_out.append(f"{section}.{key} = {value!r}")
        return "\n".join(lines_out)
