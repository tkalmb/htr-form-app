"""htrpipe -- a configurable HTR pipeline for filled-in forms.

APP DISTRIBUTION NOTE
---------------------
This copy of htrpipe ships with the Streamlit application and is trimmed to
what the application actually runs (authorised by the author, 2026-08). The
following modules of the full thesis package were removed here:

* evaluation and statistics: ``evaluate``, ``metrics``, ``htrcompare``,
  ``compare_plots``, ``checkbox_bootstrap``, ``trocr_eval_adapter``
  (the app extracts; it never scores against ground truth)
* out-of-scope engines: ``dsocr`` (DeepSeek, separate conda env),
  ``surya_ocr``, ``recognize_vlm`` (crop-level VLM ablation)
* interactive/diagnostic tooling: ``roi_html``, ``parse_audit``,
  ``parse_diagnose``, ``formparse2`` (superseded by ``formparse``),
  ``config_additions`` (documentation snippet)

Added for the app: ``fewshot`` (in-context demonstrations for the whole-page
Qwen condition), trimmed to the functions the app uses; the kept functions'
logic is unchanged.

Nothing else that remains was modified beyond this file's imports and one
documented fix in ``postprocess_setup.schema_for_layout``. For the full
package, see the thesis repository.

Typical use::

    from htrpipe import (
        PipelineConfig, load_layout_spec, load_images,
        preprocess_pages, extract_crops, recognize_all, classify_checkboxes,
        build_predictions_frame, postprocess_predictions, PostprocessResources,
    )

Adapting to a different form means writing a new layout JSON: which fields
exist, where they are, what type each is, and which correction rule it needs.
No code changes.
"""

from .config import (
    CheckboxConfig,
    HTRConfig,
    LineSegmentConfig,
    PagePreprocessConfig,
    PipelineConfig,
    PostprocessConfig,
    RoiAlignConfig,
    RoiPreprocessConfig,
)
from .io_data import (
    FormImage,
    load_images,
    make_run_name,
    slugify,
    load_manifest_ids,
    load_predictions,
    load_split_ids,
    save_predictions,
)
from .schema import (
    FIELD_TYPES,
    POSTPROCESS_RULES,
    FieldSpec,
    LayoutSpec,
    layout_from_rois,
    load_layout_spec,
)
from .preprocess import (
    crop_to_text,
    extract_roi,
    preprocess_page,
    preprocess_roi,
    remove_lines,
    to_gray,
    to_rgb,
)
from .roi import (
    ROISelector,
    abs_to_rel,
    align_rois,
    alignment_report,
    rel_to_abs,
    show_crops,
    show_rois,
    check_interactive_backend,
    interactive_diagnostics,
)
from .segment import (
    merge_multiline_predictions,
    reassemble_field_crop,
    segment_lines,
    split_multiline_fields,
)
from .recognize import build_engine, engine_handles_multiline, recognize_forms
from .checkbox import CheckboxResult, checkbox_report, ink_ratio, select_marked_option
from .postprocess import (
    CommentLLM,
    PostprocessResources,
    diff_predictions,
    postprocess_predictions,
    validate_resources,
)
from .pipeline import (
    CropSet,
    build_predictions_frame,
    classify_checkboxes,
    extract_crops,
    preprocess_pages,
    recognize_all,
)

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig", "PagePreprocessConfig", "RoiPreprocessConfig",
    "RoiAlignConfig", "LineSegmentConfig", "CheckboxConfig", "HTRConfig",
    "PostprocessConfig",
    "FormImage", "load_images", "load_split_ids", "load_manifest_ids",
    "save_predictions", "load_predictions", "make_run_name", "slugify",
    "FieldSpec", "LayoutSpec", "load_layout_spec", "layout_from_rois",
    "FIELD_TYPES", "POSTPROCESS_RULES",
    "to_gray", "to_rgb", "preprocess_page", "preprocess_roi", "extract_roi",
    "remove_lines", "crop_to_text",
    "ROISelector", "align_rois", "alignment_report", "rel_to_abs", "abs_to_rel",
    "show_rois", "show_crops",
    "check_interactive_backend", "interactive_diagnostics",
    "segment_lines", "split_multiline_fields", "merge_multiline_predictions",
    "reassemble_field_crop",
    "build_engine", "recognize_forms", "engine_handles_multiline",
    "ink_ratio", "select_marked_option", "CheckboxResult", "checkbox_report",
    "PostprocessResources", "postprocess_predictions", "validate_resources",
    "diff_predictions", "CommentLLM",
    "CropSet", "preprocess_pages", "extract_crops", "recognize_all",
    "classify_checkboxes", "build_predictions_frame",
]
