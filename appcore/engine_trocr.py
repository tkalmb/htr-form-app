"""TrOCR path: the htrpipe stages, called in the exact order the thesis
evaluation used. No stage is reimplemented here -- this module only
sequences existing functions and collects their diagnostics for the UI.

Stage order (identical to the evaluation):
    preprocess_pages -> reference ROIs (first page) -> align_rois
    -> extract_crops -> recognize_all -> classify_checkboxes
    -> build_predictions_frame

Failure handling note: the TrOCR stages are *batch* functions by design
(recognition runs with batch_size=8 across all crops, matching the evaluated
throughput setting). Per-page degradation is already built into htrpipe --
a failed alignment falls back per the config and is visible in
``alignment_report``; an empty crop yields an empty string. This module
therefore reports those diagnostics instead of re-wrapping every form in its
own try/except, which would change the evaluated batching behaviour.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from htrpipe import (
    PipelineConfig,
    align_rois,
    alignment_report,
    build_predictions_frame,
    checkbox_report,
    classify_checkboxes,
    engine_handles_multiline,
    extract_crops,
    preprocess_pages,
    recognize_all,
)


def make_roi_overlay(reference_image, rois: Dict[str, tuple]):
    """Matplotlib figure of the layout boxes on the reference page.

    Uses ``htrpipe.show_rois`` (the pipeline's own overlay function) and
    captures the figure it draws so Streamlit can render it. ``plt.show()``
    inside show_rois is a no-op on the app's non-interactive backend, so the
    figure is still alive to grab.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)   # headless-safe; no window is opened
    import matplotlib.pyplot as plt

    from htrpipe import show_rois

    show_rois(reference_image, rois, title="Field regions on the first page")
    return plt.gcf()


def run(forms, layout, cfg: PipelineConfig, engine,
        progress: Optional[Callable[[str], None]] = None):
    """Run the full TrOCR pipeline on ``forms``.

    Parameters
    ----------
    forms : list of ``htrpipe.FormImage``, as returned by ``load_images``.
    layout : ``LayoutSpec`` with the user-confirmed schema applied.
    cfg : ``PipelineConfig`` built by ``presets.build_pipeline_config``.
    engine : the recognition engine from ``models.load_trocr`` (cached).
    progress : optional callback taking a status string, for the UI.

    Returns a dict with the predictions frame and every diagnostic the
    evaluation inspects: alignment report, checkbox report and per-form
    checkbox statuses (the review step turns ``no_mark``/``ambiguous`` into
    advisory flags).
    """
    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    say(f"Preprocessing {len(forms)} page(s) ...")
    pages = preprocess_pages(forms, cfg)

    # The reference page is the FIRST page of the batch -- the evaluation's
    # convention. The layout's ROIs
    # are projected onto it, then aligned onto every other page.
    reference_image = pages[0]
    reference_rois = layout.reference_rois(reference_image)

    say("Aligning field regions to every page ...")
    alignments = align_rois(reference_image, reference_rois, pages, cfg.align)
    align_df = alignment_report(alignments, [f.doc_id for f in forms])

    say("Extracting and preprocessing field crops ...")
    # TrOCR is a single-line recognizer, so long_text fields are split into
    # lines for it -- but only for engines that need it, asked the same way
    # the evaluation asks.
    crop_sets = extract_crops(
        pages, alignments, forms, layout, cfg,
        split_multiline=not engine_handles_multiline(cfg.htr),
    )

    say("Recognising all field crops (batched) ...")
    raw_predictions = recognize_all(crop_sets, engine, cfg)

    checkbox_results = None
    checkbox_df = None
    if layout.checkbox_groups:
        say("Classifying checkbox groups ...")
        checkbox_results, _option_crops = classify_checkboxes(
            pages, alignments, layout, cfg)
        checkbox_df = checkbox_report(checkbox_results, [f.doc_id for f in forms])

    say("Assembling the predictions table ...")
    predictions_raw = build_predictions_frame(
        [f.doc_id for f in forms], raw_predictions, checkbox_results, layout)

    return {
        "predictions_raw": predictions_raw,
        "alignment_report": align_df,
        "checkbox_report": checkbox_df,
        "checkbox_results": checkbox_results,
        "reference_image": reference_image,
        "reference_rois": reference_rois,
    }
