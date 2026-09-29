"""Per-stage drivers that loop over forms.

Each function here does one stage for *all* forms and returns its intermediate
product, so the notebook can inspect every step rather than calling a single
opaque ``run()``. That matters for a thesis pipeline: when a number looks
wrong, you need to be able to look at the crops.

Stage order:

    load -> preprocess_page -> align ROIs -> extract + preprocess crops
         -> split long_text into lines -> recognize -> merge lines
         -> classify checkbox groups -> assemble frame -> post-process
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .checkbox import CheckboxResult, recognize_checkbox_group
from .config import PipelineConfig
from .io_data import FormImage
from .preprocess import extract_roi, preprocess_page, preprocess_roi
from .recognize import BaseEngine, recognize_forms
from .roi import AlignmentResult
from .schema import LayoutSpec
from .segment import merge_multiline_predictions, split_multiline_fields


@dataclass
class CropSet:
    """Crops for one form, at both stages of processing."""

    doc_id: object
    #: Field crops after ``preprocess_roi``, before line splitting.
    fields: Dict[str, np.ndarray] = dc_field(default_factory=dict)
    #: Same, but ``long_text`` fields replaced by ``"{field}_1"``, ``_2``, ...
    recognizer_inputs: Dict[str, np.ndarray] = dc_field(default_factory=dict)
    #: ``{field: n_lines}`` for fields that were split.
    line_map: Dict[str, int] = dc_field(default_factory=dict)
    #: Raw (unpreprocessed) crops of checkbox parents, kept for inspection.
    checkbox_parents: Dict[str, np.ndarray] = dc_field(default_factory=dict)


def preprocess_pages(forms: Sequence[FormImage], cfg: PipelineConfig,
                     progress: bool = False) -> List[np.ndarray]:
    """Full-page preprocessing for every loaded form."""
    pages = []
    for i, form in enumerate(forms):
        pages.append(preprocess_page(form.image, cfg.page))
        if progress and (i + 1) % 10 == 0:
            print(f"  preprocessed {i + 1}/{len(forms)} pages")
    return pages


def extract_crops(
    pages: Sequence[np.ndarray],
    alignments: Sequence[AlignmentResult],
    forms: Sequence[FormImage],
    layout: LayoutSpec,
    cfg: PipelineConfig,
    split_multiline: bool = True,
) -> List[CropSet]:
    """Cut and preprocess every field crop, splitting multi-line fields.

    ``ignore`` fields are still cropped (so they can be looked at) but are
    excluded from the recognizer inputs. ``checkbox_group`` parents are cropped
    *without* ``preprocess_roi``, because that function tightens each crop to
    its own content and would destroy the consistent framing the ink-ratio
    comparison depends on.

    ``split_multiline=False`` hands ``long_text`` fields to the recognizer
    whole. Pass ``not engine_handles_multiline(cfg.htr)``: line splitting
    exists because TrOCR is a single-line recognizer, and imposing it on a
    model without that limitation would both handicap the model and make the
    line segmenter a confound. With no fields split, ``line_map`` is empty and
    ``merge_multiline_predictions`` becomes a no-op, so nothing downstream
    needs to know which path was taken.
    """
    multiline = set(layout.multiline_fields) if split_multiline else set()
    checkbox_names = {f.name for f in layout.checkbox_groups}
    out: List[CropSet] = []

    for page, alignment, form in zip(pages, alignments, forms):
        crop_set = CropSet(doc_id=form.doc_id)

        for spec in layout.fields:
            box = alignment.rois.get(spec.name)
            if box is None:
                continue

            if spec.name in checkbox_names:
                crop_set.checkbox_parents[spec.name] = extract_roi(page, box)
                continue

            raw = extract_roi(page, box)
            crop_set.fields[spec.name] = preprocess_roi(raw, cfg.roi)

        recognizer_fields = {
            name: crop for name, crop in crop_set.fields.items()
            if layout[name].is_recognized
        }
        split, line_map = split_multiline_fields(recognizer_fields, cfg.lines, fields=multiline)
        crop_set.recognizer_inputs = split
        crop_set.line_map = line_map
        out.append(crop_set)

    return out


def recognize_all(
    crop_sets: Sequence[CropSet],
    engine: BaseEngine,
    cfg: PipelineConfig,
) -> List[Dict[str, str]]:
    """Recognize every crop and merge multi-line fields back together."""
    raw = recognize_forms([cs.recognizer_inputs for cs in crop_sets], engine)
    return [
        merge_multiline_predictions(prediction, crop_set.line_map, cfg.lines)
        for prediction, crop_set in zip(raw, crop_sets)
    ]


def classify_checkboxes(
    pages: Sequence[np.ndarray],
    alignments: Sequence[AlignmentResult],
    layout: LayoutSpec,
    cfg: PipelineConfig,
) -> Tuple[List[Dict[str, CheckboxResult]], List[Dict[str, Dict[str, np.ndarray]]]]:
    """Classify every checkbox group on every form.

    Returns ``(results_per_form, option_crops_per_form)``; the crops come back
    so the notebook can spot-check that each option box was isolated correctly,
    which is the usual cause of a bad ink-ratio comparison.
    """
    groups = layout.checkbox_groups
    results: List[Dict[str, CheckboxResult]] = []
    crops: List[Dict[str, Dict[str, np.ndarray]]] = []

    for page, alignment in zip(pages, alignments):
        per_form_results: Dict[str, CheckboxResult] = {}
        per_form_crops: Dict[str, Dict[str, np.ndarray]] = {}
        for spec in groups:
            box = alignment.rois.get(spec.name)
            if box is None:
                continue
            result, option_crops = recognize_checkbox_group(page, box, spec, cfg.checkbox)
            per_form_results[spec.name] = result
            per_form_crops[spec.name] = option_crops
        results.append(per_form_results)
        crops.append(per_form_crops)

    return results, crops


def build_predictions_frame(
    doc_ids: Sequence,
    text_predictions: Sequence[Dict[str, str]],
    checkbox_results: Optional[Sequence[Dict[str, CheckboxResult]]],
    layout: LayoutSpec,
    unresolved_checkbox: str = "",
):
    """Assemble one row per form, one column per output field.

    Column order follows the layout, so the CSV is stable across runs and
    across models -- a prerequisite for diffing two runs.

    ``unresolved_checkbox`` is what a ``no_mark``/``ambiguous`` group writes.
    The empty string keeps the CSV clean; the accompanying status report is
    where those cases are actually visible.
    """
    import pandas as pd

    columns = layout.output_columns
    rows = []

    for i, doc_id in enumerate(doc_ids):
        row = {column: "" for column in columns}
        for name, value in text_predictions[i].items():
            if name in row:
                row[name] = value
        if checkbox_results is not None:
            for name, result in checkbox_results[i].items():
                if name in row:
                    row[name] = result.selected if result.selected is not None else unresolved_checkbox
        rows.append(row)

    frame = pd.DataFrame(rows, columns=columns)
    frame.index = pd.Index(list(doc_ids), name="doc_id")
    return frame
