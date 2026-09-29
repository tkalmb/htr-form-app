"""Checkbox recognition by relative ink density.

No model is needed: within one group of mutually-exclusive options, the marked
box has more dark pixels than its siblings. The decision is therefore a
*comparison across siblings*, not an absolute threshold per box -- which is
what makes it robust to scan darkness varying between forms.

Binarisation is where that comparison can quietly break. Otsu always returns a
threshold, including for a crop containing nothing but paper noise: it splits
the noise distribution and reports a large "ink" fraction for an empty box.
Thresholding each option crop independently therefore inflates blank options,
sometimes past the marked one. The threshold is consequently computed **once
per group**, from the pooled pixels of all sibling crops, and applied to every
sibling -- so a blank box scores near zero and the ratios live on a common
scale, which is what the argmax assumes in the first place.

Two further consequences worth stating explicitly when writing this up:

* The rule is an argmax over options, so it **assumes exactly one option is
  marked per group**. Independent (multi-select) checkboxes would need a
  per-box absolute threshold instead, and a different confidence measure.
* Option crops are taken as fractions of the parent field crop and are *not*
  routed through ``preprocess_roi``. That function crops each ROI to a tight
  bounding box around its content, which is right for handwriting but would
  give the sibling boxes different framings and invalidate the comparison.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import CheckboxConfig
from .preprocess import extract_roi, to_gray
from .schema import FieldSpec

#: Status values reported alongside each prediction.
STATUS_OK = "ok"
STATUS_NO_MARK = "no_mark"
STATUS_AMBIGUOUS = "ambiguous"

#: Defaults for the binarisation knobs, used when ``CheckboxConfig`` does not
#: define them. Add the same names to that dataclass to make them configurable
#: per run; they are read via ``getattr`` so an older config still works.
_BINARISE_DEFAULTS = {
    "threshold_scope": "group",   # "group" (shared Otsu) or "per_option" (old)
    "denoise_ksize": 3,           # median blur before thresholding; 0 = off
    "open_ksize": 2,              # morphological opening after; 0 = off
    "min_contrast": 25.0,         # see _pooled_threshold
}


def _setting(cfg: CheckboxConfig, name: str):
    return getattr(cfg, name, _BINARISE_DEFAULTS[name])


# ---------------------------------------------------------------------------
# Binarisation
# ---------------------------------------------------------------------------

def _prepare(crop: np.ndarray, denoise_ksize: int) -> np.ndarray:
    """Grayscale + optional median blur. Median rather than Gaussian: scanner
    speckle is impulsive, and a median filter removes it without softening the
    edges of a pen stroke."""
    gray = to_gray(crop)
    if gray.size == 0:
        return gray
    if gray.dtype != np.uint8:
        # A float image in [0, 1] would make every grey-level threshold below
        # meaningless (and min_contrast unsatisfiable), so normalise first.
        gray = np.asarray(gray, dtype=np.float32)
        if gray.max() <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    if denoise_ksize and denoise_ksize >= 3:
        k = int(denoise_ksize) | 1                    # must be odd
        gray = cv2.medianBlur(gray, k)
    return gray


def _pooled_threshold(grays: Sequence[np.ndarray],
                      min_contrast: float) -> Optional[float]:
    """One Otsu threshold for a whole group of sibling crops.

    Returns ``None`` when the pooled pixels are too flat to contain ink at all
    -- i.e. the darkest pixels are within ``min_contrast`` of the paper level.
    That is the "nothing was ticked and there is no printed border inside the
    crops" case, where any threshold would just be cutting noise in half.

    Two degenerate inputs are handled explicitly, because Otsu does not fail on
    them -- it returns a misleading answer:

    * **Already-binarised crops.** With only two distinct grey values, every
      cut between them maximises Otsu's between-class variance equally, and
      OpenCV returns the lowest one (0). A threshold of 0 then matches only
      pure-black pixels, which is right if ink is 0 and catches *nothing* if
      the image polarity is inverted. The midpoint of the two modes is used
      instead, which is correct either way.
    * **Inverted polarity.** If more than half the pooled pixels end up on the
      ink side of the threshold, the crops are almost certainly white-on-black
      (or the ROI has slipped onto a dark region). Ratios computed from that
      are meaningless, so it warns rather than reporting a confident answer.
    """
    usable = [g.ravel() for g in grays if g.size]
    if not usable:
        return None

    pooled = np.concatenate(usable)

    # Checked before anything else: on a form crop the paper must dominate. If
    # the median pixel is dark, the crops are inverted (white ink on black) or
    # the ROI has slipped onto a dark region, and every number computed below
    # is meaningless. Without this check the low-contrast guard would return
    # None and the group would be reported as a plain `no_mark`.
    if float(np.median(pooled)) < 128:
        warnings.warn(
            f"checkbox crops are predominantly dark (median grey "
            f"{float(np.median(pooled)):.0f}). Expected dark ink on light "
            f"paper -- the crops are probably inverted, or the option ROIs "
            f"have slipped off the boxes. Ink ratios for this group are not "
            f"trustworthy.",
            stacklevel=3,
        )

    # The paper level can use a percentile -- paper is the overwhelming
    # majority. The dark level cannot: a tick mark is often well under 1% of
    # the pooled pixels (a 4-option group of generously drawn crops can put the
    # ink at ~0.3%), so percentile(1) returns paper and the contrast test
    # wrongly concludes the group is blank. The minimum is safe here precisely
    # because _prepare has already median-blurred away isolated speckle, and a
    # surviving dark speck only produces a tiny ink ratio that cfg.min_ink
    # rejects downstream.
    paper = float(np.percentile(pooled, 95))
    darkest = float(pooled.min())
    if paper - darkest < min_contrast:
        return None

    distinct = np.unique(pooled)
    if distinct.size <= 2:                            # already binarised
        threshold = float(distinct.mean()) if distinct.size == 2 else None
        if threshold is None:
            return None
    else:
        thr, _ = cv2.threshold(pooled.reshape(1, -1).astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        threshold = float(thr)
        # Otsu can still land on a tie at an extreme; fall back to the midpoint
        # between the two robust modes rather than trusting 0 or 255.
        if threshold <= 0.0 or threshold >= 254.0:
            threshold = (paper + darkest) / 2.0

    if float((pooled <= threshold).mean()) > 0.5:
        warnings.warn(
            f"checkbox crops: {float((pooled <= threshold).mean()):.0%} of "
            f"pooled pixels fall on the ink side of threshold {threshold:.0f}. "
            f"Ink should be a small minority -- the crops are probably "
            f"inverted (white ink on black) or the ROI has slipped onto a dark "
            f"region. Ink ratios from this group are not trustworthy.",
            stacklevel=3,
        )
    return threshold


def _mask_from_threshold(gray: np.ndarray, threshold: float,
                         open_ksize: int) -> np.ndarray:
    mask = ((gray <= threshold).astype(np.uint8)) * 255
    if open_ksize and open_ksize >= 2:
        k = int(open_ksize)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return mask


def binarize(crop: np.ndarray, threshold: Optional[float] = None,
             denoise_ksize: int = 3, open_ksize: int = 2) -> np.ndarray:
    """Ink mask for one crop: 255 where ink, 0 where paper.

    ``threshold=None`` means per-crop Otsu. Pass a shared threshold when
    comparing sibling boxes -- see the module docstring for why that matters.
    Useful on its own for eyeballing what the classifier actually sees.
    """
    gray = _prepare(crop, denoise_ksize)
    if gray.size == 0:
        return gray

    if threshold is None:
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if open_ksize and open_ksize >= 2:
            k = int(open_ksize)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        return mask

    return _mask_from_threshold(gray, threshold, open_ksize)


def ink_ratio(crop: np.ndarray, threshold: Optional[float] = None,
              denoise_ksize: int = 0, open_ksize: int = 0) -> float:
    """Fraction of pixels classified as ink.

    The defaults reproduce the original behaviour exactly (per-crop Otsu, no
    denoising), so existing calls and any already-computed numbers are
    unchanged. :func:`select_marked_option` passes a group-shared threshold
    instead; that is the path a decision should go through.
    """
    if to_gray(crop).size == 0:
        return 0.0
    mask = binarize(crop, threshold=threshold,
                    denoise_ksize=denoise_ksize, open_ksize=open_ksize)
    return float((mask > 0).mean())


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class CheckboxResult:
    """Outcome for one checkbox group on one form."""

    selected: Optional[str]
    status: str
    ratios: Dict[str, float]
    threshold: Optional[float] = None     # the shared cut actually used

    @property
    def margin(self) -> float:
        """Relative gap between the best and second-best option."""
        if len(self.ratios) < 2:
            return float("nan")
        top, second = sorted(self.ratios.values(), reverse=True)[:2]
        return (top - second) / (top + 1e-6)


def option_ink_ratios(option_crops: Dict[str, np.ndarray],
                      cfg: CheckboxConfig) -> Tuple[Dict[str, float], Optional[float]]:
    """Ink ratio per option, on a common scale. Returns ``(ratios, threshold)``.

    Exposed separately so the threshold and the ratios can be inspected on the
    validation set without invoking the decision rule.
    """
    denoise = _setting(cfg, "denoise_ksize")
    opening = _setting(cfg, "open_ksize")
    scope = _setting(cfg, "threshold_scope")

    if scope == "per_option":                          # original behaviour
        ratios = {name: ink_ratio(crop, None, denoise, opening)
                  for name, crop in option_crops.items()}
        return ratios, None

    if scope != "group":
        raise ValueError(
            f"threshold_scope must be 'group' or 'per_option', got {scope!r}")

    names = list(option_crops)
    grays = [_prepare(option_crops[n], denoise) for n in names]
    threshold = _pooled_threshold(grays, _setting(cfg, "min_contrast"))

    if threshold is None:                              # nothing dark anywhere
        return {n: 0.0 for n in names}, None

    ratios = {}
    for name, gray in zip(names, grays):
        if gray.size == 0:
            ratios[name] = 0.0
            continue
        ratios[name] = float((_mask_from_threshold(gray, threshold, opening) > 0).mean())
    return ratios, threshold


def select_marked_option(
    option_crops: Dict[str, np.ndarray],
    cfg: CheckboxConfig,
) -> CheckboxResult:
    """Pick the marked option, or report why no confident pick was possible.

    ``no_mark``   -- even the best candidate has too little ink (field left
    blank, or the crop missed the boxes entirely).
    ``ambiguous`` -- the top two are too close to separate; flag for review
    rather than guessing.
    """
    if len(option_crops) < 2:
        raise ValueError("a checkbox group needs at least 2 options to compare")

    ratios, threshold = option_ink_ratios(option_crops, cfg)
    ordered = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
    (best_name, best_ratio), (_, second_ratio) = ordered[0], ordered[1]

    if best_ratio < cfg.min_ink:
        return CheckboxResult(None, STATUS_NO_MARK, ratios, threshold)

    margin = (best_ratio - second_ratio) / (best_ratio + 1e-6)
    if margin < cfg.margin_threshold:
        return CheckboxResult(None, STATUS_AMBIGUOUS, ratios, threshold)

    return CheckboxResult(best_name, STATUS_OK, ratios, threshold)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def crop_options(
    page: np.ndarray,
    parent_box: Sequence[float],
    spec: FieldSpec,
) -> Dict[str, np.ndarray]:
    """Cut each option out of the parent field crop.

    Option coordinates are fractions of the parent crop, deliberately: a
    checkbox-sized region has too little texture for its own reliable ORB
    homography, while the parent field box is already aligned per form, so a
    proportional sub-crop of it is both simpler and more robust.
    """
    parent = extract_roi(page, parent_box)
    if parent.size == 0:
        return {name: parent for name in spec.options}

    h, w = parent.shape[:2]
    crops: Dict[str, np.ndarray] = {}
    for name, (rx1, ry1, rx2, ry2) in spec.options.items():
        x1, x2 = int(rx1 * w), int(rx2 * w)
        y1, y2 = int(ry1 * h), int(ry2 * h)
        crops[name] = parent[max(0, y1):y2, max(0, x1):x2]
    return crops


def recognize_checkbox_group(
    page: np.ndarray,
    parent_box: Sequence[float],
    spec: FieldSpec,
    cfg: CheckboxConfig,
) -> Tuple[CheckboxResult, Dict[str, np.ndarray]]:
    """Classify one checkbox group; returns the result and the crops used."""
    crops = crop_options(page, parent_box, spec)
    return select_marked_option(crops, cfg), crops


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def checkbox_report(results_per_form: Sequence[Dict[str, CheckboxResult]], doc_ids: Sequence):
    """Long-format table of every group's status and ink ratios.

    Kept separate from the predictions CSV because ``status``, the raw ratios
    and the threshold are diagnostics, not field values -- but they are exactly
    what you need to justify a threshold choice or explain a flagged form.
    """
    import pandas as pd

    rows = []
    for doc_id, per_form in zip(doc_ids, results_per_form):
        for field_name, result in per_form.items():
            row = {
                "doc_id": doc_id,
                "field": field_name,
                "selected": result.selected,
                "status": result.status,
                "margin": round(result.margin, 4),
                "threshold": (round(result.threshold, 1)
                              if result.threshold is not None else None),
            }
            row.update({f"ink_{opt}": round(value, 4) for opt, value in result.ratios.items()})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["doc_id", "field"])


def separability_report(report, truth: Dict, field: str = "studies"):
    """Do the marked and unmarked ink ratios actually separate?

    ``report`` -- the DataFrame from :func:`checkbox_report`
    ``truth``  -- ``{doc_id: option_name}`` ground truth for ``field``

    What you want is the marked ``min`` clearly above the unmarked ``max``. If
    those overlap, no ``min_ink`` or ``margin_threshold`` setting will fix it
    and the option geometry needs re-measuring. Run this on **validation**.
    """
    import pandas as pd

    sub = report.xs(field, level="field")
    ink_cols = [c for c in sub.columns if c.startswith("ink_")]

    rows = []
    for doc_id, row in sub.iterrows():
        marked = truth.get(doc_id)
        for col in ink_cols:
            option = col[len("ink_"):]
            rows.append({"doc_id": doc_id, "option": option,
                         "ratio": row[col], "is_marked": option == marked})

    long = pd.DataFrame(rows)
    summary = long.groupby("is_marked")["ratio"].describe()[
        ["count", "min", "25%", "50%", "75%", "max"]]

    marked = long.loc[long.is_marked, "ratio"]
    unmarked = long.loc[~long.is_marked, "ratio"]
    if len(marked) and len(unmarked):
        if marked.min() <= unmarked.max():
            print(f"OVERLAP: marked min {marked.min():.4f} <= "
                  f"unmarked max {unmarked.max():.4f} -- geometry problem, "
                  f"not a threshold problem")
        else:
            print(f"clean separation: unmarked max {unmarked.max():.4f} < "
                  f"marked min {marked.min():.4f}")
    return summary, long
