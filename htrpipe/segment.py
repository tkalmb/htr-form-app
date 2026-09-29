"""Splitting multi-line field crops into single lines, and merging back.

TrOCR is a single-line recognizer: a crop containing two lines of handwriting
is not something it can transcribe correctly, so ``long_text`` fields are split
before recognition and re-joined afterwards.

Method: peaks in the row ink profile
------------------------------------
Each line of handwriting produces one dominant hump in the row-wise ink profile
-- the x-height zone, where most strokes are. The number of lines is therefore
the number of *prominent peaks*, and the boundaries are the minima between
them. Nothing here depends on knowing the size of the writing.

That last point is the whole design. Two earlier attempts failed on it:

* **Fixed-pixel projection thresholds** (``min_line_height``, ``gap_threshold``)
  cannot separate a descender fragment from a short line, because both are the
  same height; and no gap threshold both rejects descender bands and splits
  lines whose ascenders and descenders touch. The two requirements move the
  parameter in opposite directions.
* **Thresholds relative to an estimated character height** removed the pixel
  constants but replaced them with a single point of failure. On faint or
  speckled scans Otsu fragments the writing into hundreds of specks, so the
  median connected-component height measures noise rather than characters --
  measured at 5.5 px on a crop whose real character height was ~40 px. Every
  threshold being a multiple of that number, the whole algorithm collapsed at
  once: the split trigger fell to 11 px, the smoothing window to 1 px, and the
  junk-band filter to 2.5 px. It produced 9 lines on a 2-line crop.

Peak prominence and valley depth are both *ratios within the profile itself*,
so they are scale-free without estimating a scale. The discriminator that does
the real work is the valley test in :func:`_merge_shallow`: measured on real
crops, a genuine line boundary leaves a valley at or below ~0.12 of the smaller
adjacent peak, while the ascender/x-height humps *inside a single line* leave
one at ~0.6. The default sits between those, with a wide margin on both sides.

The original projection splitter is kept as ``segment_lines_projection`` and
selectable via ``cfg.method``, so the two can be compared on the same crops.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import LineSegmentConfig
from .preprocess import to_gray, upscale_to_min_height

Band = Tuple[int, int]  # (top, bottom), bottom exclusive


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------
#
# Settings are read with getattr defaults so this module works against an
# unmodified ``LineSegmentConfig``. Add them to the dataclass when you want
# them in ``cfg.describe()`` and therefore in the run manifest.

PARAM_DEFAULTS = {
    "method": "adaptive",              # "adaptive" | "projection" (original)
    # -- profile ----------------------------------------------------------
    "smooth_frac_of_height": 0.035,    # smoothing window as a fraction of crop height
    "ink_threshold_frac": 0.06,        # of the profile's robust max; used for trimming
    "min_ink_cols_frac": 0.008,        # absolute floor, fraction of crop width
    # -- peak detection ---------------------------------------------------
    "peak_min_prominence_frac": 0.15,  # of the profile's max
    "valley_max_frac": 0.35,           # merge peaks whose valley exceeds this
    "peak_min_distance_frac": 0.30,    # of the median peak spacing
    "max_lines": 10,                   # safety cap
    # -- boundaries -------------------------------------------------------
    "boundary_mode": "components",     # "components" | "seam" | "straight"
    "seam_wander_penalty": 0.03,       # keeps a seam tidy without forbidding detours
    "body_frac": 0.25,                 # of a peak's height; bounds the seam corridor
    "component_pad": 3,                # padding around each line's component bbox
    "straddle_frac": 1.0,              # >=1.0 disables geometric splitting of
                                       # merged components; see the note below
    "overlap_frac": 0.0,               # "straight" only: widen each window by
                                       # this fraction of the gap, both ways
    "trim_edges": False,               # trim the crop's outer rows to inked ones
    # -- output -----------------------------------------------------------
    "line_min_height": 0,              # per-line upscale target px; 0 disables
}


def _p(cfg: LineSegmentConfig, name: str):
    """Read a setting from ``cfg``, falling back to the module default."""
    value = getattr(cfg, name, None)
    return PARAM_DEFAULTS[name] if value is None else value


# ---------------------------------------------------------------------------
# Profile primitives
# ---------------------------------------------------------------------------

def _binarize(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask (255 = ink)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _has_usable_ink(binary: np.ndarray, lo: float = 0.0005, hi: float = 0.60) -> bool:
    """Whether a binarised crop plausibly contains text.

    Otsu always splits the histogram somewhere, so on a blank crop it marks
    scanner noise as ink and can return an almost fully "inked" mask -- the
    same failure ``crop_to_text`` avoids with a fixed cutoff. Here the profile
    *shape* is what is needed, so a fixed cutoff is not an option; rejecting
    implausible ink fractions is.
    """
    fraction = float(binary.mean()) / 255.0
    return lo < fraction < hi


def _row_profile(binary: np.ndarray) -> np.ndarray:
    """Ink pixels per row."""
    return binary.sum(axis=1).astype(np.float64) / 255.0


def _paint_mask(image: np.ndarray, mask: np.ndarray, fill: int = 255) -> np.ndarray:
    """Keep ``image`` where ``mask`` is True, fill elsewhere with ``fill``.

    ``mask`` is always 2-D (rows x cols). For a 2-D grayscale ``image`` this
    is a plain ``np.where``; for a 3-D colour ``image`` the mask is broadcast
    across the channel axis so the fill colour is applied uniformly rather
    than only to a single channel.
    """
    if image.ndim == 2:
        return np.where(mask, image, fill).astype(np.uint8)
    return np.where(mask[..., None], image, fill).astype(np.uint8)


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window) | 1)      # force odd
    if window <= 1 or window >= len(profile):
        return profile
    padded = np.pad(profile, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def _local_maxima(p: np.ndarray) -> List[int]:
    """Indices of local maxima, taking the first index of any plateau."""
    out = []
    n = len(p)
    for i in range(n):
        left = p[i - 1] if i else -np.inf
        right = p[i + 1] if i < n - 1 else -np.inf
        if p[i] >= left and p[i] > right:
            out.append(i)
    return out


def _prominence(p: np.ndarray, i: int) -> float:
    """Topographic prominence of the peak at ``i``.

    Walk outwards in both directions until a higher value or the end of the
    profile, tracking the minimum passed. The prominence is the peak's height
    above the higher of those two minima. Running off the end counts as a base
    of whatever minimum was seen, which is what makes the first and last lines
    of a crop detectable rather than clipped.
    """
    h = p[i]
    n = len(p)

    j, left_min = i - 1, h
    while j >= 0 and p[j] <= h:
        left_min = min(left_min, p[j])
        j -= 1

    j, right_min = i + 1, h
    while j < n and p[j] <= h:
        right_min = min(right_min, p[j])
        j += 1

    return float(h - max(left_min, right_min))


def _merge_shallow(p: np.ndarray, peaks: List[int], valley_max_frac: float) -> List[int]:
    """Merge adjacent peaks separated by a shallow valley.

    A single line of handwriting often produces two humps: one for the ascender
    zone, one for the x-height zone. They look like two lines to any method
    that only counts peaks. What separates them from a real line boundary is
    how far the profile falls between them, measured as a fraction of the
    smaller peak -- on real crops, ~0.6 within a line against <=0.12 across a
    boundary. Merging repeatedly on the shallowest-separated pair keeps the
    taller peak each time.
    """
    peaks = sorted(peaks)
    while len(peaks) > 1:
        ratios = [float(p[a:b + 1].min()) / min(p[a], p[b])
                  for a, b in zip(peaks, peaks[1:])]
        k = int(np.argmax(ratios))
        if ratios[k] <= valley_max_frac:
            break
        a, b = peaks[k], peaks[k + 1]
        peaks.remove(a if p[a] < p[b] else b)
    return peaks


def _enforce_spacing(p: np.ndarray, peaks: List[int], min_distance_frac: float) -> List[int]:
    """Drop peaks implausibly close together relative to the median line pitch.

    Self-calibrating: the pitch comes from the peaks already found, so this
    adds no length constant. Only applied with three or more peaks, since two
    peaks give no pitch to compare against.
    """
    if len(peaks) <= 2:
        return peaks
    pitch = float(np.median(np.diff(peaks)))
    if pitch <= 0:
        return peaks

    keep = [peaks[0]]
    for q in peaks[1:]:
        if q - keep[-1] < min_distance_frac * pitch:
            if p[q] > p[keep[-1]]:
                keep[-1] = q
        else:
            keep.append(q)
    return keep


# ---------------------------------------------------------------------------
# Curved boundaries (seam carving)
# ---------------------------------------------------------------------------

def _seam_between(binary: np.ndarray, y_top: int, y_bot: int,
                  y_straight: int, wander_penalty: float) -> np.ndarray:
    """Minimum-ink path across the crop between two rows, one row per column.

    A straight horizontal cut cannot separate two lines of handwriting whose
    descenders and ascenders interleave: wherever it is placed, it goes through
    a stroke. Cutting the "g" of *Werbung* in half loses the tail from its own
    line and strands a meaningless fragment in the next one, and the recognizer
    is hurt twice.

    A seam is free to move up or down one row per column, so it can pass around
    a descender and leave it attached to the line it belongs to. Cost is ink
    crossed, plus a small penalty for straying from the profile minimum -- with
    no penalty the seam takes long detours through blank space that separate
    nothing. This is the standard dynamic-programming line-separation approach
    (cf. A* / seam-based methods in the handwriting segmentation literature);
    verify the specific references before citing them.
    """
    band = binary[y_top:y_bot + 1].astype(np.float64) / 255.0
    n_rows, width = band.shape
    if n_rows < 3 or width < 2:
        return np.full(max(width, 1), y_straight, dtype=int)

    rows = np.arange(n_rows)
    penalty = wander_penalty * np.abs(rows - (y_straight - y_top))[:, None] / n_rows
    cost = band + penalty

    accum = np.empty_like(cost)
    back = np.zeros(cost.shape, dtype=np.int8)
    accum[:, 0] = cost[:, 0]

    for x in range(1, width):
        left = accum[:, x - 1]
        up = np.concatenate(([np.inf], left[:-1]))     # came from row-1
        down = np.concatenate((left[1:], [np.inf]))    # came from row+1
        options = np.vstack([up, left, down])
        choice = np.argmin(options, axis=0)
        accum[:, x] = cost[:, x] + options[choice, rows]
        back[:, x] = choice - 1

    y = int(np.argmin(accum[:, -1]))
    seam = np.empty(width, dtype=int)
    seam[-1] = y
    for x in range(width - 1, 0, -1):
        y = int(np.clip(y + back[y, x], 0, n_rows - 1))
        seam[x - 1] = y

    return seam + y_top


def _lines_from_seams(image: np.ndarray, binary: np.ndarray,
                      windows: List[Band], smoothed: np.ndarray,
                      cfg: LineSegmentConfig) -> List[np.ndarray]:
    """Cut ``image`` into line images along seams instead of straight rows.

    Each line keeps its full width; pixels belonging to a neighbouring line are
    set to white rather than cropped away, so the returned images are still
    plain rectangles the recognizer can consume. ``image`` keeps whatever
    channel count it came in with -- the seam geometry is computed from
    ``binary``/``smoothed`` (always single-channel), and only the final
    paint-and-crop touches ``image`` itself, via :func:`_paint_mask`.
    """
    h, w = image.shape[:2]
    penalty = float(_p(cfg, "seam_wander_penalty"))

    body_frac = float(_p(cfg, "body_frac"))
    peaks = [a + int(np.argmax(smoothed[a:b])) if b > a else a for a, b in windows]

    seams: List[np.ndarray] = []
    for i, ((_, a_bot), _) in enumerate(zip(windows, windows[1:])):
        peak_a, peak_b = peaks[i], peaks[i + 1]

        # Corridor: from the bottom of the upper line's *body* to the top of
        # the lower line's body, rather than peak to peak. The distinction
        # matters. Given a free choice between peaks, a minimum-ink seam
        # escapes a descender by passing *above* where it starts -- cheaper
        # than routing under it -- which severs the descender from its own line
        # even more thoroughly than a straight cut would. Starting the corridor
        # below the body forces the seam to commit: cross the stroke, or go
        # under it.
        lo = peak_a + 1
        while lo < peak_b and smoothed[lo] > body_frac * smoothed[peak_a]:
            lo += 1
        hi = peak_b - 1
        while hi > lo and smoothed[hi] > body_frac * smoothed[peak_b]:
            hi -= 1
        lo, hi = min(lo, h - 1), min(max(hi, lo + 2), h - 1)

        if hi <= lo:
            seams.append(np.full(w, a_bot, dtype=int))
            continue
        seams.append(_seam_between(binary, lo, hi, int(np.clip(a_bot, lo, hi)), penalty))

    uppers = [np.zeros(w, dtype=int)] + seams
    lowers = seams + [np.full(w, h, dtype=int)]

    rows = np.arange(h)[:, None]
    out: List[np.ndarray] = []
    for upper, lower in zip(uppers, lowers):
        mask = (rows >= upper[None, :]) & (rows < lower[None, :])
        if not mask.any():
            continue
        painted = _paint_mask(image, mask, 255)
        out.append(painted[int(upper.min()):int(lower.max())])
    return out


def _peak_bands(smoothed: np.ndarray, peaks: Sequence[int],
                body_frac: float, height: int) -> List[Band]:
    """The x-height band around each peak: contiguous rows above
    ``body_frac`` of that peak's height.

    These are the rows where a line's *writing* lives, as opposed to the gaps
    its ascenders and descenders reach into.
    """
    bands: List[Band] = []
    for pk in peaks:
        cutoff = body_frac * smoothed[pk]
        top = pk
        while top > 0 and smoothed[top - 1] > cutoff:
            top -= 1
        bottom = pk
        while bottom < height - 1 and smoothed[bottom + 1] > cutoff:
            bottom += 1
        bands.append((top, bottom + 1))
    return bands


def _lines_from_components(image: np.ndarray, binary: np.ndarray,
                           windows: List[Band], smoothed: np.ndarray,
                           cfg: LineSegmentConfig) -> List[np.ndarray]:
    """Cut into lines by assigning whole ink components, not rows.

    ``image`` keeps whatever channel count it came in with; component
    detection runs on ``binary``/``smoothed`` (always single-channel), and
    only the final paint-and-crop touches ``image`` itself.

    Any horizontal cut -- straight or curved -- has to answer "which side does
    this stroke belong to?" with a rule about *position*. That is the wrong
    question. The descender of a *g* belongs to the line its bowl belongs to,
    however far down it hangs, and the two are the same connected component.

    So each component is assigned to whichever line's profile peak is nearest
    its ink centroid, and a line's crop is everything assigned to it. A
    descender travels with its own glyph; an umlaut, a comma or a dot on an
    *i* travels with the nearest line rather than being clipped as low-ink
    noise; and no stroke is ever divided.

    The residual limitation is real and worth stating: when a descender
    physically *touches* a letter on the next line, the two become one
    component and the pair is assigned together. Nothing that cuts a raster
    image can separate them; that needs a recognizer that reads two lines at
    once. It is rarer than overlap, and it degrades to the same outcome a
    straight cut would give.
    """
    h, w = image.shape[:2]
    peaks = np.array([a + int(np.argmax(smoothed[a:b])) if b > a else a
                      for a, b in windows], dtype=float)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return [image[a:b] for a, b in windows]

    # Vote with *body* ink only. A component is assigned to the line whose
    # x-height band holds most of its ink, not to the peak nearest its
    # centroid. The distinction is what a flourish costs: a capital D whose
    # loop sweeps up into the line above has a centroid above that line's
    # peak, so a centroid rule hands the whole letter to the wrong line -- the
    # word below silently loses its first character. Restricting the vote to
    # the bands around each peak excludes the inter-line gap where flourishes
    # and descenders live, so only the writing's body decides, and an upward
    # flourish and a downward descender stop being mirror-image traps.
    bands = _peak_bands(smoothed, [int(pk) for pk in peaks],
                        float(_p(cfg, "body_frac")), h)
    band_of = np.full(h, -1, dtype=np.int32)
    for k, (b0, b1) in enumerate(bands):
        band_of[b0:b1] = k

    owner = np.zeros(n_labels, dtype=np.int32)
    ink_rows, ink_cols = np.nonzero(binary)
    ink_labels = labels[ink_rows, ink_cols]
    ink_bands = band_of[ink_rows]

    n_lines = len(windows)
    votes = np.zeros((n_labels, n_lines), dtype=np.int64)
    valid = ink_bands >= 0
    np.add.at(votes, (ink_labels[valid], ink_bands[valid]), 1)

    # Straddle detection uses the full line *windows*, not the bands. The two
    # answer different questions and need different row sets. Ownership asks
    # "whose writing is this?", so it must ignore the gap, where flourishes and
    # descenders live. Straddling asks "does this component contain writing
    # from two lines?", and a letter joined across the boundary often sits in
    # the gap rather than inside the next line's band -- a capital D touching
    # the descender above it does exactly that. Judged by bands, such a
    # component votes only for the upper line and never looks divided at all.
    window_of = np.zeros(h, dtype=np.int32)
    for k, (a, b) in enumerate(windows):
        window_of[a:b] = k
    win_votes = np.zeros((n_labels, n_lines), dtype=np.int64)
    np.add.at(win_votes, (ink_labels, window_of[ink_rows]), 1)

    line_of_row = window_of  # rows -> lines, for components that must be divided
    straddle_frac = float(_p(cfg, "straddle_frac"))
    straddling: List[int] = []
    unresolved: List[int] = []

    for i in range(1, n_labels):
        total = votes[i].sum()
        if not total:
            unresolved.append(i)
            owner[i] = -1
            continue
        order = np.argsort(votes[i])[::-1]
        win_total = win_votes[i].sum()
        win_order = np.argsort(win_votes[i])[::-1]
        win_minority = win_votes[i][win_order[1]] if len(win_order) > 1 else 0
        if win_total and win_minority >= straddle_frac * win_total:
            # Two lines' writing joined into one component -- the y of "lady"
            # touching "gold" below it, or a capital D touching the descender
            # above. Assigning the whole component to one line costs the other
            # line a letter outright, so the component is divided along the row
            # boundary instead.
            #
            # OFF BY DEFAULT (straddle_frac = 1.0), because it is not reliably
            # an improvement. When the intruding letter is written high, the
            # profile minimum sits *below* it: the component is correctly
            # identified as straddling, but the row cut still leaves the letter
            # on the wrong line and moves a fragment of the descender to the
            # other one -- two errors instead of one. It helps when the joined
            # letters sit at their own lines' normal heights and hurts when
            # they do not, and which case a crop falls into cannot be told from
            # the component alone. Lower to ~0.12 to enable and measure the
            # effect on CER before keeping it.
            straddling.append(i)
            owner[i] = int(order[0])
        else:
            owner[i] = int(order[0])

    # Components lying entirely between the bands: umlaut dots, the tittle of
    # an i, commas. Nearest peak gets these wrong whenever the mark is written
    # high, which for umlauts is most of the time. A diacritic belongs to the
    # letter it sits over, so inherit from the nearest ink in the same columns
    # instead, and fall back to nearest peak only when nothing overlaps.
    if unresolved:
        left = stats[:, cv2.CC_STAT_LEFT]
        right = left + stats[:, cv2.CC_STAT_WIDTH]
        top = stats[:, cv2.CC_STAT_TOP]
        bottom = top + stats[:, cv2.CC_STAT_HEIGHT]
        resolved = np.array([i for i in range(1, n_labels) if owner[i] >= 0])

        for i in unresolved:
            if resolved.size:
                overlaps = (right[resolved] > left[i]) & (left[resolved] < right[i])
                candidates = resolved[overlaps]
                if candidates.size:
                    gap = np.maximum(top[candidates] - bottom[i],
                                     top[i] - bottom[candidates])
                    owner[i] = int(owner[candidates[int(np.argmin(gap))]])
                    continue
            owner[i] = int(np.argmin(np.abs(centroids[i][1] - peaks)))

    line_of = owner[labels]
    if straddling:
        divided = np.isin(labels, straddling)
        rows_grid = np.broadcast_to(line_of_row[:, None], labels.shape)
        line_of = np.where(divided, rows_grid, line_of)
    pad = int(_p(cfg, "component_pad"))

    out: List[np.ndarray] = []
    for k in range(len(windows)):
        mask = (labels > 0) & (line_of == k)
        if not mask.any():
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        r0, r1 = max(0, rows[0] - pad), min(h, rows[-1] + pad + 1)
        c0, c1 = max(0, cols[0] - pad), min(w, cols[-1] + pad + 1)
        painted = _paint_mask(image, mask, 255)
        out.append(painted[r0:r1, c0:c1])
    return out


# ---------------------------------------------------------------------------
# Public: segmentation
# ---------------------------------------------------------------------------

def line_bounds(roi: np.ndarray, cfg: LineSegmentConfig) -> List[Band]:
    """Row windows ``(top, bottom)``, one per detected line, top to bottom.

    Windows tile the inked region contiguously: interior boundaries are the
    profile minimum between adjacent peaks, so every row belongs to exactly one
    line and no stroke can be cut in half. An ascender overlapping the line
    above is reassigned to a neighbour rather than truncated -- truncation is
    the expensive error for a recognizer, misassignment is cheap.

    Returns a single window for a one-line crop, and ``[]`` for an empty or
    ink-free crop.

    Detection-only: ``roi`` may be grayscale or colour, converted internally
    via :func:`to_gray`. The returned windows are row indices, so applying
    them to the original (possibly colour) crop is the caller's job -- see
    :func:`segment_lines`.
    """
    gray = to_gray(roi)
    if gray.size == 0:
        return []

    h, w = gray.shape[:2]
    binary = _binarize(gray)
    if not _has_usable_ink(binary):
        return []

    profile = _row_profile(binary)
    smoothed = _smooth(profile, round(_p(cfg, "smooth_frac_of_height") * h))

    peak_height = float(smoothed.max())
    if peak_height <= 0:
        return [(0, h)]

    peaks = [i for i in _local_maxima(smoothed)
             if _prominence(smoothed, i) >= _p(cfg, "peak_min_prominence_frac") * peak_height]
    if not peaks:
        return [(0, h)]

    peaks = _merge_shallow(smoothed, peaks, _p(cfg, "valley_max_frac"))
    peaks = _enforce_spacing(smoothed, peaks, _p(cfg, "peak_min_distance_frac"))

    if len(peaks) > _p(cfg, "max_lines"):
        # Not text, or badly cropped. One bad crop beats ten fragments.
        return [(0, h)]

    # Outer edges. Trimming is OFF by default, and that default matters: an
    # earlier version trimmed to rows whose *smoothed* profile cleared an ink
    # threshold, which silently deleted exactly the marks that carry the least
    # ink -- umlaut dots, descender tails, punctuation. The parent crop has
    # already been bounded by ``crop_to_text``, so keeping its full height
    # costs a few blank rows and cannot lose anything.
    top, bottom = 0, h
    if _p(cfg, "trim_edges"):
        raw_ink = np.flatnonzero(_row_profile(binary) > 0)
        pad = getattr(cfg, "pad", None)
        pad = 3 if pad is None else int(pad)
        if raw_ink.size:
            top = max(0, int(raw_ink[0]) - pad)
            bottom = min(h, int(raw_ink[-1]) + pad + 1)

    if len(peaks) == 1:
        return [(top, bottom)]

    overlap = float(_p(cfg, "overlap_frac"))
    edges = [top]
    for a, b in zip(peaks, peaks[1:]):
        edges.append(a + int(np.argmin(smoothed[a:b + 1])))
    edges.append(bottom)

    windows = [(edges[i], edges[i + 1]) for i in range(len(peaks))]
    if overlap > 0:
        grown = []
        for i, (a, b) in enumerate(windows):
            up = 0 if i == 0 else int(overlap * (b - a))
            dn = 0 if i == len(windows) - 1 else int(overlap * (b - a))
            grown.append((max(top, a - up), min(bottom, b + dn)))
        windows = grown
    return windows


def segment_lines(roi: np.ndarray, cfg: LineSegmentConfig) -> List[np.ndarray]:
    """Split ``roi`` into line crops, ordered top to bottom.

    Returns ``[roi]`` unchanged when zero or one line is found, so this stays a
    safe no-op on single-line fields.

    ``cfg.line_min_height`` optionally upscales each line crop. This is not the
    same as ``cfg.roi.min_height``, which ``preprocess_roi`` applies to the
    whole field crop and which therefore never fires on a multi-line comment --
    the parent is already taller than the target while its individual lines are
    not. Whether the upscale helps CER is an empirical question; leave it at 0
    until it has been measured on val.
    """
    if _p(cfg, "method") == "projection":
        return segment_lines_projection(roi, cfg)

    gray = to_gray(roi)
    windows = line_bounds(gray, cfg)
    if len(windows) <= 1:
        return [roi]

    mode = _p(cfg, "boundary_mode")
    if mode in ("components", "seam"):
        binary = _binarize(gray)
        smoothed = _smooth(_row_profile(binary),
                           round(_p(cfg, "smooth_frac_of_height") * gray.shape[0]))
        cutter = _lines_from_components if mode == "components" else _lines_from_seams
        # `roi`, not `gray`: gray/binary/smoothed are detection-only, derived
        # from roi purely to find where the lines are. What gets cut and
        # returned should be the crop as it actually looked going in.
        lines = cutter(roi, binary, windows, smoothed, cfg)
    else:
        lines = [roi[top:bottom] for top, bottom in windows]

    target = int(_p(cfg, "line_min_height"))
    out = []
    for crop in lines:
        if crop.size == 0:
            continue
        out.append(upscale_to_min_height(crop, target) if target else crop)

    return out or [roi]


def segment_lines_projection(roi: np.ndarray, cfg: LineSegmentConfig) -> List[np.ndarray]:
    """The original fixed-pixel projection splitter, behaviour unchanged.

    Retained so the two methods can be compared on identical crops. An ablation
    that reports both is worth more than a silent replacement: it is the only
    way to attribute a change in comment-field CER to segmentation rather than
    to the recognizer.
    """
    gray = to_gray(roi)
    if gray.size == 0:
        return [roi]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(binary, axis=1)

    max_projection = projection.max()
    if max_projection == 0:
        return [roi]

    text_rows = np.where(projection > max_projection * cfg.projection_threshold_frac)[0]
    if text_rows.size == 0:
        return [roi]

    gaps = np.where(np.diff(text_rows) > cfg.gap_threshold)[0]

    bounds: List[Tuple[int, int]] = []
    start = text_rows[0]
    for g in gaps:
        bounds.append((start, text_rows[g]))
        start = text_rows[g + 1]
    bounds.append((start, text_rows[-1]))

    bounds = [(s, e) for s, e in bounds if (e - s) >= cfg.min_line_height]
    if len(bounds) <= 1:
        return [roi]

    height = gray.shape[0]
    return [roi[max(0, s - cfg.pad):min(height, e + cfg.pad), :] for s, e in bounds]


# ---------------------------------------------------------------------------
# Public: split / merge (API unchanged)
# ---------------------------------------------------------------------------

def split_multiline_fields(
    crops: Dict[str, np.ndarray],
    cfg: LineSegmentConfig,
    fields: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Split the named fields of one form into per-line crops.

    Parameters
    ----------
    crops : ``{field: crop}`` for a single form.
    fields : which fields may span multiple lines. Pass the layout's
        ``multiline_fields``. ``None`` checks every field, which risks
        splitting a single-line field whose ascenders and descenders happen to
        leave a gap -- not recommended once the layout is known.

    Returns
    -------
    ``(new_crops, line_map)`` where split fields appear as ``"{field}_1"``,
    ``"{field}_2"``, ... and ``line_map`` records ``{field: n_lines}`` so
    :func:`merge_multiline_predictions` can put them back together without
    guessing from the key names.
    """
    new_crops: Dict[str, np.ndarray] = {}
    line_map: Dict[str, int] = {}
    wanted = None if fields is None else set(fields)

    for name, crop in crops.items():
        if wanted is not None and name not in wanted:
            new_crops[name] = crop
            continue

        lines = segment_lines(crop, cfg)
        if len(lines) > 1:
            line_map[name] = len(lines)
            for i, line in enumerate(lines, start=1):
                new_crops[f"{name}_{i}"] = line
        else:
            new_crops[name] = crop

    return new_crops, line_map


def merge_multiline_predictions(
    predictions: Dict[str, str],
    line_map: Dict[str, int],
    cfg: LineSegmentConfig,
) -> Dict[str, str]:
    """Re-join per-line predictions into one value per field.

    The separator comes from ``cfg.join_separator`` and must match whatever
    the normalization step assumes about multi-line values -- if the two
    disagree, every multi-line field is charged edit-distance errors it did
    not actually make.
    """
    merged = dict(predictions)
    for base_field, n_lines in line_map.items():
        parts = [merged.pop(f"{base_field}_{i}", "") for i in range(1, n_lines + 1)]
        merged[base_field] = cfg.join_separator.join(p.strip() for p in parts if p.strip())
    return merged


def reassemble_field_crop(crops: Dict[str, np.ndarray], field: str) -> np.ndarray:
    """Stack a field's line crops back into one image, for visual inspection.

    Only used for display. Line crops share the parent crop's width (splitting
    cuts rows, never columns) *unless* ``line_min_height`` upscaling was
    applied, which scales width too; widths are equalised here so the display
    path cannot raise on a config that recognition handles fine.
    """
    if field in crops:
        return crops[field]

    line_keys = sorted(
        (k for k in crops if k.startswith(f"{field}_")),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    if not line_keys:
        raise KeyError(f"no crop found for field {field!r}")

    images = [crops[k] for k in line_keys]
    width = max(im.shape[1] for im in images)
    padded = [
        im if im.shape[1] == width
        else cv2.copyMakeBorder(im, 0, 0, 0, width - im.shape[1],
                                cv2.BORDER_CONSTANT, value=255)
        for im in images
    ]
    return np.concatenate(padded, axis=0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def estimate_char_height(binary: np.ndarray) -> float:
    """Rough character height from ink connected components.

    **Diagnostic only -- nothing in the segmentation depends on this.** It was
    load-bearing in an earlier version and that was the defect: on faint or
    speckled scans Otsu fragments the writing, the median component height
    measures specks rather than characters, and every threshold derived from it
    fails together. Kept because it is useful for spotting such crops -- a
    value far below ``height / n_lines`` means the binarisation is fragmenting
    the writing, which is worth knowing even though the splitter now tolerates
    it.
    """
    h, w = binary.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    heights = []
    for i in range(1, n):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if ch < max(3, 0.02 * h) or area < 6:
            continue                      # speckle
        if ch > 0.85 * h:
            continue                      # border or full-height artefact
        if cw > 0.9 * w and ch < 4:
            continue                      # rule line that survived remove_lines
        heights.append(ch)

    if not heights:
        return max(6.0, h / 3.0)
    return float(np.median(heights))


def diagnose(roi: np.ndarray, cfg: LineSegmentConfig) -> dict:
    """Per-crop record for tuning without hand-labelling every crop.

    The unsupervised signals worth sweeping on:

    * ``weakest_line_share`` -- smallest fraction of the crop's ink landing in
      any one line. Near zero means a line was emitted holding almost nothing.
    * ``min_valley_ratio`` -- the *shallowest* accepted boundary, as a fraction
      of the smaller adjacent peak. Values approaching ``valley_max_frac`` mean
      the splitter is working near its decision boundary on that crop; a crop
      with a value around 0.3 or above is one to look at by eye.
    * ``looks_undersplit`` -- one line on a crop whose profile still shows
      several ink humps.
    """
    gray = to_gray(roi)
    binary = _binarize(gray)
    profile = _row_profile(binary)
    h = gray.shape[0]
    smoothed = _smooth(profile, round(_p(cfg, "smooth_frac_of_height") * h))
    windows = line_bounds(gray, cfg)

    total = float(profile.sum()) or 1.0
    masses = [float(profile[a:b].sum()) for a, b in windows]

    peak_height = float(smoothed.max())
    raw_peaks = ([i for i in _local_maxima(smoothed)
                  if _prominence(smoothed, i) >= _p(cfg, "peak_min_prominence_frac") * peak_height]
                 if peak_height > 0 else [])

    ratios = []
    if len(windows) > 1:
        for (a, _), (_, _) in zip(windows, windows[1:]):
            pass
        centres = [a + int(np.argmax(smoothed[a:b])) for a, b in windows]
        ratios = [float(smoothed[x:y + 1].min()) / min(smoothed[x], smoothed[y])
                  for x, y in zip(centres, centres[1:])]

    return {
        "n_lines": len(windows),
        "height": int(h),
        "windows": windows,
        "char_height_diag": round(estimate_char_height(binary), 1),
        "line_ink_share": [round(m / total, 4) for m in masses],
        "weakest_line_share": round(min(masses) / total, 4) if masses else 0.0,
        "min_valley_ratio": round(max(ratios), 3) if ratios else None,
        "n_raw_peaks": len(raw_peaks),
        "looks_undersplit": bool(len(windows) == 1 and len(raw_peaks) > 1),
        "no_usable_ink": not _has_usable_ink(binary),
    }


def plot_diagnosis(roi: np.ndarray, cfg: LineSegmentConfig, title: str = ""):
    """Crop with boundaries overlaid, beside the smoothed profile and peaks.

    The right panel is the thing to read when a split looks wrong: solid lines
    are accepted boundaries, dots are the detected peaks (one per line).
    """
    import matplotlib.pyplot as plt

    gray = to_gray(roi)
    binary = _binarize(gray)
    profile = _row_profile(binary)
    h = gray.shape[0]
    smoothed = _smooth(profile, round(_p(cfg, "smooth_frac_of_height") * h))
    windows = line_bounds(gray, cfg)
    centres = [a + int(np.argmax(smoothed[a:b])) for a, b in windows] if windows else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2),
                             gridspec_kw={"width_ratios": [3, 1]})
    axes[0].imshow(gray, cmap="gray")
    for top, bottom in windows:
        axes[0].axhline(top, color="tab:red", lw=1.0)
        axes[0].axhline(bottom, color="tab:red", lw=1.0, ls=":")
    axes[0].set_title(f"{title}  ({len(windows)} line(s))", fontsize=9, loc="left")
    axes[0].axis("off")

    axes[1].plot(smoothed, np.arange(len(smoothed)), lw=1.0)
    for c in centres:
        axes[1].plot(smoothed[c], c, "o", color="tab:green", ms=5)
    for top, _ in windows[1:]:
        axes[1].axhline(top, color="tab:red", lw=0.8)
    axes[1].invert_yaxis()
    axes[1].set_title("row ink profile", fontsize=9)
    plt.tight_layout()
    return fig
