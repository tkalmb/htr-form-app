"""Locating field regions on each form.

Two ways to get *reference* ROIs (pick one in the notebook):

1. ``layout`` -- read them from a layout JSON (:mod:`htrpipe.schema`)
2. ``draw``   -- drag them once with :class:`ROISelector` (needs a working
   ipympl canvas), or with :func:`htrpipe.roi_html.roi_picker` (needs nothing)
3. ``paste``  -- a literal ``{field: (x1, y1, x2, y2)}`` dict in the notebook

All three produce the same thing: a dict of boxes on one reference page.
:func:`align_rois` then transfers those boxes onto every other page via ORB
feature matching + a RANSAC homography.

The docTR "automatic" detection path from the original notebook is **not**
included. It produced unlabelled word boxes with no mapping to field names,
so it could not feed a field-typed pipeline; it was marked WIP in the source
notebook and is better kept as a separate experiment than presented here as an
interchangeable option.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import RoiAlignConfig
from .preprocess import to_gray

Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def rel_to_abs(rois_rel: Dict[str, Sequence[float]], image: np.ndarray) -> Dict[str, Box]:
    """Fractions (0..1) -> pixel coordinates for ``image``."""
    h, w = image.shape[:2]
    return {
        name: (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        for name, (x1, y1, x2, y2) in rois_rel.items()
    }


def abs_to_rel(rois_abs: Dict[str, Sequence[float]], image: np.ndarray) -> Dict[str, Tuple[float, ...]]:
    """Pixel coordinates -> fractions (0..1) of ``image``. Inverse of
    :func:`rel_to_abs`. Use this to turn clicked boxes into resolution-
    independent layout entries."""
    h, w = image.shape[:2]
    return {
        name: (x1 / w, y1 / h, x2 / w, y2 / h)
        for name, (x1, y1, x2, y2) in rois_abs.items()
    }


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------
#
# This path depends on ipympl rendering a live canvas widget inside JupyterLab,
# which is the fragile part -- not the code below. If the figure appears but
# does not respond to dragging, you are almost certainly looking at a static
# PNG that matplotlib_inline rendered on top of (or instead of) the widget.
# Run `interactive_diagnostics()` to tell the two apart, and fall back to
# htrpipe.roi_html.roi_picker, which needs no interactive backend at all.


def interactive_diagnostics() -> dict:
    """Report why interactive drawing is or is not going to work.

    The failure this exists for is silent: `%matplotlib widget` only touches
    the *kernel*, so it succeeds even when the JupyterLab frontend cannot
    render the canvas -- and a figure still appears, because the inline
    backend's post-execute hook draws it as a PNG. A PNG looks identical to a
    live canvas and swallows every mouse event.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    info = {"backend": matplotlib.get_backend()}
    print(f"backend                : {info['backend']}")

    for pkg in ("ipympl", "ipywidgets", "matplotlib"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            info[pkg] = None
        print(f"{pkg:<23}: {info[pkg]}")

    fig = plt.figure()
    canvas = type(fig.canvas).__name__
    plt.close(fig)
    info["canvas_class"] = canvas
    print(f"canvas class           : {canvas}")

    live = "ipympl" in type(fig.canvas).__module__ or canvas.startswith("Canvas")
    info["interactive"] = bool(live)

    # A leftover flush_figures hook re-renders every figure as an inline PNG
    # after the cell finishes, which is what makes a live canvas look dead.
    hooks = []
    try:
        ip = get_ipython()                                    # noqa: F821
        hooks = [getattr(f, "__name__", str(f))
                 for f in ip.events.callbacks.get("post_execute", [])]
    except Exception:
        pass
    info["post_execute_hooks"] = hooks
    stale = [h for h in hooks if "flush_figures" in h]
    if stale:
        print(f"post_execute hooks     : {hooks}")
        print("  ^ matplotlib_inline's flush_figures is still attached. It "
              "re-renders figures as static PNGs, which cannot be dragged on. "
              "Restart the kernel and run `%matplotlib widget` before "
              "importing pyplot anywhere.")
    if not live:
        print("\nNot an interactive canvas -> ROISelector will display a dead "
              "image. Use htrpipe.roi_html.roi_picker instead.")
    return info


class ROISelector:
    """Drag one box per field on a displayed image.

    Requires a working ipympl canvas: run ``%matplotlib widget`` in its own
    cell *before* constructing this, and ``%matplotlib inline`` afterwards to
    restore normal plotting.

    **Drag a rectangle** for each name in ``field_names``, in order -- a single
    click is discarded as a stray (``minspanx``/``minspany``). Read the result
    from :attr:`rois`, or :meth:`as_relative` for layout-JSON coordinates.

    If nothing responds to the mouse, do not debug this class: run
    :func:`interactive_diagnostics` and switch to
    :func:`htrpipe.roi_html.roi_picker`.
    """

    def __init__(self, field_names: Sequence[str], image: np.ndarray,
                 figsize=(10, 7), check_backend: bool = True):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
        from IPython.display import display

        if check_backend:
            check_interactive_backend()

        self.field_names = list(field_names)
        self.image = image
        self.rois: Dict[str, Box] = {}
        self.current_index = 0

        # plt.ioff() around figure creation is the documented ipympl idiom: it
        # stops the figure being auto-shown a second time as an inline PNG,
        # which is the usual reason a canvas "renders but ignores the mouse".
        was_interactive = plt.isinteractive()
        plt.ioff()
        try:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        finally:
            if was_interactive:
                plt.ion()

        for attr, value in (("header_visible", False),
                            ("toolbar_visible", True),
                            ("footer_visible", True)):
            try:                              # ipympl-only canvas attributes
                setattr(self.fig.canvas, attr, value)
            except Exception:
                pass

        self.ax.imshow(self.image, cmap="gray" if self.image.ndim == 2 else None)
        self.ax.set_title(self._status_text())
        self.ax.axis("off")

        # `rectprops` was renamed to `props` in matplotlib 3.5.
        style = dict(facecolor="lime", edgecolor="lime", alpha=0.2, fill=True)
        key = "props" if _mpl_at_least(matplotlib.__version__, (3, 5)) else "rectprops"

        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=False,          # blitting hides the rubber band on ipympl
            button=[1],
            minspanx=5, minspany=5, spancoords="pixels",
            interactive=False,
            **{key: style},
        )

        self.fig.tight_layout()
        display(self.fig.canvas)    # explicit; do not rely on plt.show()

    # -- internals ---------------------------------------------------------

    def _status_text(self) -> str:
        if self.current_index < len(self.field_names):
            last = list(self.rois)[-1:]
            tail = f"   (last: {last[0]}={self.rois[last[0]]})" if last else ""
            return (f"Drag ROI {self.current_index + 1}/{len(self.field_names)}: "
                    f"'{self.field_names[self.current_index]}'{tail}")
        return f"All {len(self.field_names)} ROIs selected."

    def _draw_box(self, name: str, box: Box) -> None:
        import matplotlib.pyplot as plt

        self.ax.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            edgecolor="lime", facecolor="none", linewidth=2))
        self.ax.text(box[0] + 4, box[1] + 4, name, color="lime",
                     fontsize=10, fontweight="bold", va="top")

    def _on_select(self, eclick, erelease) -> None:
        if self.current_index >= len(self.field_names):
            return
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return                            # drag ended outside the axes

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        box = (int(round(min(x1, x2))), int(round(min(y1, y2))),
               int(round(max(x1, x2))), int(round(max(y1, y2))))

        name = self.field_names[self.current_index]
        self.rois[name] = box
        self._draw_box(name, box)

        self.current_index += 1
        # Status goes in the title, not print(): stdout emitted from a canvas
        # event callback does not reliably reach the cell output under ipympl.
        self.ax.set_title(self._status_text())
        if self.current_index >= len(self.field_names):
            self.selector.set_active(False)
        self.fig.canvas.draw_idle()

    def _repaint(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray" if self.image.ndim == 2 else None)
        self.ax.axis("off")
        for name, box in self.rois.items():
            self._draw_box(name, box)
        self.ax.set_title(self._status_text())
        self.selector.set_active(True)
        self.fig.canvas.draw_idle()

    # -- public ------------------------------------------------------------

    def undo(self) -> None:
        """Drop the most recent box and step back to that field."""
        if not self.rois:
            return
        name = list(self.rois)[-1]
        del self.rois[name]
        self.current_index = self.field_names.index(name)
        self._repaint()

    def reset(self) -> None:
        self.rois = {}
        self.current_index = 0
        self._repaint()

    @property
    def complete(self) -> bool:
        return len(self.rois) == len(self.field_names)

    @property
    def missing(self) -> List[str]:
        """Fields still without a box -- use this in the assertion message."""
        return [n for n in self.field_names if n not in self.rois]

    def as_relative(self) -> Dict[str, Tuple[float, ...]]:
        """Drawn boxes as fractions -- paste this into a layout JSON."""
        return abs_to_rel(self.rois, self.image)


def check_interactive_backend(raise_on_fail: bool = True) -> str:
    """Raise unless an interactive backend is active.

    Cheap guard so a dead image fails loudly instead of producing
    ``only 0/13 fields were clicked`` two cells later.
    """
    import matplotlib

    backend = matplotlib.get_backend()
    if "inline" in backend.lower() or backend.lower() in ("agg", "template"):
        message = (
            f"matplotlib backend is {backend!r}, which cannot deliver mouse "
            f"events. Run `%matplotlib widget` in its own cell above and "
            f"re-run. If that still does not work, use "
            f"htrpipe.roi_html.roi_picker."
        )
        if raise_on_fail:
            raise RuntimeError(message)
        print(message)
    return backend


def _mpl_at_least(version: str, target: Tuple[int, int]) -> bool:
    parts = []
    for chunk in version.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts) >= tuple(target)


# ---------------------------------------------------------------------------
# Homography alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Adjusted ROIs for one page, plus how much to trust them."""

    rois: Dict[str, Box]
    n_matches: int = 0
    n_inliers: int = 0
    homography: Optional[np.ndarray] = None
    ok: bool = True
    reason: str = ""

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def align_rois(
    reference_image: np.ndarray,
    reference_rois: Dict[str, Sequence[float]],
    pages: Sequence[np.ndarray],
    cfg: RoiAlignConfig,
    progress: bool = False,
) -> List[AlignmentResult]:
    """Transfer ``reference_rois`` onto each page in ``pages``.

    ORB keypoints are matched between the reference and each page, a
    homography is fitted with RANSAC, and the four corners of every reference
    box are warped through it. The axis-aligned bounding box of the warped
    corners becomes the adjusted ROI.

    Unlike the original implementation, this reports match quality and honours
    ``cfg.on_low_inliers``. A homography fitted to mostly-outlier matches does
    not fail loudly -- it returns a plausible-looking matrix that puts the ROIs
    in the wrong place, which is far harder to notice downstream than an error.
    """
    if not cfg.enabled:
        static = {name: tuple(int(v) for v in box) for name, box in reference_rois.items()}
        return [AlignmentResult(rois=dict(static), ok=True, reason="alignment disabled")
                for _ in pages]

    orb = cv2.ORB_create(cfg.orb_features)
    ref_gray = to_gray(reference_image)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    if des_ref is None or len(kp_ref) < 4:
        raise ValueError(
            "ORB found too few keypoints in the reference image to align "
            "anything. Is the reference page blank or extremely low contrast?"
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    results: List[AlignmentResult] = []

    for i, page in enumerate(pages):
        fallback = {name: tuple(int(v) for v in box) for name, box in reference_rois.items()}

        page_gray = to_gray(page)
        kp_page, des_page = orb.detectAndCompute(page_gray, None)

        if des_page is None or len(kp_page) < 4:
            results.append(_handle_failure(cfg, fallback, "too few ORB keypoints on page", i))
            continue

        matches = sorted(matcher.match(des_ref, des_page), key=lambda m: m.distance)
        matches = matches[: cfg.max_matches]
        if len(matches) < 4:
            results.append(_handle_failure(cfg, fallback, f"only {len(matches)} matches", i))
            continue

        src = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_page[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_threshold)
        if homography is None:
            results.append(_handle_failure(cfg, fallback, "homography could not be estimated", i))
            continue

        n_inliers = int(mask.sum()) if mask is not None else 0
        adjusted = warp_rois(reference_rois, homography)

        if n_inliers < cfg.min_inliers:
            result = _handle_failure(
                cfg, fallback,
                f"only {n_inliers} RANSAC inliers (< min_inliers={cfg.min_inliers})", i,
                adjusted=adjusted,
            )
            result.n_matches = len(matches)
            result.n_inliers = n_inliers
            result.homography = homography
            results.append(result)
            continue

        results.append(AlignmentResult(
            rois=adjusted, n_matches=len(matches), n_inliers=n_inliers,
            homography=homography, ok=True,
        ))

        if progress:
            print(f"page {i}: {n_inliers}/{len(matches)} inliers")

    return results


def _handle_failure(cfg, fallback, reason, index, adjusted=None) -> AlignmentResult:
    message = f"ROI alignment for page {index}: {reason}"
    if cfg.on_low_inliers == "raise":
        raise RuntimeError(message)
    warnings.warn(message, stacklevel=3)
    use = fallback if (cfg.on_low_inliers == "fallback" or adjusted is None) else adjusted
    return AlignmentResult(rois=use, ok=False, reason=reason)


def warp_rois(rois: Dict[str, Sequence[float]], homography: np.ndarray) -> Dict[str, Box]:
    """Warp each box's four corners and take the axis-aligned bounding box."""
    out: Dict[str, Box] = {}
    for name, (x1, y1, x2, y2) in rois.items():
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, homography)
        xs, ys = warped[:, 0, 0], warped[:, 0, 1]
        out[name] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return out


def alignment_report(results: Sequence[AlignmentResult], doc_ids: Sequence):
    """Per-page match quality as a DataFrame -- check this before trusting a
    cross-layout run."""
    import pandas as pd

    return pd.DataFrame([
        {"doc_id": doc_id, "matches": r.n_matches, "inliers": r.n_inliers,
         "inlier_ratio": round(r.inlier_ratio, 3), "ok": r.ok, "reason": r.reason}
        for doc_id, r in zip(doc_ids, results)
    ]).set_index("doc_id")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def show_rois(image: np.ndarray, rois: Dict[str, Sequence[float]],
              title: str = "ROIs", figsize=(10, 10), fontsize: int = 9) -> None:
    """Draw labelled boxes over an image."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    for name, (x1, y1, x2, y2) in rois.items():
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor="lime", facecolor="none"))
        ax.text(x1, y1 - 6, name, color="lime", fontsize=fontsize, fontweight="bold")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_crops(crops: Dict[str, np.ndarray], title: str = "", max_cols: int = 3,
               figsize_per_row=(12, 1.8)) -> None:
    """Grid of field crops, for eyeballing preprocessing quality."""
    import matplotlib.pyplot as plt

    names = list(crops)
    if not names:
        print("no crops to show")
        return

    n_rows = (len(names) + max_cols - 1) // max_cols
    fig, axes = plt.subplots(n_rows, max_cols,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        crop = crops[name]
        if crop.size:
            ax.imshow(crop, cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(names):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
