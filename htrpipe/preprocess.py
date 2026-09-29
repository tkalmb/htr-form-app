"""Image preprocessing: full page, then per-field crops.

Colour contract
---------------
Every function here now preserves the channel count of its input: an RGB
(or RGBA) array in produces an RGB array out; a grayscale array in stays
grayscale. Earlier versions forced everything to 2-D grayscale as the first
step of ``preprocess_page``/``preprocess_roi``, which meant page and field
colour was thrown away *before* crops were ever extracted -- inpainting,
CLAHE, resizing etc. never had colour to lose, but only because it had
already been discarded upstream.

The pattern used throughout: measurement operations (skew-angle estimation,
the line-removal mask, the text bounding box) need a single channel and
always derive one via :func:`to_gray`; the actual pixel-changing operations
(rotate, inpaint, blur, resize, crop) are applied to the image as given, at
whatever channel count it has. cv2's own inpaint/resize/warpAffine/medianBlur
all accept 1- or 3-channel 8-bit input, so this mostly falls out for free --
the exceptions (CLAHE, large-kernel median blur) are handled explicitly
below.

One deliberate exception: ``cfg.binarize`` in :func:`preprocess_roi` always
collapses to a grayscale binary image. Otsu thresholding is a single-channel
operation and a binary mask has no colour to preserve by definition -- if a
downstream model needs colour for a given field, that field's config should
leave ``binarize = False``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import PagePreprocessConfig, RoiPreprocessConfig


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2-D grayscale view of ``image`` (RGB, RGBA or already gray)."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"unsupported image shape {image.shape}")


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel RGB view of ``image``. Used at the recognizer
    boundary, where models expect 3 channels. A no-op for images that are
    already RGB -- which, now that colour survives preprocessing, is the
    common case for scans that started out coloured."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid) -> np.ndarray:
    """CLAHE, channel-count-agnostic.

    For grayscale images this is exactly the old behaviour. For colour
    images, CLAHE is applied to the L channel in LAB space rather than to
    each RGB channel independently: boosting contrast per-channel shifts
    hue and saturation (the channels don't move together), whereas LAB
    separates lightness from colour so contrast improves without the ink
    colour drifting.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _median_blur_any(image: np.ndarray, ksize: int) -> np.ndarray:
    """cv2.medianBlur, but safe for any kernel size on colour images.

    OpenCV only allows ``ksize > 5`` on single-channel 8-bit images; calling
    it on a 3-channel image with a larger kernel raises. Splitting into
    channels and blurring each separately sidesteps that limit. For
    ``ksize <= 5`` this calls straight through to ``cv2.medianBlur``, which
    already supports multi-channel input, so behaviour there is unchanged.
    """
    if image.ndim == 2 or ksize <= 5:
        return cv2.medianBlur(image, ksize)
    channels = cv2.split(image)
    return cv2.merge([cv2.medianBlur(c, ksize) for c in channels])


# ---------------------------------------------------------------------------
# Page level
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Fold any ``minAreaRect`` angle into ``[-45, 45]``.

    OpenCV has reported this angle in different ranges across versions
    (``[-90, 0)`` historically, ``(0, 90]`` in some builds). Folding modulo 90
    handles both, since a rectangle rotated by ``a`` and by ``a + 90`` are the
    same rectangle with width and height swapped.
    """
    angle = angle % 90.0
    return angle - 90.0 if angle > 45.0 else angle


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
        flags=cv2.INTER_CUBIC,
    )


#: Which way to rotate to *undo* a measured skew angle: ``+1`` or ``-1``.
#: Discovered from the first page that actually needs deskewing, then reused.
_ROTATION_SIGN: Optional[float] = None


def reset_rotation_calibration() -> None:
    """Forget the learned rotation direction (used in tests)."""
    global _ROTATION_SIGN
    _ROTATION_SIGN = None


def rotation_sign() -> Optional[float]:
    """The learned rotation direction, or ``None`` if nothing has needed it yet."""
    return _ROTATION_SIGN


def estimate_skew_angle(gray: np.ndarray) -> Optional[float]:
    """Estimate page skew in degrees from the largest contour's min-area rect.

    Takes a grayscale image -- pass a colour image through :func:`to_gray`
    first, or call :func:`deskew` directly, which does that for you.
    Returns ``None`` when no contour is found. The result is folded into
    ``[-45, 45]`` -- see :func:`_normalize_angle`.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    return _normalize_angle(cv2.minAreaRect(max(contours, key=cv2.contourArea))[-1])


def deskew(image: np.ndarray, min_angle: float = 1.0, verify: bool = True) -> np.ndarray:
    """Rotate ``image`` to remove page skew. No-op below ``min_angle`` degrees.

    Returns an image with the same channel count as ``image``: colour in,
    colour out. The angle is always *estimated* from a grayscale view (skew
    detection doesn't need colour and is cheaper without it); the direction
    check below also rotates only that grayscale view, since it just needs
    the resulting residual angle, not a viewable image. The winning
    rotation is then applied once to the real input, so a colour page costs
    exactly one ``warpAffine`` call, same as before.

    The direction that *undoes* a measured skew depends on the angle
    convention of the installed OpenCV build, and getting it wrong does not
    fail visibly -- it rotates the page the wrong way, **doubling** the skew.
    This is not hypothetical: an earlier version of this code used
    ``getRotationMatrix2D(center, -angle, ...)``, which on OpenCV 4.13 doubles
    the skew of every page it touches. Only pages under the ``min_angle``
    cutoff escaped, so the fault was invisible on near-straight scans.

    Rather than hard-code a sign, this measures the result. With
    ``verify=True`` the first page that needs deskewing tries both directions
    and keeps whichever leaves less residual skew; the winning direction is
    cached, so later pages cost one extra angle estimate rather than an extra
    rotation. If neither direction improves on doing nothing, the original is
    returned -- the stage degrades to a no-op, never to a worse image.
    """
    global _ROTATION_SIGN

    gray = to_gray(image)
    angle = estimate_skew_angle(gray)
    if angle is None or abs(angle) < min_angle:
        return image

    if not verify:
        return _rotate(image, (_ROTATION_SIGN or 1.0) * angle)

    order = [1.0, -1.0] if _ROTATION_SIGN is None else [_ROTATION_SIGN, -_ROTATION_SIGN]

    best_residual, best_sign = float("inf"), None
    for sign in order:
        rotated_gray = _rotate(gray, sign * angle)
        residual = estimate_skew_angle(rotated_gray)
        residual = abs(residual) if residual is not None else float("inf")
        if residual < best_residual:
            best_residual, best_sign = residual, sign
        if residual <= abs(angle) * 0.25:
            break  # clearly the right direction; no need to try the other

    if best_sign is None or best_residual >= abs(angle):
        return image  # neither direction helped -- leave the page alone

    _ROTATION_SIGN = best_sign
    return _rotate(image, best_sign * angle)


def preprocess_page(image: np.ndarray, cfg: PagePreprocessConfig) -> np.ndarray:
    """Optional denoise -> optional CLAHE -> optional deskew.

    Returns an array with the same channel count as ``image``: a colour
    scan in produces a colour page out. Previously this started with
    ``to_gray(image)``, which discarded colour before any of the later
    pipeline stages -- including field extraction -- ever saw it.
    """
    out = image

    if cfg.denoise:
        out = _median_blur_any(out, cfg.denoise_kernel)

    if cfg.clahe:
        out = _apply_clahe(out, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

    if cfg.deskew:
        out = deskew(out, min_angle=cfg.deskew_min_angle, verify=cfg.deskew_verify)

    return out


# ---------------------------------------------------------------------------
# ROI level
# ---------------------------------------------------------------------------

def remove_lines(
    image: np.ndarray,
    horizontal_kernel=(40, 1),
    vertical_kernel=(1, 40),
    dilate_kernel=(3, 3),
    iterations: int = 2,
    length_frac: Optional[float] = None,
) -> np.ndarray:
    """Inpaint over straight horizontal/vertical rules (form field lines).

    Morphological opening with a long thin kernel keeps only structures at
    least that long in one direction -- i.e. ruled lines, not handwriting.
    The mask is dilated slightly so the anti-aliased edges of a line go too,
    then Telea inpainting fills the removed pixels from their surroundings
    (less destructive to strokes crossing the line than flat white fill).

    The mask is always computed from a grayscale view of ``image`` (line
    detection doesn't need colour), but ``cv2.inpaint`` is called on
    ``image`` itself -- it accepts 1- or 3-channel 8-bit input directly, so
    a colour crop is inpainted in colour with no extra branching needed.

    ``length_frac`` optionally sizes the opening kernels relative to the
    crop rather than in absolute pixels -- the horizontal kernel becomes
    ``length_frac * width``, the vertical one ``length_frac * height``, with
    the supplied absolute values as a floor. A fixed 40 px kernel means
    something different on a 300 px crop than on a 1200 px one, so a rule
    removed cleanly at one scan resolution can survive at another. A surviving
    rule is the worst possible input to line segmentation: it is a row of
    near-maximal ink that anchors a band and, under a max-relative threshold,
    raises the bar for every genuine text row in the crop.

    Left at ``None`` (the default) behaviour is unchanged.
    """
    gray = to_gray(image)

    if length_frac is not None:
        h, w = gray.shape[:2]
        horizontal_kernel = (max(horizontal_kernel[0], int(length_frac * w)), 1)
        vertical_kernel = (1, max(vertical_kernel[1], int(length_frac * h)))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = np.zeros(binary.shape, dtype=np.uint8)

    for kernel_size in (horizontal_kernel, vertical_kernel):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel),
                      iterations=iterations)

    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def crop_to_text(
    image: np.ndarray,
    threshold: int = 200,
    denoise_kernel: int = 5,
    padding: int = 5,
) -> np.ndarray:
    """Crop to the bounding box of the dark pixels, with padding.

    The bounding box is found from a grayscale view of ``image``; the crop
    itself slices ``image`` as given, so a colour crop in yields a colour
    crop out.

    A fixed threshold (not Otsu) is used on purpose: Otsu always splits the
    histogram somewhere, so on a blank crop it would invent "text" out of
    scanner noise. A fixed cutoff simply finds nothing, and the original crop
    is returned unchanged.
    """
    gray = to_gray(image)

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    #denoised = cv2.medianBlur(binary, denoise_kernel)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    return image[y:y + h, x:x + w]


def upscale_to_min_height(image: np.ndarray, min_height: int) -> np.ndarray:
    """Scale ``image`` up until it is at least ``min_height`` rows tall.

    ``cv2.resize`` is channel-count-agnostic, so this works unchanged for
    both grayscale and colour crops.

    Extracted from :func:`preprocess_roi` so :mod:`htrpipe.segment` can
    apply it per line. Applied to a whole multi-line field crop the check never
    fires -- a three-line comment is already far taller than 32 px -- while
    each of its individual lines may be well under the target. The upscale was
    therefore silently skipped for exactly the crops written smallest.
    """
    if image.size == 0 or image.shape[0] >= min_height:
        return image
    scale = min_height / image.shape[0]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_roi(roi: np.ndarray, cfg: RoiPreprocessConfig,
                   defer_upscale: bool = False) -> np.ndarray:
    """Prepare one field crop for recognition.

    Returns an array with the same channel count as ``roi``, *except* when
    ``cfg.binarize`` is set: Otsu thresholding is single-channel by
    definition and its output is a black/white mask, so that step always
    collapses to grayscale regardless of what came in. If a field needs to
    stay in colour for a downstream model, leave ``binarize = False`` for it.

    ``defer_upscale`` skips the ``min_height`` step so a multi-line field
    can be split first and each line upscaled individually. The default
    ``False`` reproduces the original behaviour exactly.
    """
    out = roi

    if out.size == 0:
        return out

    if cfg.clahe:
        out = _apply_clahe(out, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

    if cfg.remove_lines:
        out = remove_lines(
            out,
            horizontal_kernel=cfg.line_horizontal_kernel,
            vertical_kernel=cfg.line_vertical_kernel,
            dilate_kernel=cfg.line_dilate_kernel,
            iterations=cfg.line_iterations,
        )

    if cfg.denoise:
        out = _median_blur_any(out, cfg.denoise_kernel)

    if not defer_upscale:
        out = upscale_to_min_height(out, cfg.min_height)

    if cfg.binarize:
        # Colour-destructive by nature -- see docstring above.
        gray = to_gray(out)
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if cfg.crop_to_text:
        out = crop_to_text(
            out,
            threshold=cfg.crop_threshold,
            denoise_kernel=cfg.crop_denoise_kernel,
            padding=cfg.crop_padding,
        )

    return out


def extract_roi(page: np.ndarray, box) -> np.ndarray:
    """Crop ``box = (x1, y1, x2, y2)`` from ``page``, clipped to its bounds.

    Clipping matters: homography-adjusted coordinates can fall partly outside
    the page, and NumPy slicing with a negative start silently wraps around to
    the far edge instead of erroring.
    """
    h, w = page.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    return page[y1:y2, x1:x2]
