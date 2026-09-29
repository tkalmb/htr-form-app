"""Turn whatever the user gives us into a directory of page images.

Why a directory and not a list of images: ``htrpipe.load_images`` already
handles image reading, doc_id extraction from filenames, ordering and
filtering. Rather than reimplementing any of that, this module's only job is
to make sure the input *is* a directory of images, and then hand over.

Three input routes:

1. A server directory path (typed into the app) -- used as-is.
2. Uploaded image files -- written into a fresh session directory.
3. An uploaded PDF -- each page rendered to PNG at 300 dpi.

300 dpi is pinned, not configurable: the TrOCR model and the ROI layouts were
built against 300 dpi scans, and a different resolution silently breaks both
the absolute ROI coordinates and the recognition quality.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
from typing import List, Optional, Tuple

from htrpipe import FormImage, load_images

#: Fixed rendering resolution. See the module docstring for why this is not a
#: setting the user can change.
PDF_RENDER_DPI = 300

#: File endings accepted from the image uploader. Mirrors
#: ``htrpipe.io_data.IMAGE_SUFFIXES`` so the two stay in agreement.
ACCEPTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def new_session_dir(prefix: str = "htr_app_input_") -> pathlib.Path:
    """Create a fresh temporary directory for this session's input images.

    A *fresh* directory per run matters: leftovers from a previous upload
    would be picked up by ``load_images`` and silently mixed into the batch.
    """
    return pathlib.Path(tempfile.mkdtemp(prefix=prefix))


def save_uploaded_images(uploaded_files, target_dir: pathlib.Path) -> int:
    """Write Streamlit ``UploadedFile`` objects into ``target_dir``.

    Filenames are kept as uploaded, because ``load_images`` derives each
    document id from the last run of digits in the filename -- renaming the
    files would change the ids the user sees in the results table.

    Returns the number of files written.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for uf in uploaded_files:
        suffix = pathlib.Path(uf.name).suffix.lower()
        if suffix not in ACCEPTED_IMAGE_SUFFIXES:
            # Skip silently-unusable files here; the caller reports the count
            # difference to the user.
            continue
        (target_dir / uf.name).write_bytes(uf.getbuffer())
        n += 1
    return n


def render_pdf_to_images(pdf_bytes: bytes, target_dir: pathlib.Path,
                         stem: str = "page") -> int:
    """Render every PDF page to a PNG at ``PDF_RENDER_DPI``.

    Uses PyMuPDF because it is a pinned dependency of this project and needs
    no external system binary (pdf2image would require the poppler system
    package, which cannot be assumed on a server).

    Pages are numbered from 1 so that ``load_images``' doc_id extraction
    (last digit run in the filename) yields 1, 2, 3, ... in page order.

    Returns the number of pages written.
    """
    import pymupdf  # imported lazily: only needed on the PDF route

    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI)
            # Zero-padded page number keeps filename sorting == page order.
            out_path = target_dir / f"{stem}-{page_index + 1:03d}.png"
            pixmap.save(str(out_path))
            n += 1
    return n


def load_forms(image_dir, limit: Optional[int] = None) -> List[FormImage]:
    """Load all page images from ``image_dir`` via htrpipe.

    Thin wrapper so the app has exactly one place where images enter the
    pipeline. All reading, id-extraction and ordering behaviour is
    ``htrpipe.load_images``'s, unchanged.
    """
    return load_images(image_dir, limit=limit)


def describe_forms(forms: List[FormImage]) -> Tuple[int, str]:
    """Short human-readable summary for the UI: count and size of the batch."""
    if not forms:
        return 0, "no images loaded"
    w, h = forms[0].size
    same_size = all(f.size == (w, h) for f in forms)
    size_note = f"{w} x {h} px" if same_size else "MIXED sizes (check your scans)"
    return len(forms), f"{len(forms)} page(s), {size_note}"


def cleanup_dir(path) -> None:
    """Delete a session input directory. Best-effort; never raises."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
