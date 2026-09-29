"""Loading form images and writing predictions.

Ground truth is deliberately **not** required anywhere here. The original
``load_layout()`` took its file list from ``GroundTruth_<LAYOUT>.json``, which
meant the pipeline could not run on unlabelled forms at all. Here, images are
discovered from a directory; a manifest is one optional way to order/filter
them, and a split file is another, but neither is needed.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np

#: Default doc_id extraction: the LAST run of digits in the filename stem.
#: ``300dpi-A-001.png`` -> ``1``. The leading "300" is skipped because it is
#: not the last group.
_TRAILING_DIGITS = re.compile(r"(\d+)(?!.*\d)", re.DOTALL)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class FormImage:
    """One loaded scan."""

    doc_id: Union[int, str]
    image: np.ndarray  # RGB, uint8
    path: str

    @property
    def size(self):
        h, w = self.image.shape[:2]
        return (w, h)


def doc_id_from_filename(path, pattern: re.Pattern = _TRAILING_DIGITS) -> Union[int, str]:
    """Derive a document id from a filename, falling back to the stem."""
    stem = pathlib.Path(path).stem
    match = pattern.search(stem)
    if match:
        return int(match.group(1))
    return stem


def normalize_doc_id(doc_id):
    """Make ``"7"`` and ``7`` compare equal -- manifests and CSVs disagree
    about this often enough to be worth centralising."""
    if isinstance(doc_id, str) and doc_id.isdigit():
        return int(doc_id)
    return doc_id


def read_image_rgb(path) -> np.ndarray:
    """Read an image as RGB. Raises rather than returning ``None`` -- a silent
    ``None`` from ``cv2.imread`` on a bad path is a classic source of a
    confusing crash three stages later."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_images(
    image_dir,
    doc_ids: Optional[Iterable] = None,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
    id_pattern: re.Pattern = _TRAILING_DIGITS,
    limit: Optional[int] = None,
) -> List[FormImage]:
    """Load every image in ``image_dir``, optionally filtered to ``doc_ids``.

    Parameters
    ----------
    image_dir : directory containing the scans (not searched recursively).
    doc_ids : optional iterable of ids to keep. Use this to reproduce a split
        without the pipeline needing any labels -- pass the ids from your
        ``split_assignment.json``.
    limit : stop after this many images; handy for a quick smoke run.

    Returns images sorted by doc_id, so ordering is deterministic and does not
    depend on filesystem enumeration order.
    """
    image_dir = pathlib.Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image_dir does not exist or is not a directory: {image_dir}")

    wanted = None
    if doc_ids is not None:
        wanted = {normalize_doc_id(d) for d in doc_ids}

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in tuple(suffixes))
    if not paths:
        raise FileNotFoundError(
            f"no images with suffixes {tuple(suffixes)} found in {image_dir}"
        )

    out: List[FormImage] = []
    for path in paths:
        doc_id = normalize_doc_id(doc_id_from_filename(path, id_pattern))
        if wanted is not None and doc_id not in wanted:
            continue
        out.append(FormImage(doc_id=doc_id, image=read_image_rgb(path), path=str(path)))

    if wanted is not None:
        found = {f.doc_id for f in out}
        missing = wanted - found
        if missing:
            raise ValueError(
                f"{len(missing)} requested doc_id(s) had no matching image in "
                f"{image_dir}: {sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )

    out.sort(key=lambda f: (isinstance(f.doc_id, str), f.doc_id))
    if limit is not None:
        out = out[:limit]
    return out


def load_split_ids(split_json, key: str) -> List:
    """Read a list of doc_ids from a split-assignment JSON.

    A split file lists *which* documents belong to a split -- it carries no
    labels, so using it here keeps the pipeline ground-truth-free.
    """
    with open(split_json, encoding="utf-8") as fh:
        data = json.load(fh)
    if key not in data:
        raise KeyError(f"{split_json}: no split named {key!r} (available: {sorted(data)})")
    return [normalize_doc_id(d) for d in data[key]]


def load_manifest_ids(manifest_json) -> List:
    """Read the doc_ids listed in a manifest, ignoring any ground truth in it."""
    with open(manifest_json, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return [normalize_doc_id(item["doc_id"]) for item in manifest]


def slugify(text: str, max_length: int = 40) -> str:
    """Reduce a string to filename-safe characters."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-")
    return slug[:max_length].strip("-") or "unnamed"


def make_run_name(
    layout: str,
    model_path: str,
    split_key: Optional[str] = None,
    timestamp: bool = True,
    extra: Optional[str] = None,
) -> str:
    """Build a run name from what actually distinguishes one run from another.

    ``make_run_name("A", "./checkpoints/TrOCR-scads-finetuned-40", "val")``
    gives ``A_TrOCR-scads-finetuned-40_val_20260729-1432``.

    The model is reduced to the final path component, so a local checkpoint
    directory and a Hub id both give something readable. The timestamp keeps
    re-runs from silently overwriting each other -- which matters when the
    thing that changed between two runs was a setting rather than a filename.
    """
    parts = [slugify(layout, 12), slugify(pathlib.Path(str(model_path)).name)]
    if split_key:
        parts.append(slugify(split_key, 16))
    if extra:
        parts.append(slugify(extra, 24))
    if timestamp:
        from datetime import datetime
        parts.append(datetime.now().strftime("%Y%m%d-%H%M"))
    return "_".join(parts)


def save_predictions(df, path, index_label: str = "doc_id") -> pathlib.Path:
    """Write a predictions DataFrame to CSV and return the path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label=index_label, encoding="utf-8")
    return path


def load_predictions(path):
    """Read a predictions CSV back, with doc_ids normalised and NaN -> ""."""
    import pandas as pd

    df = pd.read_csv(path, index_col=0, encoding="utf-8")
    df.index = pd.Index([normalize_doc_id(i) for i in df.index], name="doc_id")
    return df.fillna("").astype(str)
