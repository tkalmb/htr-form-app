"""Chandra page-level extraction.

PROVENANCE -- read before editing
---------------------------------
This module is the evaluation implementation that produced the thesis
results, taken over with one kind of change only: module-level settings
became parameters, defaulting to the evaluated values below. Model loading
and transcription are ``htrpipe.chandra_ocr.ChandraOCR``; parsing is
``htrpipe.formparse.parse_labeled_text``.

The evaluated condition uses the VENDOR backend with prompt type "ocr". One
hard-won rule, preserved because it cost a debugging session: parse the
RAW HTML, never the markdown -- ``parse_markdown`` discards
``<input type="checkbox" checked>``, which is exactly where the model records
which box was ticked.
"""

from __future__ import annotations

import pathlib
import time
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Evaluated settings
# ---------------------------------------------------------------------------

BACKEND = "vendor"                 # transformers | vendor -- evaluated: vendor
PROMPT_TYPE = "ocr"                # "ocr_layout" returns regions without text
ATTN_IMPL = "flash_attention_2"    # falls back to sdpa, and the manifest says so
MAX_PIXELS = 4096 * 28 * 28
MIN_PIXELS = 256 * 28 * 28
MAX_NEW_TOKENS = 2048              # the only runaway-generation cap (by
                                   # design; the evaluated value)
DO_SAMPLE = False
REPETITION_PENALTY = 1.0           # 1.0 deliberately: markup is repetitive
SEED = 42

#: Printed labels with accepted variants. Variants exist
#: because the three layouts print some labels differently.
DEFAULT_FIELD_LABELS: Dict[str, object] = {
    "last_name":      "Name",
    "first_name":     "Vorname",
    "monthly_salary": ["Monatliches Einkommen", "Einkommen"],
    "birthday":       "Geburtsdatum",
    "email":          ["E-Mail-Adresse", "E-Mail", "EMail-Adresse"],
    "phone":          "Telefonnummer",
    "studies":        "Studiengang",
    "street":         "Stra\u00dfe",
    "house_number":   "Hausnummer",
    "postal_code":    ["PLZ", "Postleitzahl"],
    "city":           "Ort",
    "comment":        "Kommentar",
    "signature":      "Unterschrift",
}

#: Printed text that is not a field but must still terminate the preceding
#: field's span. The app appends the user's own entries to
#: this list -- that is the "what comes after the last field" anchor.
DEFAULT_STOP_LABELS: List[str] = [
    "Allgemeines Formular zur Datenerfassung",
    "bitte ankreuzen",
    "Anschrift",
]


# ---------------------------------------------------------------------------
# Model loading (evaluated settings; module globals became parameters)
# ---------------------------------------------------------------------------

def load_runner(model_path: Optional[str] = None,
                prompt_type: str = PROMPT_TYPE,
                attn_implementation: str = ATTN_IMPL,
                min_pixels: int = MIN_PIXELS,
                max_pixels: int = MAX_PIXELS,
                backend: str = BACKEND,
                seed: int = SEED,
                max_new_tokens: int = MAX_NEW_TOKENS,
                do_sample: bool = DO_SAMPLE,
                repetition_penalty: float = REPETITION_PENALTY):
    """Load ChandraOCR exactly as the evaluation did.

    Returns ``(runner, meta)``. ``meta`` carries the vendor-package probe and
    the resolved attention implementation for the run manifest.
    """
    import torch

    from htrpipe import chandra_ocr
    from htrpipe.chandra_ocr import ChandraOCR

    torch.manual_seed(seed)

    probe = chandra_ocr.probe_vendor_package()
    if not probe["importable"] and backend == "vendor":
        raise RuntimeError(
            "chandra-ocr package is not importable but backend='vendor' was "
            "requested. Install with: pip install 'chandra-ocr[hf]'"
        )

    runner, model_meta = ChandraOCR.load(
        model_path or chandra_ocr.MODEL_ID,
        prompt_type=prompt_type,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        need_prompt_text=(backend == "transformers"),
    )
    runner.max_new_tokens = max_new_tokens
    runner.do_sample = do_sample
    runner.repetition_penalty = repetition_penalty

    meta = dict(model_meta)
    meta.update({
        "backend": backend,
        "prompt_type": prompt_type,
        "vendor_probe": probe,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    })
    return runner, meta


# ---------------------------------------------------------------------------
# Parsing (evaluated settings)
# ---------------------------------------------------------------------------

def parse_output(raw: str, runner, target_fields, field_labels,
                 checkbox_fields, line_join: str, stop_labels: List[str]):
    """raw model text -> (row, meta, preview). Never raises.

    Parses the RAW HTML, not the markdown. ``chandra.output.parse_markdown``
    is a lossy conversion: it discards ``<input type="checkbox" checked>``,
    which is exactly where the model records which box was ticked. The
    markdown is still produced, but only as a human-readable preview.
    """
    from htrpipe import formparse

    row, report = formparse.parse_labeled_text(
        raw, target_fields, field_labels, checkbox_fields, line_join,
        stop_labels=stop_labels)
    report["parse_source"] = "raw_html" if formparse.looks_like_html(raw) else "raw_text"
    preview, how = runner.to_markdown(raw)
    report["preview_source"] = how
    return row, report, preview


# ---------------------------------------------------------------------------
# Batch loop (evaluated control flow; console output became a callback)
# ---------------------------------------------------------------------------

def run_batch(forms,
              runner,
              target_fields,
              field_labels: Dict[str, object],
              checkbox_fields: Dict[str, List[str]],
              output_columns: List[str],
              raw_text_dir,
              line_join: str = " ",
              stop_labels: Optional[List[str]] = None,
              backend: str = BACKEND,
              progress: Optional[Callable[[int, int, str], None]] = None):
    """Process every form. Failures never abort the batch.

    Raw model output is written per page into ``raw_text_dir`` -- exactly as
    in the evaluation -- so a batch can be re-parsed later without re-running
    the model.

    Returns ``(rows, runlog_df, failures)``.
    """
    import pandas as pd

    stop_labels = list(stop_labels) if stop_labels is not None else list(DEFAULT_STOP_LABELS)
    raw_text_dir = pathlib.Path(raw_text_dir)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

    rows, records, failures = {}, [], []

    for i, (doc_id, path) in enumerate(forms, 1):
        raw, row, meta, err = "", None, {}, None
        t0 = time.perf_counter()
        try:
            raw, _ = runner.transcribe(path, backend=backend)
            row, meta, _ = parse_output(raw, runner, target_fields, field_labels,
                                        checkbox_fields, line_join, stop_labels)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        (raw_text_dir / f"{doc_id}.txt").write_text(raw or "", encoding="utf-8")

        if row is None:
            row = {c: "" for c in output_columns}
            failures.append({"doc_id": doc_id, "error": err, "raw": raw})

        rows[doc_id] = row
        records.append({
            "doc_id": doc_id, "seconds": round(dt, 2), "raw_chars": len(raw or ""),
            "ok": err is None,
            "n_anchored": meta.get("n_anchored", pd.NA),
            "n_empty": sum(1 for v in row.values() if not v),
            "checkbox": ",".join(f"{k}:{v}" for k, v in meta.get("checkbox", {}).items()),
            "duplicate_labels": ",".join(meta.get("duplicates", [])),
            "stops_used": ",".join(meta.get("stops_used", [])),
            "parse_source": meta.get("parse_source", ""),
            "error": err,
        })

        if progress is not None:
            status = "ok" if err is None else "FAIL"
            progress(i, len(forms),
                     f"doc_id {doc_id} {status} {dt:.1f}s "
                     f"anchored={records[-1]['n_anchored']}/{len(output_columns)}")

    runlog = pd.DataFrame(records).set_index("doc_id")
    return rows, runlog, failures
