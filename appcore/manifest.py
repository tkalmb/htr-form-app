"""Run manifest: everything needed to trace an export back to its settings.

Follows the same philosophy as the evaluation's run manifests: a result
without its configuration is not evidence. The post-processing block is built
by ``htrpipe.postprocess_setup.manifest_entry`` -- the same function the
evaluation used -- so a post-processed number always carries the note about
which rules are TrOCR-calibrated.

App-specific additions on top of the evaluation manifests:
  * ``config_matches_evaluated`` -- whether the chosen presets equal a
    configuration the thesis actually measured
  * ``manual_edits`` -- every cell the user changed by hand in the review
    step (doc_id, field, before, after)
  * ``plausibility_flags`` -- how many advisory flags were raised (the flags
    themselves ship in the reasons CSV next to the predictions)
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Dict, List, Optional


def build(*,
          run_name: str,
          model_kind: str,
          model_meta: dict,
          layout_name: str,
          layout_source: str,
          schema_rows: List[dict],
          page_preset: str,
          field_preset: str,
          config_matches_evaluated: bool,
          n_forms: int,
          n_failures: int,
          seconds_total: Optional[float],
          postprocess_block: Optional[dict],
          n_plausibility_flags: int,
          manual_edits: List[dict],
          stop_labels: Optional[List[str]] = None,
          extra: Optional[dict] = None) -> dict:
    """Assemble the manifest dict. Pure function; writing is separate."""
    import torch
    import transformers

    import htrpipe

    manifest = {
        "run_name": run_name,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "app": "htr-form-app",
        "htrpipe_version": htrpipe.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_kind": model_kind,
        "model": model_meta,
        "layout": layout_name,
        "layout_source": layout_source,
        "field_schema_rows": schema_rows,
        "preprocessing": {
            "page_preset": page_preset,
            "field_preset": field_preset,
            "config_matches_evaluated": config_matches_evaluated,
        },
        "results": {
            "n_forms": n_forms,
            "n_failed": n_failures,
            "seconds_total": seconds_total,
        },
        "postprocess": postprocess_block if postprocess_block is not None
        else {"applied": False},
        "plausibility": {
            "note": ("advisory flags only; no value was modified by these "
                     "checks -- they are not post-processing"),
            "n_flags": n_plausibility_flags,
        },
        "manual_edits": manual_edits,
    }
    if stop_labels is not None:
        manifest["chandra_stop_labels"] = stop_labels
    if extra:
        manifest.update(extra)
    return manifest


def write(manifest: dict, path) -> pathlib.Path:
    """Write the manifest as pretty-printed JSON next to the predictions."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path
