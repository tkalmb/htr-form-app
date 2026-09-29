"""One JSON per form type, holding everything the app knows about it.

A form configuration is an htrpipe layout file with two optional extras:

* each field may carry a ``label`` -- how the field is captioned on the
  printed form (a string, or a list of accepted spellings)
* the file may carry top-level ``stop_labels`` -- printed text that is not a
  field but still ends the preceding field's span (Chandra)

Both extras are invisible to htrpipe: ``load_layout_spec`` ignores keys it
does not know, so a configuration with labels loads as an ordinary layout
and the TrOCR path is unaffected. The flip side is that the labels do NOT
survive into the ``LayoutSpec`` -- they are read straight from the JSON with
``labels_from_raw`` below. That is what lets one file serve all three
models: TrOCR reads the ROIs through htrpipe, the whole-page models read the
labels through this module, and neither trips over the other's data.

Fields without measured ROIs (a form defined by hand, for the whole-page
models only) are written with ``"roi": null``. Loading such a file gives the
semantics and labels but no geometry, so the app offers it for the VLM paths
and not for TrOCR.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

from htrpipe import LayoutSpec, load_layout_spec

#: Files in the layouts directory that the app offers. Any .json qualifies --
#: a configuration saved from the app and a hand-written layout are the same
#: kind of file.
CONFIG_GLOB = "*.json"


def available_configs(layouts_dir) -> List[str]:
    """Configuration names (file stems) in the layouts directory."""
    d = pathlib.Path(layouts_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob(CONFIG_GLOB) if p.is_file())


def read_raw(path) -> dict:
    """The configuration file as a plain dict (no htrpipe parsing)."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def has_rois(raw: dict) -> bool:
    """True when every field carries a measured region.

    A configuration without regions cannot drive the TrOCR path, so the app
    checks this before offering it there.
    """
    fields = raw.get("fields") or []
    return bool(fields) and all(f.get("roi") for f in fields)


def load_layout(path) -> LayoutSpec:
    """Load a configuration as a LayoutSpec (requires measured regions)."""
    return load_layout_spec(path)


def labels_from_raw(raw: dict) -> Dict[str, object]:
    """{field name: label} for the fields that carry one.

    A single string stays a string; a list of accepted spellings stays a
    list, which is what the Chandra extractor expects.
    """
    labels: Dict[str, object] = {}
    for field in raw.get("fields") or []:
        label = field.get("label")
        if label:
            labels[field["name"]] = label
    return labels


def stop_labels_from_raw(raw: dict) -> List[str]:
    """Extra stop labels stored with the configuration (may be empty)."""
    return list(raw.get("stop_labels") or [])


def build(schema: Dict[str, dict],
          labels: Optional[Dict[str, object]] = None,
          stop_labels: Optional[List[str]] = None,
          source_layout: Optional[LayoutSpec] = None,
          name: str = "custom") -> dict:
    """Assemble a configuration dict from what the app currently holds.

    ``schema`` is the step-2 table (semantics); ``labels`` and
    ``stop_labels`` come from step 4. ``source_layout`` supplies the measured
    regions and checkbox option positions when the run is based on a layout
    file -- without it the fields are written with ``"roi": null`` and the
    configuration serves the whole-page models only.

    Geometry is never invented here: a field's ROI is either the measured one
    or null.
    """
    labels = labels or {}
    fields = []
    for field_name, entry in schema.items():
        field: dict = {"name": field_name, "type": entry["type"]}

        roi = None
        options = None
        if source_layout is not None:
            try:
                spec = source_layout[field_name]
                roi = list(spec.roi)
                options = ({k: list(v) for k, v in spec.options.items()}
                           if spec.options else None)
            except KeyError:
                pass
        field["roi"] = roi
        if options:
            field["options"] = options
        elif entry.get("options"):
            # Option NAMES only (a hand-defined form has no measured boxes).
            field["options"] = list(entry["options"])

        rule = entry.get("postprocess", "none")
        if rule and rule != "none":
            field["postprocess"] = rule
        if entry.get("pattern"):
            field["pattern"] = entry["pattern"]
        if entry.get("lexicon"):
            field["lexicon"] = entry["lexicon"]
        if field_name in labels:
            field["label"] = labels[field_name]
        fields.append(field)

    config = {
        "layout": name,
        "coordinate_space": (source_layout.coordinate_space
                             if source_layout is not None else "relative"),
        "reference_size": None,
        "_comment": [
            "Form configuration written by htr-form-app.",
            "'roi' is the measured field region (null when the form was "
            "defined by hand, which limits it to the whole-page models).",
            "'label' is how the field is captioned on the printed form; "
            "'stop_labels' is printed text that ends the preceding field.",
        ],
        "fields": fields,
    }
    if stop_labels:
        config["stop_labels"] = list(stop_labels)
    return config


def save(config: dict, path) -> pathlib.Path:
    """Write a configuration to disk, creating the directory if needed."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def rows_from_raw(raw: dict, pattern_columns, internal_to_ui) -> List[dict]:
    """Field table rows from a configuration that has no measured regions.

    Mirrors ``schema_editor.rows_from_layout`` but reads the plain dict,
    because such a file cannot be loaded as a LayoutSpec (htrpipe requires a
    region per field).
    """
    rows = []
    for field in raw.get("fields") or []:
        pattern_choice, custom = pattern_columns(field.get("pattern") or "")
        options = field.get("options") or []
        if isinstance(options, dict):
            options = list(options)
        rows.append({
            "name": field["name"],
            "type": internal_to_ui.get(field.get("type", "text"), "short"),
            "postprocess": field.get("postprocess", "none"),
            "pattern": pattern_choice,
            "custom_pattern": custom,
            "options": " | ".join(options),
            "lexicon": field.get("lexicon", ""),
        })
    return rows
