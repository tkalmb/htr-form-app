"""Bridge between the field table the user edits and htrpipe's LayoutSpec.

Step 2 of the app edits one row per field: name, type, post-processing rule,
expected pattern, checkbox options, lexicon. Printed labels are NOT here --
they are model-specific (only the whole-page models read them) and are
collected in step 4 instead.

The table speaks a simplified type vocabulary ("short", "long",
"checkbox", "ignore"); this module translates it to htrpipe's internal
``FIELD_TYPES``. Internally text and number stay distinct because the
number/pattern pairing drives the plausibility checks -- the translation
picks "number" exactly when the row has an expected pattern.

Vocabularies for rules come from htrpipe's ``POSTPROCESS_RULES`` -- the
single source of truth; nothing is redefined here.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

from htrpipe import POSTPROCESS_RULES, FieldSpec, LayoutSpec

#: Column order of the editable table in step 2.
ROW_COLUMNS = ["name", "type", "postprocess", "pattern", "custom_pattern",
               "options", "lexicon"]

# ---------------------------------------------------------------------------
# Simplified type vocabulary (UI) <-> htrpipe FIELD_TYPES (pipeline)
# ---------------------------------------------------------------------------

UI_TYPE_CHOICES = ["short", "long", "checkbox", "ignore"]

INTERNAL_TO_UI = {
    "text": "short",
    "number": "short",
    "long_text": "long",
    "checkbox_group": "checkbox",
    "ignore": "ignore",
}

#: Older saved field tables used the previous wording; accept both so a
#: schema file from an earlier version still loads.
_LEGACY_UI_TYPES = {"text/number": "short", "long text": "long"}


def _ui_to_internal(ui_type: str, has_pattern: bool) -> str:
    """Translate a table type to the pipeline type.

    "short" maps to internal "number" when the row declares an expected
    pattern, otherwise "text" -- the pattern is what makes a field numeric
    as far as the checks are concerned.
    """
    if ui_type == "checkbox":
        return "checkbox_group"
    if ui_type == "ignore":
        return "ignore"
    if ui_type == "long":
        return "long_text"
    return "number" if has_pattern else "text"


# ---------------------------------------------------------------------------
# Pattern presets
# ---------------------------------------------------------------------------

#: Human-readable pattern presets. The UI shows these names; the pipeline
#: receives the regex. "custom regex" switches to the free-text column so a
#: user can still supply their own expression.
NAMED_PATTERNS = {
    "decimal amount (e.g. 256,99 €)": r"^\d+([.,]\d+)?\s?€?$",
    "phone number (digits, spaces, ()+-/)": r"^[\d\s()+\-/]+$",
    "house number (e.g. 12a)": r"^\d+[a-zA-Z]?$",
    "postal code (exactly 5 digits)": r"^\d{5}$",
    "whole number (digits only)": r"^\d+$",
}
CUSTOM_PATTERN_LABEL = "custom regex"
PATTERN_CHOICES = [""] + list(NAMED_PATTERNS) + [CUSTOM_PATTERN_LABEL]

#: Reverse map so layouts whose regexes match a preset show the friendly name.
_REGEX_TO_NAME = {rx: name for name, rx in NAMED_PATTERNS.items()}


def pattern_columns_from_regex(regex: str) -> tuple:
    """(pattern choice, custom text) for a stored regex."""
    if not regex:
        return "", ""
    if regex in _REGEX_TO_NAME:
        return _REGEX_TO_NAME[regex], ""
    return CUSTOM_PATTERN_LABEL, regex


def resolve_pattern(row: dict) -> str:
    """The regex a row's pattern columns actually mean ("" when none)."""
    choice = str(row.get("pattern", "")).strip()
    if not choice:
        return ""
    if choice in NAMED_PATTERNS:
        return NAMED_PATTERNS[choice]
    if choice == CUSTOM_PATTERN_LABEL:
        return str(row.get("custom_pattern", "")).strip()
    # Legacy rows (saved before the presets existed) stored the raw regex in
    # the pattern column itself; honour it.
    return choice


# ---------------------------------------------------------------------------
# Layout -> rows (prefill helper)
# ---------------------------------------------------------------------------

def rows_from_layout(layout: LayoutSpec) -> List[dict]:
    """Prefill the step-2 table from a layout file.

    Used by the "start from a layout file" helper, and gives the TrOCR path
    field names that match the layout's -- which step 4 requires.
    """
    rows = []
    for f in layout.fields:
        pattern_choice, custom = pattern_columns_from_regex(f.pattern or "")
        rows.append({
            "name": f.name,
            "type": INTERNAL_TO_UI.get(f.type, "short"),
            "postprocess": f.effective_postprocess,
            "pattern": pattern_choice,
            "custom_pattern": custom,
            "options": " | ".join(f.options) if f.options else "",
            "lexicon": f.lexicon or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Rows -> schema
# ---------------------------------------------------------------------------

def rows_to_schema(rows: List[dict]) -> Tuple[Dict[str, dict], List[str], List[str]]:
    """Convert edited rows into (schema, problems, notices).

    ``schema`` entries carry htrpipe keys (type/postprocess/pattern/lexicon)
    plus ``options`` (checkbox option names), which the step-4 builders
    consume. ``problems`` block the Apply; ``notices`` report the silent
    clean-ups on rows whose type makes a column inapplicable (the table
    widget cannot grey out single cells, so inapplicable entries are cleared
    here and said out loud).
    """
    schema: Dict[str, dict] = {}
    problems: List[str] = []
    notices: List[str] = []
    seen = set()

    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            problems.append("a row has an empty field name")
            continue
        if name in seen:
            problems.append(f"duplicate field name: {name!r}")
            continue
        seen.add(name)

        ui_type = str(row.get("type", "")).strip()
        ui_type = _LEGACY_UI_TYPES.get(ui_type, ui_type)
        rule = str(row.get("postprocess", "none")).strip() or "none"
        pattern = resolve_pattern(row)
        lexicon = str(row.get("lexicon", "")).strip() or None
        options = [o.strip() for o in
                   str(row.get("options", "")).split("|") if o.strip()]

        if ui_type not in UI_TYPE_CHOICES:
            problems.append(f"field {name!r}: unknown type {ui_type!r} "
                            f"(one of {UI_TYPE_CHOICES})")
            continue
        if rule not in POSTPROCESS_RULES:
            problems.append(f"field {name!r}: unknown postprocess {rule!r} "
                            f"(one of {sorted(POSTPROCESS_RULES)})")
            continue

        # ---- columns that do not apply to this type are cleared ----------
        if ui_type in ("checkbox", "ignore"):
            if rule != "none":
                notices.append(f"field {name!r}: postprocess {rule!r} does "
                               f"not apply to type {ui_type!r} -- set to "
                               f"'none'")
                rule = "none"
            if pattern:
                notices.append(f"field {name!r}: pattern does not apply to "
                               f"type {ui_type!r} -- cleared")
                pattern = ""
            if lexicon:
                notices.append(f"field {name!r}: lexicon does not apply to "
                               f"type {ui_type!r} -- cleared")
                lexicon = None
        if ui_type != "checkbox" and options:
            notices.append(f"field {name!r}: checkbox options only apply to "
                           f"type 'checkbox' -- cleared")
            options = []

        if pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                problems.append(f"field {name!r}: invalid regex: {exc}")
                continue

        # Structural requirements: a rule whose required input is missing
        # would not fail -- it would silently do nothing, which is worse.
        if rule == "numeric" and not pattern:
            problems.append(f"field {name!r}: rule 'numeric' needs a pattern")
        if rule == "lexicon" and not lexicon:
            problems.append(f"field {name!r}: rule 'lexicon' needs a lexicon name")

        entry = {"type": _ui_to_internal(ui_type, bool(pattern)),
                 "postprocess": rule}
        if pattern:
            entry["pattern"] = pattern
        if lexicon:
            entry["lexicon"] = lexicon
        if options:
            entry["options"] = options
        schema[name] = entry

    return schema, problems, notices


# ---------------------------------------------------------------------------
# Schema -> LayoutSpec (step 4)
# ---------------------------------------------------------------------------

def apply_schema_to_layout(layout: LayoutSpec, schema: Dict[str, dict]) -> LayoutSpec:
    """TrOCR path: user semantics onto a measured layout (geometry untouched).

    Field names must match the layout file exactly (htrpipe's strict mode):
    the ROIs are keyed by name, and a renamed field would orphan its box.

    Checkbox OPTIONS are geometry: each option name is tied to a measured box
    in the layout file. Empty options in the table inherit the layout's;
    non-empty ones must match it exactly, otherwise the mismatch is rejected
    loudly rather than ignored silently.
    """
    layout_names = {f.name for f in layout.fields}
    schema_names = set(schema)
    if layout_names != schema_names:
        missing = sorted(layout_names - schema_names)
        extra = sorted(schema_names - layout_names)
        raise ValueError(
            "field names must match the layout file exactly. "
            + (f"Missing from the table: {missing}. " if missing else "")
            + (f"Not in the layout: {extra}. " if extra else "")
            + "Use 'start from a layout file' in step 2 to align them.")

    cleaned = {}
    for name, entry in schema.items():
        entry = dict(entry)
        edited_options = entry.pop("options", None)
        if edited_options:
            layout_options = list(layout[name].options or [])
            if edited_options != layout_options:
                raise ValueError(
                    f"field {name!r}: checkbox options come from the layout "
                    f"file ({layout_options}) and cannot be changed in the "
                    f"table. Edit the layout JSON instead, or leave the "
                    f"options cell empty to inherit them.")
        cleaned[name] = entry
    return layout.with_schema(cleaned, strict=True)


def build_custom_layout(schema: Dict[str, dict], name: str = "custom") -> LayoutSpec:
    """VLM path: a LayoutSpec with no measured geometry.

    The whole-page models never read ROI positions, but every downstream
    stage (column order, post-processing, plausibility) is driven by a
    LayoutSpec -- so one is synthesised with placeholder boxes. Checkbox
    fields need their option NAMES here (for the prompt and the extractor);
    the placeholder positions are never read.
    """
    fields = []
    for field_name, entry in schema.items():
        options = None
        if entry["type"] == "checkbox_group":
            names_list = entry.get("options") or []
            if len(names_list) < 2:
                raise ValueError(
                    f"field {field_name!r}: a checkbox field needs at least "
                    f"two option names (step 2, column 'checkbox options', "
                    f"separated by |).")
            n = len(names_list)
            options = {
                opt: (i / n + 0.01, 0.05, (i + 1) / n - 0.01, 0.95)
                for i, opt in enumerate(names_list)
            }
        fields.append(FieldSpec(
            name=field_name,
            type=entry["type"],
            roi=(0.0, 0.0, 1.0, 1.0),   # placeholder: whole page, never cropped
            postprocess=entry.get("postprocess"),
            pattern=entry.get("pattern"),
            lexicon=entry.get("lexicon"),
            options=options,
            note="custom form; ROI is a placeholder (VLM page-level only)",
        ))
    return LayoutSpec(name=name, fields=fields, coordinate_space="relative")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_rows(rows: List[dict], path) -> pathlib.Path:
    """Persist the edited table as JSON so a configuration is reusable."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_rows(source) -> List[dict]:
    """Load a previously saved table (path or uploaded file object).

    Older files may contain a "label" column (labels now live in step 4) or
    raw regexes in "pattern"; both still load -- unknown columns are dropped
    and raw regexes are honoured by ``resolve_pattern``.
    """
    if hasattr(source, "read"):
        data = json.loads(source.read().decode("utf-8"))
    else:
        data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("field schema file must contain a JSON list of rows")
    return [{col: row.get(col, "") for col in ROW_COLUMNS} for row in data]
