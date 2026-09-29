"""Form layout schema: which fields exist, where they are, how to treat them.

Two-level design
----------------
Each field carries a coarse ``type`` and an optional ``postprocess`` rule.

``type`` drives the *mechanics* of the pipeline:

===============  =========================================================
type             effect
===============  =========================================================
``text``         single-line crop -> recognizer
``number``       single-line crop -> recognizer (separate evaluation group)
``long_text``    line segmentation before recognition, merge after
``checkbox_group``  no recognizer; ink-ratio argmax over ``options``
``ignore``       cropped for inspection only, never recognized or scored
===============  =========================================================

``postprocess`` drives the *correction rule* applied afterwards, and is
independent of type: ``city`` and ``comment`` are both text but need
``lexicon`` and ``llm`` respectively; ``birthday`` is a number needing
``date``, not the plain ``numeric`` digit fix.

Keeping these separate is deliberate. A single coarse type cannot express the
seven correction strategies the post-processing stage implements, and
collapsing them would silently discard most of that work.

Adapting to a new form means writing a new layout JSON -- no code changes.
"""

from __future__ import annotations

import json
import pathlib
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Field types the pipeline knows how to route.
FIELD_TYPES = {"text", "number", "long_text", "checkbox_group", "ignore"}

#: Post-processing rules implemented in :mod:`htrpipe.postprocess`.
POSTPROCESS_RULES = {
    "none",           # leave value untouched
    "numeric",        # digit-confusion fix, gated on the field's `pattern`
    "date",           # calendar-aware date correction (numeric or spelled month)
    "lexicon",        # nearest-neighbour against a named lexicon
    "email",          # whitespace/case normalisation + domain correction
    "street_suffix",  # correct only the trailing street-type word
    "name",           # digit/symbol cleanup + capitalisation, no lexicon
    "llm",            # local LLM with an accept/reject guardrail
}

#: Applied when a field declares no explicit ``postprocess``.
DEFAULT_POSTPROCESS_BY_TYPE = {
    "text": "none",
    "number": "numeric",
    "long_text": "none",
    "checkbox_group": "none",
    "ignore": "none",
}

#: Types that get sent to the text recognizer.
RECOGNIZED_TYPES = {"text", "number", "long_text"}


def _require_positive_area(box, label: str) -> None:
    """Reject a degenerate box.

    A zero-area ROI does not crash -- it produces an empty crop, which the
    recognizer turns into an empty string that looks like a plain recognition
    failure. Catching it at load time makes the real cause obvious.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"{label} {tuple(box)} has zero or negative area; expected "
            f"(x1, y1, x2, y2) with x2 > x1 and y2 > y1"
        )


@dataclass
class FieldSpec:
    """One field on the form."""

    name: str
    type: str
    #: ``(x1, y1, x2, y2)`` in the layout's declared coordinate space.
    roi: Tuple[float, float, float, float]
    postprocess: Optional[str] = None
    #: Regex the value is expected to match. Used by the ``numeric`` rule as a
    #: gate (correct only if the raw value fails it, keep only if the fix
    #: passes it). Living in the layout file rather than in code is what makes
    #: the rule reusable across differently-formatted forms.
    pattern: Optional[str] = None
    #: ``checkbox_group`` only: option name -> ROI *relative to this field's
    #: own crop* (0..1), not to the page.
    options: Optional[Dict[str, Tuple[float, float, float, float]]] = None
    #: ``lexicon`` rule only: key into the lexicons dict supplied at runtime.
    lexicon: Optional[str] = None
    #: Free-text note carried through to ``describe()``; for documenting a
    #: layout decision next to the field it affects.
    note: Optional[str] = None
    eval_group: Optional[str] = None   # falls back to type when None

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(
                f"field {self.name!r}: unknown type {self.type!r} "
                f"(expected one of {sorted(FIELD_TYPES)})"
            )
        rule = self.effective_postprocess
        if rule not in POSTPROCESS_RULES:
            raise ValueError(
                f"field {self.name!r}: unknown postprocess rule {rule!r} "
                f"(expected one of {sorted(POSTPROCESS_RULES)})"
            )
        if len(self.roi) != 4:
            raise ValueError(f"field {self.name!r}: roi must have 4 values, got {self.roi}")
        _require_positive_area(self.roi, f"field {self.name!r}: roi")

        if self.type == "checkbox_group":
            if not self.options:
                raise ValueError(
                    f"field {self.name!r}: type 'checkbox_group' requires an "
                    f"'options' mapping of option name -> relative ROI"
                )
            if len(self.options) < 2:
                raise ValueError(
                    f"field {self.name!r}: a checkbox group needs at least 2 "
                    f"options to compare ink ratios against"
                )
            for opt, box in self.options.items():
                if len(box) != 4:
                    raise ValueError(f"field {self.name!r}, option {opt!r}: roi must have 4 values")
                if not all(0.0 <= v <= 1.0 for v in box):
                    raise ValueError(
                        f"field {self.name!r}, option {opt!r}: option ROIs are "
                        f"fractions of the parent crop and must lie in [0, 1], got {box}"
                    )
                _require_positive_area(box, f"field {self.name!r}, option {opt!r}:")
        elif self.options:
            raise ValueError(f"field {self.name!r}: 'options' is only valid for type 'checkbox_group'")

        if rule == "numeric" and not self.pattern:
            warnings.warn(
                f"field {self.name!r} uses the 'numeric' rule but declares no "
                f"'pattern'; the rule is a no-op without one.",
                stacklevel=2,
            )
        if rule == "lexicon" and not self.lexicon:
            raise ValueError(
                f"field {self.name!r}: the 'lexicon' rule requires a 'lexicon' key "
                f"naming which lexicon to match against"
            )
        if self.pattern is not None:
            re.compile(self.pattern)  # fail loudly at load, not mid-run

    @property
    def effective_postprocess(self) -> str:
        if self.postprocess is not None:
            return self.postprocess
        return DEFAULT_POSTPROCESS_BY_TYPE[self.type]

    @property
    def is_recognized(self) -> bool:
        """True if this field's crop goes to the text recognizer."""
        return self.type in RECOGNIZED_TYPES

    @property
    def is_multiline(self) -> bool:
        """True if the crop must be split into lines before recognition.

        TrOCR is a single-line recognizer, so a multi-line crop has to be
        segmented first and the per-line predictions concatenated after.
        """
        return self.type == "long_text"

    @property
    def is_checkbox(self) -> bool:
        return self.type == "checkbox_group"

    @property
    def compiled_pattern(self):
        return re.compile(self.pattern) if self.pattern else None


@dataclass
class LayoutSpec:
    """A whole form layout: its fields and the space its ROIs are given in."""

    name: str
    fields: List[FieldSpec]
    #: ``"absolute"`` (pixels of the reference image) or ``"relative"``
    #: (fractions of width/height, resolution-independent).
    coordinate_space: str = "absolute"
    #: ``(width, height)`` the absolute coordinates were measured at. Optional,
    #: but recording it lets the loader warn when a page of a different size is
    #: used as reference. Left ``None`` rather than guessed.
    reference_size: Optional[Tuple[int, int]] = None
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.coordinate_space not in {"absolute", "relative"}:
            raise ValueError(
                f"coordinate_space must be 'absolute' or 'relative', got {self.coordinate_space!r}"
            )
        names = [f.name for f in self.fields]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate field names in layout {self.name!r}: {sorted(duplicates)}")
        if self.coordinate_space == "relative":
            for f in self.fields:
                if not all(0.0 <= v <= 1.0 for v in f.roi):
                    raise ValueError(
                        f"field {f.name!r}: coordinate_space is 'relative' but roi "
                        f"{f.roi} falls outside [0, 1]"
                    )

    # -- lookups ---------------------------------------------------------

    def __getitem__(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"no field {name!r} in layout {self.name!r}")

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def by_type(self, *types: str) -> List[FieldSpec]:
        return [f for f in self.fields if f.type in types]

    @property
    def recognized_fields(self) -> List[FieldSpec]:
        return [f for f in self.fields if f.is_recognized]

    @property
    def multiline_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.is_multiline]

    @property
    def checkbox_groups(self) -> List[FieldSpec]:
        return [f for f in self.fields if f.is_checkbox]

    @property
    def output_columns(self) -> List[str]:
        """Columns written to the predictions CSV: everything the pipeline
        actually produces a value for. ``ignore`` fields are excluded."""
        return [f.name for f in self.fields if f.type != "ignore"]

    # -- geometry --------------------------------------------------------

    def reference_rois(self, reference_image) -> Dict[str, Tuple[int, int, int, int]]:
        """Absolute pixel ROIs for ``reference_image``.

        Relative layouts are scaled to the image; absolute layouts are used
        as-is, with a warning if the reference image is not the size the
        coordinates were measured at.
        """
        h, w = reference_image.shape[:2]
        if self.coordinate_space == "relative":
            return {
                f.name: (int(f.roi[0] * w), int(f.roi[1] * h),
                         int(f.roi[2] * w), int(f.roi[3] * h))
                for f in self.fields
            }

        if self.reference_size is not None and tuple(self.reference_size) != (w, h):
            warnings.warn(
                f"layout {self.name!r} declares absolute coordinates measured at "
                f"{tuple(self.reference_size)} (w, h) but the reference image is "
                f"{(w, h)}. The ROIs will not line up. Either use a reference "
                f"image at the original resolution, or convert the layout to "
                f"relative coordinates with LayoutSpec.to_relative().",
                stacklevel=2,
            )
        return {f.name: tuple(int(v) for v in f.roi) for f in self.fields}

    def to_relative(self, width: int, height: int) -> "LayoutSpec":
        """Return a copy with absolute ROIs converted to fractions.

        Worth doing once per layout: relative coordinates survive a change of
        scan resolution, absolute ones do not.
        """
        if self.coordinate_space == "relative":
            return self
        new_fields = [
            FieldSpec(
                name=f.name, type=f.type,
                roi=(f.roi[0] / width, f.roi[1] / height, f.roi[2] / width, f.roi[3] / height),
                postprocess=f.postprocess, pattern=f.pattern, options=f.options,
                lexicon=f.lexicon, note=f.note,
            )
            for f in self.fields
        ]
        return LayoutSpec(
            name=self.name, fields=new_fields, coordinate_space="relative",
            reference_size=(width, height), source_path=self.source_path,
        )

    def with_rois(self, rois: Dict[str, Sequence[float]],
                  coordinate_space: Optional[str] = None) -> "LayoutSpec":
        """Copy this layout with new ROIs, keeping all field metadata.

        This is what lets the "draw" and "paste" ROI sources work: they supply
        geometry only, while the types, rules, patterns and checkbox options
        continue to come from the layout file. Fields absent from ``rois`` keep
        their existing box.
        """
        unknown = set(rois) - set(self.field_names)
        if unknown:
            raise ValueError(
                f"ROIs supplied for fields not in layout {self.name!r}: {sorted(unknown)}"
            )
        new_fields = [
            FieldSpec(
                name=f.name, type=f.type,
                roi=tuple(rois.get(f.name, f.roi)),
                postprocess=f.postprocess, pattern=f.pattern, options=f.options,
                lexicon=f.lexicon, note=f.note,
            )
            for f in self.fields
        ]
        return LayoutSpec(
            name=self.name, fields=new_fields,
            coordinate_space=coordinate_space or self.coordinate_space,
            reference_size=self.reference_size, source_path=self.source_path,
        )

    def with_schema(self, schema: Dict[str, Dict[str, Any]],
                    strict: bool = True) -> "LayoutSpec":
        """Copy this layout with field *semantics* replaced by ``schema``.

        The inverse split of :meth:`with_rois`: geometry (``roi``, checkbox
        ``options``) stays with the layout file, while ``type``,
        ``postprocess``, ``pattern``, ``lexicon`` and ``note`` come from a
        plain dict -- which is what lets them be defined and edited in a
        notebook cell instead of a JSON file.

        This factoring matters when several layouts describe the *same form
        semantics* in different positions: layouts A, B and C have identical
        fields and rules but different coordinates, so the schema belongs in
        one shared place and only the geometry per layout file.

        Field order follows ``schema``, and that order determines the column
        order of the predictions CSV -- so reordering the dict reorders the
        output.

        With ``strict=True`` the schema must name exactly the layout's fields;
        any extra or missing name raises with both lists shown. ``strict=False``
        leaves unmentioned fields as they are.
        """
        known = set(self.field_names)
        given = set(schema)

        if strict:
            problems = []
            if given - known:
                problems.append(f"not in layout {self.name!r}: {sorted(given - known)}")
            if known - given:
                problems.append(f"missing from the schema: {sorted(known - given)}")
            if problems:
                raise ValueError(
                    "field schema does not match the layout -- " + "; ".join(problems)
                )
        elif given - known:
            raise ValueError(
                f"schema names fields not in layout {self.name!r}: {sorted(given - known)}"
            )

        existing = {f.name: f for f in self.fields}
        ordered = list(schema) + [n for n in self.field_names if n not in schema]

        new_fields = []
        for name in ordered:
            current = existing[name]
            entry = schema.get(name)
            if entry is None:
                new_fields.append(current)
                continue
            unknown_keys = set(entry) - {"type", "postprocess", "pattern", "lexicon", "note"}
            if unknown_keys:
                raise ValueError(
                    f"field {name!r}: unsupported schema key(s) {sorted(unknown_keys)}. "
                    f"Geometry ('roi', 'options') belongs in the layout file."
                )
            new_type = entry.get("type", current.type)
            new_fields.append(FieldSpec(
                name=name,
                type=new_type,
                roi=current.roi,
                postprocess=entry.get("postprocess", current.postprocess),
                pattern=entry.get("pattern", current.pattern),
                # Options only belong to a checkbox group. Retyping a field
                # away from checkbox_group drops them rather than failing
                # validation on a leftover key.
                options=current.options if new_type == "checkbox_group" else None,
                lexicon=entry.get("lexicon", current.lexicon),
                note=entry.get("note", current.note),
            ))

        return LayoutSpec(
            name=self.name, fields=new_fields,
            coordinate_space=self.coordinate_space,
            reference_size=self.reference_size, source_path=self.source_path,
        )

    # -- reporting -------------------------------------------------------

    def describe(self):
        """Field table as a DataFrame, for a quick look in the notebook."""
        import pandas as pd

        return pd.DataFrame([
            {
                "field": f.name,
                "type": f.type,
                "postprocess": f.effective_postprocess,
                "recognized": f.is_recognized,
                "multiline": f.is_multiline,
                "pattern": f.pattern or "",
                "lexicon": f.lexicon or "",
                "options": ", ".join(f.options) if f.options else "",
                "note": f.note or "",
            }
            for f in self.fields
        ]).set_index("field")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "layout": self.name,
            "coordinate_space": self.coordinate_space,
            "reference_size": list(self.reference_size) if self.reference_size else None,
            "fields": [],
        }
        for f in self.fields:
            entry: Dict[str, Any] = {"name": f.name, "type": f.type, "roi": list(f.roi)}
            if f.postprocess is not None:
                entry["postprocess"] = f.postprocess
            if f.pattern:
                entry["pattern"] = f.pattern
            if f.lexicon:
                entry["lexicon"] = f.lexicon
            if f.options:
                entry["options"] = {k: list(v) for k, v in f.options.items()}
            if f.note:
                entry["note"] = f.note
            out["fields"].append(entry)
        return out

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_layout_spec(path) -> LayoutSpec:
    """Load and validate a layout JSON file."""
    path = pathlib.Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if "fields" not in raw:
        raise ValueError(f"{path}: layout file must contain a 'fields' list")

    fields = []
    for entry in raw["fields"]:
        options = entry.get("options")
        if options:
            options = {k: tuple(v) for k, v in options.items()}
        fields.append(FieldSpec(
            name=entry["name"],
            type=entry["type"],
            roi=tuple(entry["roi"]),
            postprocess=entry.get("postprocess"),
            pattern=entry.get("pattern"),
            options=options,
            lexicon=entry.get("lexicon"),
            note=entry.get("note"),
        ))

    reference_size = raw.get("reference_size")
    return LayoutSpec(
        name=raw.get("layout", path.stem),
        fields=fields,
        coordinate_space=raw.get("coordinate_space", "absolute"),
        reference_size=tuple(reference_size) if reference_size else None,
        source_path=str(path),
    )


def layout_from_rois(
    name: str,
    rois: Dict[str, Sequence[float]],
    types: Optional[Dict[str, str]] = None,
    coordinate_space: str = "absolute",
    **field_kwargs: Dict[str, Any],
) -> LayoutSpec:
    """Build a LayoutSpec from a plain ``{field: (x1, y1, x2, y2)}`` dict.

    Convenience for the "paste coordinates" and "draw interactively" paths in
    the notebook, where you have a bare dict and want a validated spec without
    hand-writing JSON. Fields default to type ``text``.
    """
    types = types or {}
    fields = [
        FieldSpec(name=fname, type=types.get(fname, "text"), roi=tuple(box),
                  **field_kwargs.get(fname, {}))
        for fname, box in rois.items()
    ]
    return LayoutSpec(name=name, fields=fields, coordinate_space=coordinate_space)
