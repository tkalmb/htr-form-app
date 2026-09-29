"""Flag-only plausibility checks on the extracted values.

IMPORTANT DISTINCTION (mirrors the thesis terminology): these checks are NOT
post-processing. Post-processing (htrpipe.postprocess) *changes* values under
validated rules. The checks here change NOTHING, ever -- they only mark cells
for human review in the app's results table. The raw value is always
preserved and exported.

What is reused from htrpipe (do not rewrite):
  * date format validity  -> ``postprocess.is_plausible_birthday``
  * number format         -> the field's own ``pattern`` via
                             ``FieldSpec.compiled_pattern``
  * checkbox confidence   -> ``CheckboxResult.status`` (``no_mark`` /
                             ``ambiguous``), already computed by the pipeline

Two further advisory flags, both specific to the whole-page Qwen model:
  * low confidence    -> the field's ``conf_min`` (probability of its least
                         certain token) is below ``CONF_MIN_THRESHOLD``, or
                         no score exists (NaN counts as 0)
  * demonstration bleed (few-shot only) -> a text field's value is identical
                         to the value of the same field on a demonstration
                         page, found with ``htrpipe.fewshot.bleeding_documents``
Neither changes a value either.

What is new here (and only here):
  * digits inside name fields
  * birthday age bounds: flagged when the date lies in the future or implies
    an age over 100 years (bounds set by the author)
  * a minimal email-format regex -- htrpipe's ``correct_email`` is a domain
    corrector and contains no format pattern to reuse

Date *parsing* for the age bounds reuses htrpipe's own private regexes
(``_DATE_NUMERIC_RE``, ``_DATE_TEXT_MONTH_RE``, ``_MONTH_TO_NUM``). Importing
private names is deliberate: writing a second date grammar here would let the
two drift apart, and the format check and the bounds check must agree on what
counts as a date.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, Iterable, List, Optional, Tuple

from htrpipe.postprocess import (
    _DATE_NUMERIC_RE,
    _DATE_TEXT_MONTH_RE,
    _MONTH_TO_NUM,
    is_plausible_birthday,
)

#: Age bound in years. A birthday implying an age above this is flagged.
MAX_AGE_YEARS = 100

#: A field whose conf_min is below this is flagged for review. HEURISTIC: set
#: by the author, NOT calibrated on validation data -- no claim is made that
#: it balances review effort against missed errors in any measured way.
CONF_MIN_THRESHOLD = 0.75

#: Minimal email shape: something@something.tld. Deliberately loose -- this is
#: an advisory flag, not an RFC validator; a false "looks fine" is cheaper
#: than drowning the reviewer in pedantic flags.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

_DIGIT_RE = re.compile(r"\d")


def _parse_date(text: str) -> Optional[_dt.date]:
    """Parse a value htrpipe's date grammar accepts into a ``date``.

    Returns ``None`` when the value does not parse or is not a real calendar
    date -- in that case the *format* flag (``is_plausible_birthday``) already
    covers it and no bounds check applies.
    """
    compact = re.sub(r"\s+", "", text or "")
    match = _DATE_NUMERIC_RE.match(compact)
    if match:
        day_s, month_s, year_s = match.groups()
        if not month_s.isdigit():
            return None
        month = int(month_s)
    else:
        match = _DATE_TEXT_MONTH_RE.match((text or "").strip())
        if not match:
            return None
        day_s, month_word, year_s = match.groups()
        month = _MONTH_TO_NUM.get(month_word.lower().strip("."), None)
        if month is None:
            return None
    try:
        year = int(year_s)
        if year < 100:               # two-digit year: same century assumption
            year += 2000             # htrpipe's leap check makes internally
        return _dt.date(year, month, int(day_s))
    except ValueError:
        return None


def check_frame(frame, layout, checkbox_results=None,
                today: Optional[_dt.date] = None,
                confidences: Optional[dict] = None,
                conf_threshold: float = CONF_MIN_THRESHOLD,
                changed_by_postprocess: Optional[set] = None,
                bleed_cells: Optional[list] = None,
                skip_docs: Iterable = (),
                reviewed_cells: Optional[set] = None,
                ) -> Tuple[Dict[Tuple[object, str], str], "object"]:
    """Run every applicable check on a predictions frame.

    Parameters
    ----------
    frame : the predictions DataFrame (doc_id index, one column per field).
    layout : the ``LayoutSpec`` the run used -- field types and rules decide
        which check applies to which column.
    checkbox_results : optional per-form dict of ``CheckboxResult`` from the
        TrOCR path; when absent (VLM paths) checkbox values are checked
        against the declared options instead.
    today : injectable current date (for tests); defaults to ``date.today()``.
    confidences : optional ``{doc_id: {field: conf_min}}`` (Qwen runs). A
        field below ``conf_threshold`` -- or without a score -- is flagged.
    changed_by_postprocess : ``(doc_id, field)`` cells that post-processing
        changed. Confidence describes the RAW model output, so for these the
        flag says the score refers to the raw value.
    bleed_cells : rows of ``fewshot.bleeding_documents`` (few-shot runs):
        cells whose value equals a demonstration's value for that field.
    skip_docs : doc_ids excluded from the confidence and bleed checks -- the
        demonstration pages, whose values were entered by hand, not predicted.
    reviewed_cells : ``(doc_id, field)`` cells a human corrected by hand. The
        confidence and bleed flags describe the MODEL's value, so they no
        longer apply once a person has replaced it; format checks still run
        on the corrected value.

    Returns ``(reasons, flags_frame)`` where ``reasons`` maps
    ``(doc_id, field) -> human-readable reason`` (several reasons for one cell
    are joined with "; ") and ``flags_frame`` is a boolean DataFrame aligned
    with ``frame`` (True = flagged).
    """
    import pandas as pd

    today = today or _dt.date.today()
    reasons: Dict[Tuple[object, str], str] = {}
    flags = pd.DataFrame(False, index=frame.index, columns=frame.columns)

    # Map doc_id -> {field: CheckboxResult} for quick lookup (TrOCR path).
    cb_by_doc = {}
    if checkbox_results is not None:
        for doc_id, per_form in zip(frame.index, checkbox_results):
            cb_by_doc[doc_id] = per_form

    for column in frame.columns:
        try:
            spec = layout[column]
        except KeyError:
            continue    # diagnostic columns pass through unchecked

        rule = spec.effective_postprocess
        pattern = spec.compiled_pattern

        for doc_id in frame.index:
            value = str(frame.at[doc_id, column])

            # ---- checkbox groups ----------------------------------------
            if spec.is_checkbox:
                result = cb_by_doc.get(doc_id, {}).get(column)
                if result is not None and result.status != "ok":
                    reasons[(doc_id, column)] = f"checkbox: {result.status}"
                    flags.at[doc_id, column] = True
                elif result is None:
                    # VLM path: no ink-ratio result exists; check the value
                    # against the declared options instead.
                    options = list(spec.options or {})
                    if value == "" or (options and value not in options):
                        reasons[(doc_id, column)] = (
                            "checkbox: no valid option recognised"
                            if value == "" else
                            f"checkbox: {value!r} is not one of {options}")
                        flags.at[doc_id, column] = True
                continue

            # Empty non-checkbox values are not implausible -- the form field
            # may genuinely be empty. The run log already counts empties.
            if not value:
                continue

            # ---- name fields: digits do not belong in names -------------
            if rule == "name" and _DIGIT_RE.search(value):
                reasons[(doc_id, column)] = "name contains digits"
                flags.at[doc_id, column] = True
                continue

            # ---- number fields: the field's own pattern -----------------
            if spec.type == "number" and pattern is not None and rule != "date":
                if not pattern.match(value):
                    reasons[(doc_id, column)] = "does not match the expected number format"
                    flags.at[doc_id, column] = True
                continue

            # ---- dates: format first, then bounds -----------------------
            if rule == "date":
                if not is_plausible_birthday(value):
                    reasons[(doc_id, column)] = "not a valid calendar date"
                    flags.at[doc_id, column] = True
                    continue
                parsed = _parse_date(value)
                if parsed is None:
                    continue
                if parsed > today:
                    reasons[(doc_id, column)] = "date lies in the future"
                    flags.at[doc_id, column] = True
                elif (today.year - parsed.year) > MAX_AGE_YEARS or (
                        today.year - parsed.year == MAX_AGE_YEARS
                        and (today.month, today.day) >= (parsed.month, parsed.day)):
                    reasons[(doc_id, column)] = f"implies an age over {MAX_AGE_YEARS} years"
                    flags.at[doc_id, column] = True
                continue

            # ---- email shape --------------------------------------------
            if rule == "email" and not EMAIL_RE.match(value):
                reasons[(doc_id, column)] = "does not look like an email address"
                flags.at[doc_id, column] = True

    # ---- further advisory flags, added ON TOP of a format reason ------------
    # A cell can be both malformed and low-confidence; both are reported.
    def add(key, text):
        if key[0] not in flags.index or key[1] not in flags.columns:
            return
        reasons[key] = f"{reasons[key]}; {text}" if key in reasons else text
        flags.at[key[0], key[1]] = True

    skip = set(skip_docs)
    changed = changed_by_postprocess or set()
    reviewed = reviewed_cells or set()

    if confidences is not None:
        for doc_id in frame.index:
            if doc_id in skip:
                continue
            per_doc = confidences.get(doc_id, {})
            for column in frame.columns:
                if (doc_id, column) in reviewed:
                    continue
                score = per_doc.get(column, float("nan"))
                # NaN (missing key, unparsed page) counts as confidence 0.
                effective = 0.0 if score != score else score
                if effective < conf_threshold:
                    text = ("no confidence score" if score != score else
                            f"low confidence ({score:.2f} < {conf_threshold:.2f})")
                    if (doc_id, column) in changed:
                        text += " -- score refers to the raw value"
                    add((doc_id, column), text)

    for cell in bleed_cells or []:
        if cell["doc_id"] in skip or (cell["doc_id"], cell["field"]) in reviewed:
            continue
        demos = ", ".join(str(d) for d in cell["from_demo"])
        add((cell["doc_id"], cell["field"]),
            f"identical to demonstration value (doc_id {demos})")

    return reasons, flags


def reasons_table(reasons: Dict[Tuple[object, str], str]):
    """Long-format DataFrame of flags for display, sorted by document."""
    import pandas as pd

    if not reasons:
        return pd.DataFrame(columns=["doc_id", "field", "reason"])
    rows = [{"doc_id": d, "field": f, "reason": r}
            for (d, f), r in reasons.items()]
    return pd.DataFrame(rows).sort_values(["doc_id", "field"]).reset_index(drop=True)
