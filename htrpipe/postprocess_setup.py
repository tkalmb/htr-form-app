"""Shared post-processing setup for every notebook that writes predictions.

Why this module exists
----------------------
``postprocess.py`` supplies the *rules*; the notebooks used to supply the
*wiring* (which rule per field, which lexicon, which thresholds) by each
holding its own copy. Four copies of a field->rule table is four chances for a
TrOCR run and a VLM run to be corrected differently, which would make any
raw-vs-post comparison between them meaningless. The table lives here once.

Scope note that belongs in the thesis, not just in this docstring
-----------------------------------------------------------------
The rules are **not** uniformly model-agnostic:

* ``lexicon``, ``date``, ``email``, ``street_suffix``, ``llm`` are properties of
  the *field* (a closed list of cities, a real calendar, a closed domain set).
  They are as valid for a VLM as for TrOCR.
* ``numeric`` and ``name`` depend on ``DEFAULT_DIGIT_CONFUSION_MAP``, which
  ``postprocess.py`` documents as fitted to *this project's TrOCR* train/val
  errors (H->4, k->4, G->6, j->9). Applying it to a VLM is applying one model's
  error profile to another's output.

So a post-processed VLM run is a **separate declared condition**, not a
drop-in replacement for the raw VLM number. ``manifest_entry()`` writes that
provenance into the run manifest so the distinction survives into the results
table.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from .config import PostprocessConfig
from .postprocess import (
    DEFAULT_DIGIT_CONFUSION_MAP,
    CommentLLM,
    PostprocessResources,
    diff_predictions,
    postprocess_predictions,
    validate_resources,
)

__all__ = [
    "BASE_FIELD_SCHEMA",
    "MODEL_CALIBRATED_RULES",
    "tuned_config",
    "schema_for_layout",
    "postprocess_layout",
    "build_resources",
    "apply_postprocessing",
    "manifest_entry",
    "free_vram",
]


# ---------------------------------------------------------------------------
# Field -> rule table (single source of truth)
# ---------------------------------------------------------------------------

#: ``type`` drives pipeline mechanics, ``postprocess`` drives correction, and
#: the two are deliberately independent: ``birthday`` is a number but needs
#: calendar validation, ``city`` is text but needs a closed lexicon.
#:
#: This is the same dict that used to sit in ``pipeline_run.ipynb`` cell 7.
BASE_FIELD_SCHEMA: Dict[str, dict] = {
    "last_name":      {"type": "text",   "postprocess": "name"},
    "first_name":     {"type": "text",   "postprocess": "name"},
    "monthly_salary": {"type": "number", "postprocess": "numeric",
                       "pattern": r"^\d+([.,]\d+)?\s?€?$"},
    "birthday":       {"type": "number", "postprocess": "date"},
    "email":          {"type": "text",   "postprocess": "email"},
    "phone":          {"type": "number", "postprocess": "numeric",
                       "pattern": r"^[\d\s()+\-/]+$"},
    "studies":        {"type": "checkbox_group"},
    "street":         {"type": "text",   "postprocess": "street_suffix"},
    "house_number":   {"type": "number", "postprocess": "numeric",
                       "pattern": r"^\d+[a-zA-Z]?$"},
    "postal_code":    {"type": "number", "postprocess": "numeric",
                       "pattern": r"^\d{5}$"},
    "city":           {"type": "text",   "postprocess": "lexicon",
                       "lexicon": "german_cities"},
    "comment":        {"type": "long_text", "postprocess": "llm"},
    "signature":      {"type": "ignore"},
}

#: Rules whose behaviour depends on a table fitted to TrOCR's errors. Reported
#: in the manifest so a post-processed VLM run is never mistaken for a
#: model-agnostic one.
MODEL_CALIBRATED_RULES = ("numeric", "name")


def tuned_config() -> PostprocessConfig:
    """The thresholds tuned on Layout A validation, as one object.

    These were selected on val for the TrOCR condition. Reusing them unchanged
    for a VLM condition is the conservative choice -- re-tuning them per model
    would make the post-processed conditions differ in two ways at once
    (the model *and* the correction aggressiveness) and nothing could be
    attributed. If you do re-tune, do it on val and report both settings.
    """
    config = PostprocessConfig()

    # Edit distance allowed per character. 0.34 means a 12-character value may
    # be corrected to a candidate 4 edits away. Lower is more conservative.
    config.lexicon_max_relative_distance = 0.34   # city names
    config.email_max_relative_distance = 0.34     # email domains
    config.street_max_relative_distance = 0.34    # street-type suffixes
    config.month_max_relative_distance = 0.34     # spelled-out month names

    # How far the correction model is allowed to move the free text.
    config.llm_max_word_edit_distance = 5
    config.llm_max_word_count_diff = 2
    config.llm_max_changed_word_fraction = 0.4
    config.llm_min_changed_word_allowance = 2
    config.llm_allow_word_splits = True
    config.llm_max_new_tokens = 128   # German words cost more tokens than English

    return config


def schema_for_layout(layout_name: str) -> Dict[str, dict]:
    """``BASE_FIELD_SCHEMA``, identical for every layout.

    Checkbox option positions are now measured for layouts A, B and C, so the
    former per-layout exception (which typed ``studies`` as ``ignore`` where
    options were missing) has been removed. ``layout_name`` is kept in the
    signature so existing callers and manifests stay unchanged.

    Edit history: the removed branch tested ``in ("D")`` -- a string, not a
    one-element tuple -- so it also fired for an empty layout name. Removed
    2026-08 (authorised by the author) rather than repaired, because no layout
    needs the exception any more.
    """
    return {name: dict(spec) for name, spec in BASE_FIELD_SCHEMA.items()}


def postprocess_layout(layout, layout_name: str):
    """A copy of ``layout`` carrying the shared rules, patterns and lexicons.

    The page-level VLM notebooks derive ``target_fields`` and ``OUTPUT_COLUMNS``
    from the *unmodified* layout, so keep the return value in a separate name
    (``pp_layout``) rather than rebinding ``layout``. Post-processing is the
    only thing that needs the rules; changing the layout the columns came from
    would change which fields the run predicts.
    """
    return layout.with_schema(schema_for_layout(layout_name))


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def build_resources(
    *,
    config: Optional[PostprocessConfig] = None,
    city_lexicon_csv: Optional[str] = None,
    city_lexicon_column: str = "name",
    email_domains: Iterable[str] = (),
    comment_llm_path: Optional[str] = None,
    layout=None,
    verbose: bool = True,
) -> PostprocessResources:
    """Assemble the external data the rules need, and say what is missing.

    ``layout`` is optional and only used for the validation report: a rule with
    no resources is a silent no-op, which is far easier to catch here than in
    the results.
    """
    resources = PostprocessResources(
        email_domains=set(email_domains),
        config=config or PostprocessConfig(),
    )

    if city_lexicon_csv:
        n = resources.load_lexicon_from_csv(
            "german_cities", city_lexicon_csv, city_lexicon_column
        )
        if verbose:
            print(f"city lexicon: {n} entries from {city_lexicon_csv}")
    elif verbose:
        print("city lexicon: none supplied - the 'lexicon' rule will be a no-op")

    if comment_llm_path:
        # Lazily loaded: constructing this costs nothing until a value is
        # actually corrected.
        resources.comment_llm = CommentLLM(comment_llm_path, config=resources.config)
        if verbose:
            print(f"comment LLM:  {comment_llm_path} (loaded on first use)")
    elif verbose:
        print("comment LLM:  none supplied - the 'llm' rule will be a no-op")

    if verbose:
        print(f"email domains: {sorted(resources.email_domains) or 'none supplied'}")

    if layout is not None:
        problems = validate_resources(layout, resources)
        if problems:
            print("\nThese rules will be no-ops with the current resources:")
            for p in problems:
                print(f"  - {p}")
        elif verbose:
            print("\nAll rules have the resources they need.")

    return resources


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_postprocessing(predictions_raw, layout, resources, verbose: bool = True):
    """Return ``(predictions_post, changes)``.

    ``changes`` is the long-format table of every cell a rule touched. It is the
    evidence for whether a rule earns its place, and for a VLM condition it is
    also the evidence for *how much* of the post-processed number is correction
    rather than recognition. Report it.
    """
    predictions_post = postprocess_predictions(predictions_raw, layout, resources)
    changes = diff_predictions(predictions_raw, predictions_post)

    if verbose:
        n_cells = max(predictions_raw.size, 1)
        print(f"{len(changes)} of {predictions_raw.size} cells changed "
              f"({len(changes) / n_cells:.1%})")
        if len(changes):
            by_field = (changes.reset_index()
                        .groupby("field").size()
                        .sort_values(ascending=False)
                        .rename("cells_changed"))
            print(by_field.to_string())

    return predictions_post, changes


def manifest_entry(
    *,
    applied: bool,
    layout_name: str,
    resources: Optional[PostprocessResources] = None,
    changes=None,
    city_lexicon_csv: Optional[str] = None,
    comment_llm_path: Optional[str] = None,
) -> dict:
    """The ``"postprocess"`` block of a run manifest.

    Records which rule ran on which field, which of those rules carry a
    TrOCR-calibrated table, and how many cells were actually changed -- the
    three things needed to defend a post-processed VLM number later.
    """
    schema = schema_for_layout(layout_name)
    rules = {name: spec.get("postprocess", "none") for name, spec in schema.items()}

    entry = {
        "applied": bool(applied),
        "field_rules": rules,
        "model_calibrated_rules": {
            "rules": list(MODEL_CALIBRATED_RULES),
            "fields": sorted(n for n, r in rules.items() if r in MODEL_CALIBRATED_RULES),
            "note": ("digit-confusion table fitted to this project's TrOCR "
                     "train/val errors; not model-agnostic"),
        },
        "city_lexicon_csv": city_lexicon_csv,
        "comment_llm_path": comment_llm_path,
    }

    if resources is not None:
        entry["email_domains"] = sorted(resources.email_domains)
        entry["n_lexicon_entries"] = {k: len(v) for k, v in resources.lexicons.items()}
        entry["digit_map_is_default"] = (
            resources.digit_map == DEFAULT_DIGIT_CONFUSION_MAP
        )
        try:
            entry["thresholds"] = dict(vars(resources.config))
        except TypeError:  # slots or an exotic config object
            entry["thresholds"] = None

    if changes is not None:
        entry["n_cells_changed"] = int(len(changes))
        if len(changes):
            entry["cells_changed_by_field"] = (
                changes.reset_index().groupby("field").size().to_dict()
            )

    return entry


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------

def free_vram(verbose: bool = True) -> Optional[float]:
    """Collect garbage and empty the CUDA cache. Returns GiB still allocated.

    Only useful *after* the notebook has dropped its own references (``del
    model`` / ``runner.unload()``); a Python object referenced by a notebook
    variable is not freed by anything this function can do.

    Relevant because the ``llm`` rule loads a second model: a 7B corrector on
    top of a resident 8B VLM is a plausible OOM on a single 48 GB card.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    if verbose:
        print(f"VRAM allocated: {allocated:.2f} GiB")
    return allocated
