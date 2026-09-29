"""htr-form-app -- Streamlit front end for the thesis HTR pipeline.

A six-step wizard around ``htrpipe`` and the page-level VLM engines:

    1. Input        -- images, a PDF, or a server directory
    2. Fields       -- name, type, post-processing rule, expected pattern
    3. Model        -- recogniser, preprocessing preset, post-processing
    4. Model setup  -- TrOCR: layout file with measured ROIs;
                       VLMs: printed field labels; Chandra: stop labels
    5. Run          -- prompt preview, then extract every form
    6. Review       -- flags, per-form inspection and correction, export

Navigation is linear: "Next" unlocks only when the current step is complete,
so a run cannot start with half a configuration.

Design rule (same as appcore): this file contains UI wiring ONLY. Anything
that touches an image, a model or a value goes through htrpipe or the
engine modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import pathlib
import time

import pandas as pd
import streamlit as st
import yaml

from appcore import (
    engine_chandra_page,
    formconfig,
    engine_qwen_page,
    engine_trocr,
    inputs,
    lexicons,
    manifest as manifest_mod,
    models,
    plausibility,
    presets,
    schema_editor,
)
from htrpipe import load_layout_spec, make_run_name, validate_resources
from htrpipe.postprocess import (
    diff_predictions,
    ensure_text_frame,
    postprocess_predictions,
    write_predictions_text,
)
from htrpipe.postprocess_setup import (
    MODEL_CALIBRATED_RULES,
    build_resources,
    manifest_entry,
    tuned_config,
)

# ---------------------------------------------------------------------------
# Configuration file (paths only -- never pipeline settings)
# ---------------------------------------------------------------------------

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


@st.cache_data
def load_app_config() -> dict:
    """Read config.yaml once per session. Paths and model ids only."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_app_config()
LAYOUTS_DIR = pathlib.Path(__file__).parent / CFG["layouts_dir"]
LEXICONS_DIR = pathlib.Path(__file__).parent / CFG["lexicons_dir"]
OUTPUT_DIR = pathlib.Path(CFG["output_dir"])

MODEL_LABELS = {
    "trocr": "TrOCR",
    "qwen": "Qwen3-VL-8B",
    "chandra": "Chandra-OCR-2",
}
MODEL_PATHS = {
    "trocr": CFG["trocr_model_path"],
    "qwen": CFG["qwen_model_path"],
    "chandra": CFG["chandra_model_path"],
}

#: One-line notes under each model choice in step 3. Comparative statements
#: reflect what was observed in the thesis evaluation.
MODEL_NOTES = {
    "trocr": "Requires fine-tuned model.",
    "qwen": "RECOMMENDED: Supports in-context examples (few-shot) and confidence flags "
            "for human review.",
    "chandra": "No in-context examples or confidence flags. Observed in the "
               "thesis evaluation to handle struck-out text better than Qwen.",
}

STEPS = ["1. Input", "2. Fields", "3. Model & settings", "4. Model setup",
         "5. Run", "6. Review & export"]

st.set_page_config(page_title="HTR Form Extraction", layout="wide")


def _build_postprocess_resources(layout=None, verbose: bool = False):
    """Resources for the correction rules: thresholds, lexicon files, email
    domains, street suffixes, corrector model. One place, so step 3's dry
    validation and the real run in step 5 can never drift apart.

    All three word lists come from user-editable files in resources/: the
    named lexicons directory, email_domains.txt and street_suffixes.txt.
    """
    root = pathlib.Path(__file__).parent
    # layout=None on purpose: build_resources only uses the layout to print a
    # validation report, and it would print it BEFORE the lexicon files below
    # are loaded -- reporting a loaded lexicon as missing. The app validates
    # separately (step 3), after everything is loaded.
    res = build_resources(
        config=tuned_config(),
        email_domains=set(lexicons.read_list(root / CFG["email_domains_file"])),
        comment_llm_path=CFG["comment_llm_path"],
        layout=None, verbose=verbose,
    )
    lexicons.load_into(res, LEXICONS_DIR)
    # Street suffixes: file order preserved (ties break in favour of earlier
    # entries). An empty or missing file falls back to htrpipe's defaults.
    suffix_file = root / CFG["street_suffixes_file"]
    if suffix_file.is_file():
        suffixes = lexicons.read_list(suffix_file)
        if suffixes:
            res.street_suffixes = suffixes
    return res


def _save_configuration_ui(schema, labels=None, stop_labels=None,
                           source_layout=None) -> None:
    """Offer to save the current setup as a reusable form configuration.

    One JSON holds the field definitions plus whatever else is known: the
    measured regions (from a layout file) and the printed labels and stop
    labels (from this step). Saved into the layouts folder, so it appears in
    the configuration lists straight away.
    """
    with st.expander("Save this configuration for next time"):
        st.caption("Writes the fields, and whatever else is set up, to "
                   f"{CFG['layouts_dir']}/ as one JSON. Loading it in step 2 "
                   "fills in every step that applies.")
        default_name = S.loaded_config_name or "myform"
        name = st.text_input("Configuration name", value=default_name,
                             key="cfg_save_name")
        safe = "".join(ch for ch in name.strip()
                       if ch.isalnum() or ch in "-_")
        target = LAYOUTS_DIR / f"{safe}.json"
        if safe and target.exists():
            st.warning(f"`{safe}.json` exists and will be overwritten.")
        if st.button("Save configuration", disabled=not safe,
                     key="cfg_save_btn"):
            config = formconfig.build(
                schema, labels=labels, stop_labels=stop_labels,
                source_layout=source_layout, name=safe)
            formconfig.save(config, target)
            S.loaded_config_name = safe
            has_geometry = source_layout is not None
            st.success(
                f"Saved to `{CFG['layouts_dir']}/{safe}.json`"
                + ("." if has_geometry else
                   " -- without measured field regions, so it serves the "
                   "whole-page models only."))


def _build_demos(demo_paths: dict, columns: list) -> list:
    """``htrpipe.fewshot.Demo`` objects for the confirmed example pages.

    ``demo_paths`` maps doc_id -> the image file the model would read for
    that page (i.e. after the same page preprocessing as every other page).
    Each image is downscaled to the demonstration pixel budget here, before
    the processor sees it, exactly as the few-shot condition specifies.
    """
    from PIL import Image

    from htrpipe import fewshot

    demos = []
    for doc_id, path in demo_paths.items():
        raw = Image.open(path).convert("RGB")
        fitted = fewshot.fit_to_pixel_budget(raw, engine_qwen_page.DEMO_MAX_PIXELS)
        values = S.demo_values.get(doc_id, {})
        demos.append(fewshot.Demo(
            doc_id=doc_id, path=pathlib.Path(path),
            values={c: values.get(c, "") for c in columns},
            image=fitted, source_size=raw.size, prompt_size=fitted.size))
    return demos


def _text_columns(layout) -> list:
    """Output columns that hold free text (everything but checkbox groups)."""
    return [f.name for f in layout.fields
            if f.type not in ("ignore", "checkbox_group")]


def _demo_labelling_ui() -> None:
    """Step 4, few-shot: enter the correct values for the k example pages.

    One page at a time: the scan on the left, a field/value table on the
    right. The values become (a) the example answers shown to the model and
    (b) the output for these pages, which the model does not read.
    """
    st.divider()
    st.subheader("Example pages (few-shot)")
    demos = demo_forms()
    k = len(demos)
    S.demo_cursor = min(S.demo_cursor, k - 1)
    i = S.demo_cursor
    form = demos[i]
    st.write(f"Enter the values exactly as written on the form -- the model "
             f"treats them as the correct answer. **Example {i + 1} of {k}** "
             f"(doc_id {form.doc_id}).")

    columns = S.layout.output_columns
    options_by_field = {f.name: list(f.options or {})
                        for f in S.layout.checkbox_groups}
    stored = S.demo_values.get(form.doc_id, {})

    img_col, val_col = st.columns([1, 1])
    with img_col:
        st.image(form.image, caption=f"doc_id {form.doc_id}",
                 use_container_width=True)
    with val_col:
        table = pd.DataFrame({
            "field": columns,
            "value": [stored.get(c, "") for c in columns],
            "allowed": [("one of: " + " | ".join(options_by_field[c]))
                        if c in options_by_field else "" for c in columns],
        })
        edited = st.data_editor(
            table, key=f"demo_editor_{form.doc_id}", hide_index=True,
            use_container_width=True, height=560, num_rows="fixed",
            disabled=["field", "allowed"],
            column_config={"value": st.column_config.TextColumn(
                "value (editable)")})

    def save_current() -> bool:
        values = {r["field"]: str(r["value"] if r["value"] is not None else "").strip()
                  for _, r in edited.iterrows()}
        bad = [f"{name}: {values[name]!r} is not one of {opts}"
               for name, opts in options_by_field.items()
               if values.get(name) and values[name] not in opts]
        if bad:
            for b in bad:
                st.error(b)
            return False
        S.demo_values[form.doc_id] = values
        return True

    back, fwd = st.columns(2)
    with back:
        if st.button("< Previous example", disabled=i == 0,
                     use_container_width=True, key="demo_prev"):
            if save_current():
                S.demo_cursor -= 1
                st.rerun()
    with fwd:
        last = i == k - 1
        label = "Save and confirm examples" if last else "Save and next example >"
        if st.button(label, type="primary", use_container_width=True,
                     key="demo_next"):
            if save_current():
                if last:
                    missing = [d.doc_id for d in demos
                               if d.doc_id not in S.demo_values]
                    if missing:
                        st.error(f"Examples not filled in yet: doc_id {missing}")
                    else:
                        S.demo_confirmed = True
                        S.run = None
                        st.success(f"{k} example page(s) confirmed.")
                else:
                    S.demo_cursor += 1
                    st.rerun()

    if S.demo_confirmed:
        st.caption(f"Confirmed examples: doc_id "
                   f"{', '.join(str(d.doc_id) for d in demos)}.")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# Two kinds of keys live here. Plain data (forms, schema, run results) and
# the user's *selections*. Selections get their own sel_* keys, separate from
# widget keys, because Streamlit DELETES a widget's state as soon as its
# widget is not rendered -- which happens whenever the user changes step.

_DEFAULTS = {
    "step": 0,
    "forms": None,             # list[FormImage]
    "input_dir": None,
    "last_input_sig": None,    # fingerprint of the last auto-loaded input
    "schema_rows": None,       # step-2 table (list of dicts)
    "loaded_config_name": None,  # configuration the session started from
    "schema": None,            # validated {field: entry} from step 2
    "base_layout": None,       # step 4 (TrOCR): LayoutSpec from the file
    "layout": None,            # final LayoutSpec the run uses
    "layout_source": None,     # provenance string for the manifest
    "label_rows": None,        # step 4 (VLM): [{field, label}]
    "labels": None,            # confirmed {field: label or [variants]}
    "stop_labels_user": "",    # step 4 (Chandra): extra stop labels
    "run": None,
    "review_key": None,        # (run_name, stage) the edit frame belongs to
    "edited_frame": None,      # accumulated manual corrections (step 6)
    "sel_model": "qwen",
    "sel_page_preset": presets.EVALUATED_PAGE_PRESET["qwen"],
    "sel_field_preset": presets.EVALUATED_FIELD_PRESET["trocr"],
    "sel_apply_pp": False,
    # few-shot (Qwen only)
    "sel_fewshot": False,
    "sel_k": 1,
    "demo_values": {},         # {doc_id: {field: value}} entered in step 4
    "demo_confirmed": False,   # all k demonstration pages filled in
    "demo_cursor": 0,          # which demonstration page is on screen
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)
S = st.session_state


# ---------------------------------------------------------------------------
# Step gating
# ---------------------------------------------------------------------------

def fewshot_active() -> bool:
    """Few-shot applies only to the whole-page Qwen model."""
    return S.sel_model == "qwen" and bool(S.sel_fewshot)


def demo_forms() -> list:
    """The first k pages of the batch: the demonstration pages."""
    return list(S.forms[:S.sel_k]) if (fewshot_active() and S.forms) else []


def step_done(i: int) -> tuple[bool, str]:
    """(is the step complete, what is still missing). Drives Next."""
    if i == 0:
        return (bool(S.forms), "load at least one page")
    if i == 1:
        return (S.schema is not None, "apply the field table (Validate and apply)")
    if i == 2:
        if fewshot_active() and S.forms and len(S.forms) < S.sel_k + 1:
            return (False, f"few-shot with k={S.sel_k} needs at least "
                           f"{S.sel_k + 1} pages ({len(S.forms)} loaded)")
        return (True, "")
    if i == 3:
        if S.layout is None:
            return (False, "confirm the model-specific setup below")
        if fewshot_active() and not S.demo_confirmed:
            return (False, "fill in and confirm the demonstration page(s)")
        return (True, "")
    if i == 4:
        return (S.run is not None, "run the extraction")
    return (True, "")


def nav_buttons() -> None:
    """Previous / Next at the bottom of every step. Next is gated."""
    st.divider()
    left, middle, right = st.columns([1, 4, 1])
    with left:
        if st.button("< Previous", disabled=S.step == 0, use_container_width=True):
            S.step -= 1
            st.rerun()
    done, missing = step_done(S.step)
    with right:
        if S.step < len(STEPS) - 1:
            if st.button("Next >", type="primary", disabled=not done,
                         use_container_width=True):
                S.step += 1
                st.rerun()
    with middle:
        if not done and missing:
            st.caption(f"To continue: {missing}")


# ---------------------------------------------------------------------------
# Sidebar: progress overview (read-only) and VRAM
# ---------------------------------------------------------------------------

st.sidebar.title("HTR Form Extraction")
for i, name in enumerate(STEPS):
    done, _ = step_done(i)
    # Ticked = the user has moved past the step AND its condition holds.
    # Without the position check, steps whose condition is trivially
    # satisfied from the start (model selection has a default; the final
    # step has no condition) would appear pre-ticked in a fresh session.
    marker = "->" if i == S.step else ("[x]" if done and i < S.step else "[ ]")
    st.sidebar.markdown(f"`{marker}` {name}")
st.sidebar.divider()
st.sidebar.caption(f"GPU memory: {models.vram_status()}")
if st.sidebar.button("Free GPU memory (unload all models)"):
    gib = models.unload_all()
    st.sidebar.success(f"Unloaded. {gib:.1f} GiB still allocated." if gib >= 0
                       else "Unloaded.")


# ===========================================================================
# STEP 1 -- INPUT
# ===========================================================================

if S.step == 0:
    st.header("Step 1 -- Load the scanned forms")
    st.write(
        "All pages must be the **same form layout**, and each form must fit "
        "on **one page**. Every uploaded page is treated as a separate form, "
        "so a two-page form cannot be processed in one go: run the first "
        "pages as one batch and the second pages as another."
    )

    route = st.radio("Input source", ["Upload images", "Upload a PDF"])

    # Loading happens the moment the selection changes -- no second "load"
    # click. The fingerprint tells a Streamlit rerun (every widget
    # interaction) apart from an actually new selection.
    new_forms = None
    sig = None

    if route == "Upload images":
        uploaded = st.file_uploader(
            "Scanned pages", accept_multiple_files=True,
            type=[s.lstrip(".") for s in inputs.ACCEPTED_IMAGE_SUFFIXES])
        if uploaded:
            sig = ("images", tuple(sorted((f.name, f.size) for f in uploaded)))
            if sig != S.last_input_sig:
                with st.spinner("Loading images ..."):
                    work_dir = inputs.new_session_dir()
                    n = inputs.save_uploaded_images(uploaded, work_dir)
                    if n < len(uploaded):
                        st.warning(f"{len(uploaded) - n} file(s) skipped "
                                   f"(unsupported type).")
                    new_forms = inputs.load_forms(work_dir)
                    S.input_dir = work_dir

    elif route == "Upload a PDF":
        pdf = st.file_uploader("PDF with one form per page", type=["pdf"])
        if pdf:
            sig = ("pdf", pdf.name, pdf.size)
            if sig != S.last_input_sig:
                work_dir = inputs.new_session_dir()
                with st.spinner("Rendering PDF pages ..."):
                    n = inputs.render_pdf_to_images(pdf.getvalue(), work_dir)
                st.info(f"Rendered {n} page(s).")
                new_forms = inputs.load_forms(work_dir)
                S.input_dir = work_dir

    if new_forms is not None:
        S.forms = new_forms
        S.last_input_sig = sig
        S.run = None          # new input invalidates old results

    if S.forms:
        n, summary = inputs.describe_forms(S.forms)
        st.success(f"Loaded: {summary}")
        st.image(S.forms[0].image,
                 caption=f"First page (doc_id {S.forms[0].doc_id})", width=420)
        if "MIXED" in summary:
            st.warning("Pages have different sizes. For the TrOCR path this "
                       "usually means mixed scan resolutions -- expect ROI "
                       "misalignment.")

    nav_buttons()


# ===========================================================================
# STEP 2 -- FIELDS (model-independent semantics)
# ===========================================================================

elif S.step == 1:
    st.header("Step 2 -- Which fields to extract, and how")
    st.write("Model-specific details (field positions, printed labels) come "
             "later, in step 4 -- here only what the fields *are*.")

    table_col, preview_col = st.columns([3, 2])

    with preview_col:
        st.subheader("First form")
        if S.forms:
            st.image(S.forms[0].image,
                     caption=f"doc_id {S.forms[0].doc_id}",
                     use_container_width=True)
        else:
            st.info("No forms loaded yet (step 1).")

    with table_col:
        # Optional prefill from a layout file -- also the easiest way to get
        # field names that match a layout for the TrOCR path.
        with st.expander("Start from a saved form configuration"):
            st.caption("A configuration holds the field definitions and, "
                       "where available, the measured field regions "
                       "(TrOCR), the printed labels and the stop labels "
                       "(whole-page models). Loading one fills in every "
                       "step that applies.")
            configs = formconfig.available_configs(LAYOUTS_DIR)
            if not configs:
                st.info(f"No configurations in {CFG['layouts_dir']}/ yet.")
            else:
                pick = st.selectbox("Configuration", configs)
                if st.button("Load this configuration"):
                    raw = formconfig.read_raw(LAYOUTS_DIR / f"{pick}.json")
                    if formconfig.has_rois(raw):
                        S.schema_rows = schema_editor.rows_from_layout(
                            formconfig.load_layout(LAYOUTS_DIR / f"{pick}.json"))
                    else:
                        # No measured regions: read the plain dict, because
                        # htrpipe requires a region per field.
                        S.schema_rows = formconfig.rows_from_raw(
                            raw, schema_editor.pattern_columns_from_regex,
                            schema_editor.INTERNAL_TO_UI)
                    # Labels and stop labels go straight to step 4.
                    stored_labels = formconfig.labels_from_raw(raw)
                    if stored_labels:
                        S.label_rows = [
                            {"field": f["name"],
                             "printed label": (" | ".join(stored_labels[f["name"]])
                                               if isinstance(stored_labels.get(f["name"]),
                                                             (list, tuple))
                                               else stored_labels.get(f["name"], ""))}
                            for f in raw.get("fields") or []]
                    stops = formconfig.stop_labels_from_raw(raw)
                    if stops:
                        S.stop_labels_user = " | ".join(stops)
                    S.loaded_config_name = pick
                    S.schema = None
                    S.layout = None
                    st.rerun()

        if S.schema_rows is None:
            S.schema_rows = [{c: "" for c in schema_editor.ROW_COLUMNS}
                             for _ in range(3)]

        st.markdown(
            "- **short** -- a single-line entry (name, date, postal code). "
            "Add an expected pattern for numeric ones.\n"
            "- **long** -- a multi-line entry (a comment box); recognised "
            "line by line and joined afterwards.\n"
            "- **checkbox** -- a group of boxes; needs its option names.\n"
            "- **ignore** -- not transcribed (a signature). For the "
            "whole-page models its printed label is still useful as a "
            "boundary marker."
        )
        lexicon_names = lexicons.available_lexicons(LEXICONS_DIR)
        edited_rows = st.data_editor(
            pd.DataFrame(S.schema_rows, columns=schema_editor.ROW_COLUMNS),
            num_rows="dynamic", use_container_width=True, key="schema_table",
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "type", options=schema_editor.UI_TYPE_CHOICES,
                    required=True),
                "postprocess": st.column_config.SelectboxColumn(
                    "postprocess", options=sorted(
                        __import__("htrpipe").POSTPROCESS_RULES),
                    help="Optional correction rule. Not applicable to "
                         "checkbox/ignore fields."),
                "pattern": st.column_config.SelectboxColumn(
                    "expected pattern", options=schema_editor.PATTERN_CHOICES,
                    help="Expected format of the value (required for rule "
                         "'numeric'). Pick a named format, or 'custom "
                         "regex' and fill the next column."),
                "custom_pattern": st.column_config.TextColumn(
                    "custom regex",
                    help="Your own regular expression -- used only when "
                         "expected pattern is 'custom regex'."),
                "options": st.column_config.TextColumn(
                    "checkbox options",
                    help="checkbox fields only: option names separated "
                         "by |. With a TrOCR layout file, leave empty to "
                         "inherit the measured options."),
                "lexicon": st.column_config.SelectboxColumn(
                    "lexicon", options=[""] + lexicon_names,
                    help="For rule 'lexicon': which list of valid values to "
                         "correct against. Lists are the files in "
                         f"{CFG['lexicons_dir']}/ -- edit or add files "
                         "there."),
            },
        ).to_dict(orient="records")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Validate and apply", type="primary"):
                schema, problems, notices = schema_editor.rows_to_schema(edited_rows)
                for note in notices:
                    st.info(note)
                if problems:
                    for prob in problems:
                        st.error(prob)
                    S.schema = None
                elif not schema:
                    st.error("Define at least one field.")
                    S.schema = None
                else:
                    S.schema = schema
                    S.schema_rows = edited_rows
                    S.layout = None     # step 4 must confirm again
                    S.run = None
                    st.success(f"{len(schema)} field(s) defined.")
        with col_b:
            st.download_button(
                "Save this field table as JSON",
                data=pd.DataFrame(edited_rows).to_json(orient="records",
                                                       indent=2),
                file_name="field_schema.json")
            loaded = st.file_uploader("... or load a saved table", type=["json"])
            if loaded is not None and st.button("Use loaded table"):
                S.schema_rows = schema_editor.load_rows(loaded)
                S.schema = None
                st.rerun()

    nav_buttons()


# ===========================================================================
# STEP 3 -- MODEL & PROCESSING SETTINGS
# ===========================================================================

elif S.step == 2:
    st.header("Step 3 -- Recogniser, preprocessing, post-processing")

    model_kind = st.radio(
        "Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
        captions=[MODEL_NOTES[k] for k in MODEL_LABELS],
        index=list(MODEL_LABELS).index(S.sel_model))
    if model_kind != S.sel_model:
        S.layout = None          # model change invalidates step 4
        # Each model starts at ITS standard configuration; deviations are a
        # deliberate act, not a leftover from the previously selected model.
        S.sel_page_preset = presets.EVALUATED_PAGE_PRESET[model_kind]
        if model_kind == "trocr":
            S.sel_field_preset = presets.EVALUATED_FIELD_PRESET["trocr"]
    S.sel_model = model_kind

    # ---- preprocessing presets ------------------------------------------
    st.subheader("Preprocessing")
    ev_page = presets.EVALUATED_PAGE_PRESET[model_kind]
    page_labels = {p: f"{p}  (evaluated)" if p == ev_page else p
                   for p in presets.PAGE_PRESETS}
    if S.sel_page_preset not in presets.PAGE_PRESETS:
        S.sel_page_preset = ev_page
    page_preset = st.selectbox(
        "Page-level preset", presets.PAGE_PRESETS,
        index=presets.PAGE_PRESETS.index(S.sel_page_preset),
        format_func=page_labels.get)
    S.sel_page_preset = page_preset

    field_preset = presets.FIELD_NONE
    if model_kind == "trocr":
        ev_field = presets.EVALUATED_FIELD_PRESET["trocr"]
        valid = presets.valid_field_presets(page_preset)
        field_labels_map = {p: f"{p}  (evaluated)" if p == ev_field else p
                            for p in valid}
        if S.sel_field_preset not in valid:
            S.sel_field_preset = ev_field if ev_field in valid else valid[0]
        field_preset = st.selectbox(
            "Field-crop preset (TrOCR only)", valid,
            index=valid.index(S.sel_field_preset),
            format_func=field_labels_map.get)
        S.sel_field_preset = field_preset
        cfg_preview = presets.build_pipeline_config(
            page_preset, field_preset, CFG["trocr_model_path"])
        with st.expander("All pipeline settings these presets expand to "
                         "(read-only)"):
            # The complete configuration, kernel sizes and all -- the same
            # dump the run manifests record.
            st.code(cfg_preview.describe(), language=None)
    else:
        st.caption("Field-crop preprocessing does not apply to whole-page "
                   "models -- there are no field crops.")
        cfg_page = presets.build_pipeline_config(
            page_preset, presets.FIELD_NONE, CFG["trocr_model_path"])
        page_lines = [ln for ln in cfg_page.describe().splitlines()
                      if ln.startswith("page.")]
        if model_kind == "qwen":
            eng = engine_qwen_page
            engine_lines = [
                f"vision.min_pixels = {eng.MIN_PIXELS}",
                f"vision.max_pixels = {eng.MAX_PIXELS}",
                f"decoding.max_new_tokens = {eng.MAX_NEW_TOKENS}",
                f"decoding.do_sample = {eng.DO_SAMPLE}",
                f"decoding.repetition_penalty = {eng.REPETITION_PENALTY}",
                f"decoding.seed = {eng.SEED}",
                f"decoding.max_parse_retries = {eng.MAX_PARSE_RETRIES}",
                f"retry_suffix = {eng.RETRY_SUFFIX.strip()!r}",
                f"confidence.score = 'conf_min'",
                f"confidence.threshold = {plausibility.CONF_MIN_THRESHOLD} "
                f"(heuristic, not calibrated on validation data)",
            ]
            if S.sel_fewshot:
                from htrpipe import fewshot as _fs
                engine_lines += [
                    f"fewshot.k = {S.sel_k}",
                    f"fewshot.demo_max_pixels = {eng.DEMO_MAX_PIXELS} "
                    f"(~{eng.DEMO_MAX_PIXELS // (28 * 28)} visual tokens per example)",
                    f"fewshot.query_preamble = {_fs.QUERY_PREAMBLE_EN!r}",
                    f"fewshot.closing = {_fs.CLOSING_INSTRUCTION!r}",
                    f"fewshot.retry_suffix = {eng.FEWSHOT_RETRY_SUFFIX.strip()!r}",
                ]
        else:
            eng = engine_chandra_page
            engine_lines = [
                f"backend = {eng.BACKEND!r}",
                f"prompt_type = {eng.PROMPT_TYPE!r}",
                f"vision.min_pixels = {eng.MIN_PIXELS}",
                f"vision.max_pixels = {eng.MAX_PIXELS}",
                f"decoding.max_new_tokens = {eng.MAX_NEW_TOKENS}",
                f"decoding.do_sample = {eng.DO_SAMPLE}",
                f"decoding.repetition_penalty = {eng.REPETITION_PENALTY}",
                f"decoding.seed = {eng.SEED}",
            ]
        with st.expander("All settings this configuration expands to "
                         "(read-only)"):
            st.code("\n".join(page_lines + engine_lines), language=None)

    # ---- few-shot (Qwen only) ----------------------------------------------
    if model_kind == "qwen":
        st.subheader("In-context examples (few-shot)")
        use_fs = st.checkbox(
            "Show the model solved example forms before each page",
            value=S.sel_fewshot,
            help="The first k pages of the batch become examples: you enter "
                 "their values by hand in step 4, they are shown to the model "
                 "together with each remaining page, and their hand-entered "
                 "values go straight into the output (they are not "
                 "recognised by the model).")
        k = S.sel_k
        if use_fs:
            k = int(st.number_input("Number of example pages (k)", min_value=1,
                                    max_value=10, value=int(S.sel_k), step=1))
            st.caption(f"Pages 1-{k} of the batch become examples; the model "
                       f"reads the remaining pages. Each example costs about "
                       f"{engine_qwen_page.DEMO_MAX_PIXELS // (28 * 28)} "
                       f"visual tokens per page read.")
        if use_fs != S.sel_fewshot or k != S.sel_k:
            S.demo_confirmed = False       # examples must be re-confirmed
            S.demo_cursor = 0
        S.sel_fewshot, S.sel_k = use_fs, k

    matches = presets.config_matches_evaluated(model_kind, page_preset,
                                               field_preset) and not fewshot_active()
    if fewshot_active():
        st.info("Few-shot is a separate condition that was NOT part of the "
                "thesis evaluation. It will run, and the export manifest "
                "records it as unevaluated.")
    elif matches:
        st.success("This configuration matches the one evaluated in the thesis.")
    else:
        st.info("This configuration was NOT part of the thesis evaluation. "
                "It will run, and the export manifest records it as "
                "unevaluated.")

    # ---- post-processing -------------------------------------------------
    st.subheader("Post-processing (optional, separate condition)")
    apply_pp = st.checkbox("Apply the field correction rules after extraction",
                           value=S.sel_apply_pp)
    S.sel_apply_pp = apply_pp

    if apply_pp and S.schema is not None:
        rules = {name: e.get("postprocess", "none")
                 for name, e in S.schema.items()}
        if any(r == "llm" for r in rules.values()):
            st.warning(
                "The `llm` rule loads a second model "
                f"({CFG['comment_llm_path']}). The recognition model is "
                "released from GPU memory first, as in the evaluation.")

        # Dry resource check on a provisional layout built from the schema.
        # Checkbox fields may not have options yet (a TrOCR layout supplies
        # them in step 4), so placeholders stand in -- rules never apply to
        # checkbox fields, so the validation outcome is unaffected.
        provisional = {
            name: ({**e, "options": e.get("options") or ["A", "B"]}
                   if e["type"] == "checkbox_group" else e)
            for name, e in S.schema.items()
        }
        dry_layout = schema_editor.build_custom_layout(provisional)
        dry = _build_postprocess_resources(layout=None)
        problems = validate_resources(dry_layout, dry)
        if problems:
            st.error("These rules would silently do nothing as configured:")
            for prob in problems:
                st.write(f"- {prob}")
        else:
            st.caption("All selected rules have the resources they need.")

        with st.expander("All post-processing settings (read-only)"):
            pp_lines = [f"{k} = {v!r}"
                        for k, v in vars(tuned_config()).items()]
            root = pathlib.Path(__file__).parent
            pp_lines += [
                f"lexicons_dir = {CFG['lexicons_dir']!r}",
                f"lexicons = {lexicons.available_lexicons(LEXICONS_DIR)!r}",
                f"email_domains ({CFG['email_domains_file']}) = "
                f"{lexicons.read_list(root / CFG['email_domains_file'])!r}",
                f"street_suffixes ({CFG['street_suffixes_file']}) = "
                f"{lexicons.read_list(root / CFG['street_suffixes_file'])!r}",
                f"comment_llm_path = {CFG['comment_llm_path']!r}",
            ]
            st.code("\n".join(pp_lines), language=None)

    nav_buttons()


# ===========================================================================
# STEP 4 -- MODEL-SPECIFIC SETUP
# ===========================================================================

elif S.step == 3:
    st.header("Step 4 -- Model-specific setup")
    model_kind = S.sel_model
    schema = S.schema
    if schema is None:
        st.warning("The field table changed -- apply it in step 2 again.")
        nav_buttons()
        st.stop()

    # ------------------------------------------------------------ TrOCR --
    if model_kind == "trocr":
        st.subheader("Field positions (layout file)")
        st.write("TrOCR reads each field from a measured region of the page. "
                 "The regions come from a configuration JSON; field names in "
                 "the file must match the table from step 2 exactly.")

        # Only configurations that actually carry measured regions can drive
        # this path; ones defined by hand (regions null) are filtered out.
        with_rois = [
            name for name in formconfig.available_configs(LAYOUTS_DIR)
            if formconfig.has_rois(formconfig.read_raw(
                LAYOUTS_DIR / f"{name}.json"))]
        source = st.selectbox("Configuration",
                              with_rois + ["upload a layout JSON"])
        base = None
        if source == "upload a layout JSON":
            up = st.file_uploader("Layout JSON (htrpipe format)", type=["json"])
            if up is not None:
                tmp = inputs.new_session_dir(prefix="htr_app_layout_") / up.name
                tmp.write_bytes(up.getbuffer())
                base = load_layout_spec(tmp)
                S.layout_source = f"uploaded:{up.name}"
        elif with_rois:
            base = load_layout_spec(LAYOUTS_DIR / f"{source}.json")
            S.layout_source = f"built-in:{source}"
        else:
            st.warning(f"No configuration in {CFG['layouts_dir']}/ has "
                       f"measured field regions. Upload one, or use a "
                       f"whole-page model.")

        if base is not None and st.button("Confirm layout", type="primary"):
            try:
                S.base_layout = base
                S.layout = schema_editor.apply_schema_to_layout(base, schema)
                S.labels = None
                S.run = None
                st.success(f"Layout `{base.name}` confirmed: "
                           f"{len(S.layout.output_columns)} field(s).")
            except ValueError as exc:
                S.layout = None
                st.error(str(exc))

        if S.layout is not None:
            st.subheader("Field regions check")
            if not S.forms:
                st.warning("Load the forms in step 1 to preview the regions.")
            elif st.button("Show field regions on the first page"):
                from htrpipe import preprocess_page
                cfg_preview = presets.build_pipeline_config(
                    S.sel_page_preset, S.sel_field_preset,
                    CFG["trocr_model_path"])
                page0 = preprocess_page(S.forms[0].image, cfg_preview.page)
                rois = S.layout.reference_rois(page0)
                import matplotlib.pyplot as plt
                fig = engine_trocr.make_roi_overlay(page0, rois)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                plt.close(fig)
                st.image(buf, width=520)
                st.caption("Boxes must sit on the handwriting. If they are "
                           "shifted, the scans do not match the layout "
                           "file's resolution or geometry.")

            _save_configuration_ui(schema, labels=None, stop_labels=None,
                                   source_layout=S.base_layout)

    # ------------------------------------------------------------- VLMs --
    else:
        st.subheader("Printed field labels")
        st.write("Enter the labels exactly as printed on the form; several "
                 "accepted spellings go separated by `|`.")
        st.caption("Fields of type 'ignore' are never transcribed, but their "
                   "label is still worth entering: it marks where the "
                   "previous field ends. Without it, a preceding long text "
                   "can run on into the ignored field's content.")

        labels_col, form_col = st.columns([3, 2])
        with form_col:
            st.markdown("**First form (reference)**")
            if S.forms:
                st.image(S.forms[0].image,
                         caption=f"doc_id {S.forms[0].doc_id} -- copy the "
                                 f"labels exactly as printed here",
                         use_container_width=True)
            else:
                st.info("No forms loaded yet (step 1).")

        field_names = list(schema)
        if S.label_rows is None or [r["field"] for r in S.label_rows] != field_names:
            defaults = engine_chandra_page.DEFAULT_FIELD_LABELS
            S.label_rows = []
            for name in field_names:
                d = defaults.get(name, "")
                if isinstance(d, (list, tuple)):
                    d = " | ".join(d)
                S.label_rows.append({"field": name, "printed label": d})

        with labels_col:
            label_rows = st.data_editor(
                pd.DataFrame(S.label_rows), key="label_table",
                use_container_width=True, num_rows="fixed", hide_index=True,
                disabled=["field"],
            ).to_dict(orient="records")

            if model_kind == "chandra":
                st.subheader("Stop labels (Chandra)")
                st.write("Printed text that FOLLOWS the last field, so the "
                         "extractor can close the final field's span. "
                         "Built-in stop labels: "
                         + " | ".join(engine_chandra_page.DEFAULT_STOP_LABELS))
                S.stop_labels_user = st.text_input(
                    "Additional stop labels (`|`-separated)",
                    value=S.stop_labels_user)

        if st.button("Confirm labels", type="primary"):
            # Labels of IGNORED fields are kept. The extractor uses every
            # known label as a boundary marker, and only transcribes the
            # non-ignored ones -- so "Unterschrift" stops the comment field
            # from swallowing the signature. Required for transcribed
            # fields, optional (but recommended) for ignored ones.
            labels = {}
            missing = []
            for row in label_rows:
                text = str(row["printed label"]).strip()
                name = row["field"]
                is_ignored = schema[name]["type"] == "ignore"
                if not text:
                    if not is_ignored:
                        missing.append(name)
                    continue
                variants = [v.strip() for v in text.split("|") if v.strip()]
                labels[name] = variants[0] if len(variants) == 1 else variants
            unlabelled_ignored = [
                row["field"] for row in label_rows
                if schema[row["field"]]["type"] == "ignore"
                and not str(row["printed label"]).strip()]
            if missing:
                st.error(f"Printed labels missing for: {', '.join(missing)}")
                S.layout = None
            else:
                try:
                    S.layout = schema_editor.build_custom_layout(schema)
                    S.labels = labels
                    S.demo_confirmed = False
                    S.label_rows = label_rows
                    S.layout_source = "fields:step2"
                    S.run = None
                    st.success(f"Labels confirmed for "
                               f"{len(labels)} field(s).")
                    if unlabelled_ignored:
                        st.warning(
                            "No label for ignored field(s): "
                            + ", ".join(unlabelled_ignored)
                            + ". They will not act as boundaries, so the "
                              "field printed before them may absorb their "
                              "content.")
                except ValueError as exc:
                    S.layout = None
                    st.error(str(exc))

        if S.layout is not None and S.labels is not None:
            stops = [s.strip() for s in S.stop_labels_user.split("|")
                     if s.strip()] if model_kind == "chandra" else None
            # A layout file supplies the measured regions when the field
            # names match it -- so a configuration saved here stays usable
            # for TrOCR too, instead of silently losing the geometry.
            source = S.base_layout
            if source is not None and {f.name for f in source.fields} != set(schema):
                source = None
            _save_configuration_ui(schema, labels=S.labels, stop_labels=stops,
                                   source_layout=source)

        # ---- few-shot: the user fills in the example pages ------------------
        if fewshot_active() and S.layout is not None and S.labels is not None:
            _demo_labelling_ui()

    nav_buttons()


# ===========================================================================
# STEP 5 -- RUN
# ===========================================================================

elif S.step == 4:
    st.header("Step 5 -- Extract all forms")
    layout = S.layout
    forms = S.forms
    if layout is None or not forms:
        st.warning("The configuration changed -- complete steps 1-4 again.")
        nav_buttons()
        st.stop()

    model_kind = S.sel_model
    page_preset = S.sel_page_preset
    field_preset = S.sel_field_preset if model_kind == "trocr" else presets.FIELD_NONE
    apply_pp = S.sel_apply_pp

    st.write(f"Model: **{MODEL_LABELS[model_kind]}** | pages: **{len(forms)}** | "
             f"page preset: **{page_preset}**"
             + (f" | field preset: **{field_preset}**" if model_kind == "trocr" else "")
             + f" | post-processing: **{'on' if apply_pp else 'off'}**")

    # ---- prompt preview (VLM): what will actually be sent per page -------
    prompt_block = None
    if model_kind == "qwen":
        target_fields = [f for f in layout.fields if f.type != "ignore"]
        checkbox_fields = {f.name: list(f.options) for f in
                           layout.checkbox_groups if f.options}
        user_prompt = engine_qwen_page.build_prompt(
            target_fields, checkbox_fields, dict(S.labels))
        prompt_block = {"system": engine_qwen_page.SYSTEM_PROMPT,
                        "user": user_prompt}
        if fewshot_active():
            from htrpipe import fewshot
            from PIL import Image
            demo_ids = [f.doc_id for f in demo_forms()]
            query_forms = [f for f in forms if f.doc_id not in demo_ids]
            preview_demos = _build_demos(
                {f.doc_id: f.path for f in demo_forms()}, layout.output_columns)
            preview_msgs, preview_imgs = engine_qwen_page.build_page_messages(
                Image.open(query_forms[0].path).convert("RGB"), user_prompt,
                preview_demos, layout.output_columns)
            rendered = fewshot.render_messages(
                preview_msgs, processor_max_pixels=engine_qwen_page.MAX_PIXELS)
            prompt_block.update({
                "query_preamble": fewshot.QUERY_PREAMBLE_EN,
                "closing": fewshot.CLOSING_INSTRUCTION,
                "retry_suffix": engine_qwen_page.FEWSHOT_RETRY_SUFFIX,
                "rendered_first_page": rendered,
            })
            with st.expander("Prompt that will be sent (per page, examples "
                             "included)", expanded=False):
                st.caption(f"{len(preview_imgs)} images per call: "
                           f"{len(preview_demos)} example(s) + the page to "
                           f"read. Shown for the first page to read "
                           f"(doc_id {query_forms[0].doc_id}).")
                st.code(rendered, language=None)
        else:
            prompt_block["retry_suffix"] = engine_qwen_page.RETRY_SUFFIX
            with st.expander("Prompt that will be sent (per page)", expanded=False):
                st.markdown("**System prompt**")
                st.code(engine_qwen_page.SYSTEM_PROMPT, language=None)
                st.markdown("**User prompt**")
                st.code(user_prompt, language=None)
    elif model_kind == "chandra":
        user_stops = [s.strip() for s in S.stop_labels_user.split("|") if s.strip()]
        stop_labels = engine_chandra_page.DEFAULT_STOP_LABELS + user_stops
        try:
            from htrpipe.chandra_ocr import resolve_vendor_prompt
            prompt_text, prompt_source = resolve_vendor_prompt(
                engine_chandra_page.PROMPT_TYPE)
        except Exception as exc:
            prompt_text, prompt_source = (
                f"(unavailable before first model load: {exc})", "unresolved")
        prompt_block = {
            "type": engine_chandra_page.PROMPT_TYPE,
            "source": prompt_source,
            "text": prompt_text,
            "field_labels": dict(S.labels or {}),
            "stop_labels": stop_labels,
        }
        with st.expander("Prompt that will be sent (per page)", expanded=False):
            st.caption("Chandra uses the vendor's fixed OCR prompt; the field "
                       "labels below drive the extractor that reads its "
                       "output, not the prompt itself.")
            st.code(prompt_text, language=None)
            st.markdown("**Field labels for extraction:** "
                        + ", ".join(f"{k} <- {v}" for k, v in (S.labels or {}).items()))
            st.markdown("**Stop labels:** " + " | ".join(stop_labels))

    if st.button("Run extraction", type="primary"):
        run_name = make_run_name(layout.name, MODEL_PATHS[model_kind], "app")
        run_dir = OUTPUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = st.progress(0.0)
        status_box = st.status("Running ...", expanded=True)
        t_start = time.perf_counter()

        # When post-processing follows, recognition fills only part of the
        # bar -- a full bar should mean the run is finished, not that the
        # slowest stage is done and something else is still working.
        recognition_share = 0.9 if apply_pp else 1.0

        def page_progress(i: int, n: int, msg: str) -> None:
            progress_bar.progress(i / n * recognition_share)
            status_box.write(f"[{i}/{n}] {msg}")

        # Only one recognition model may live on the GPU at a time.
        models.unload_all(except_kind=model_kind)

        checkbox_results = None
        confidences = None      # {doc_id: {field: conf_min}} (Qwen only)
        demos, demo_ids = None, []
        diagnostics = {}
        model_meta: dict = {}
        runlog = None
        failures: list = []

        if model_kind == "trocr":
            cfg = presets.build_pipeline_config(
                page_preset, field_preset, CFG["trocr_model_path"])
            engine = models.load_trocr(CFG["trocr_model_path"])
            out = engine_trocr.run(forms, layout, cfg, engine,
                                   progress=status_box.write)
            progress_bar.progress(recognition_share)
            predictions_raw = out["predictions_raw"]
            checkbox_results = out["checkbox_results"]
            diagnostics = {"alignment_report": out["alignment_report"],
                           "checkbox_report": out["checkbox_report"]}
            model_meta = {"model_path": CFG["trocr_model_path"],
                          "engine": "trocr", "num_beams": 1, "batch_size": 8}

        else:
            # Whole-page models read image FILES.
            #
            # Preset "none": the raw scan paths go straight in.
            #
            # Preset "deskew only" (the evaluated pipeline): htrpipe's deskew
            # is applied to each page IN COLOUR (deskew preserves channels)
            # and written to a temp PNG. Below its 1-degree minimum angle the
            # function returns the image unchanged, so straight scans reach
            # the model pixel-identical to the raw condition -- which is how
            # the thesis runs behaved.
            #
            # The full preset goes through preprocess_page (which returns
            # grayscale by design) and is unevaluated anyway.
            form_pairs = [(f.doc_id, f.path) for f in forms]
            if page_preset != presets.PAGE_NONE:
                from PIL import Image

                prep_dir = inputs.new_session_dir(prefix="htr_app_prep_")
                status_box.write("Applying page preprocessing before the VLM ...")
                if page_preset == presets.PAGE_DESKEW:
                    from htrpipe.preprocess import deskew
                    processed = [deskew(f.image) for f in forms]
                else:
                    from htrpipe import preprocess_page
                    cfg_tmp = presets.build_pipeline_config(
                        page_preset, presets.FIELD_NONE, CFG["trocr_model_path"])
                    processed = [preprocess_page(f.image, cfg_tmp.page)
                                 for f in forms]
                form_pairs = []
                for f, arr in zip(forms, processed):
                    prep_path = prep_dir / f"prep-{f.doc_id}.png"
                    Image.fromarray(arr).save(prep_path)
                    form_pairs.append((f.doc_id, str(prep_path)))

            target_fields = [f for f in layout.fields if f.type != "ignore"]
            checkbox_fields = {f.name: list(f.options) for f in
                               layout.checkbox_groups if f.options}
            labels = dict(S.labels or {})

            if model_kind == "qwen":
                # Few-shot: the first k pages are examples. They are excluded
                # from inference; their hand-entered values are the output.
                demo_ids = [f.doc_id for f in demo_forms()]
                demo_paths = {d: pth for d, pth in form_pairs if d in demo_ids}
                query_pairs = [(d, pth) for d, pth in form_pairs
                               if d not in demo_ids]
                demos = (_build_demos(demo_paths, layout.output_columns)
                         if demo_ids else None)

                model, processor, model_meta = models.load_qwen(CFG["qwen_model_path"])

                if demos:
                    from htrpipe import fewshot
                    from PIL import Image
                    # Leakage guard: no example page may also be predicted.
                    fewshot.assert_disjoint(demos, [d for d, _ in query_pairs])
                    # Plumbing guard: prove every example image reaches the
                    # model before spending a whole batch on it.
                    check_msgs, check_imgs = engine_qwen_page.build_page_messages(
                        Image.open(query_pairs[0][1]).convert("RGB"),
                        prompt_block["user"], demos, layout.output_columns)
                    try:
                        verify_report = fewshot.verify_prompt_images(
                            processor, check_msgs, check_imgs)
                    except AssertionError as exc:
                        status_box.update(label="Stopped", state="error")
                        st.error("The example images would not reach the model, "
                                 "so the run was stopped before it started:\n\n"
                                 f"{exc}")
                        st.stop()
                    status_box.write(
                        f"Verified: {verify_report['n_images_encoded']} images "
                        f"reach the model ({len(demos)} example(s) + 1 page).")
                    prompt_block["verify_report"] = verify_report

                rows, runlog, failures, confidences = engine_qwen_page.run_batch(
                    query_pairs, model, processor, prompt_block["user"],
                    layout.output_columns, demos=demos, progress=page_progress)

                for d in demo_ids:          # hand-entered values go straight in
                    rows[d] = {c: S.demo_values[d].get(c, "")
                               for c in layout.output_columns}
            else:
                runner, model_meta = models.load_chandra(CFG["chandra_model_path"])
                if prompt_block["source"] == "unresolved":
                    try:
                        from htrpipe.chandra_ocr import resolve_vendor_prompt
                        prompt_block["text"], prompt_block["source"] = \
                            resolve_vendor_prompt(engine_chandra_page.PROMPT_TYPE)
                    except Exception:
                        pass
                rows, runlog, failures = engine_chandra_page.run_batch(
                    form_pairs, runner, target_fields, labels, checkbox_fields,
                    layout.output_columns, raw_text_dir=run_dir / "rawtext",
                    stop_labels=prompt_block["stop_labels"],
                    progress=page_progress)

            predictions_raw = (
                pd.DataFrame.from_dict(rows, orient="index")
                  .reindex(columns=layout.output_columns)
                  .rename_axis("doc_id")
                  .sort_index()
            )
            predictions_raw = ensure_text_frame(predictions_raw,
                                                name="predictions_raw")

            if demos:
                from htrpipe import fewshot
                query_frame = predictions_raw.drop(index=demo_ids)
                bleed = fewshot.demo_bleed_report(query_frame, demos,
                                                  _text_columns(layout))
                prompt_block["bleed_raw"] = {
                    "n_cells_matching_a_demo": int(bleed["n_matching_a_demo"].sum()),
                    "per_field_rate": {k: (None if pd.isna(v) else float(v))
                                       for k, v in bleed["rate"].items()},
                }

        seconds_total = time.perf_counter() - t_start
        status_box.write(f"Recognition finished in {seconds_total:.1f} s. "
                         f"Failures: {len(failures)}.")

        # ---- post-processing (separate condition) ------------------------
        predictions_post, changes, pp_block = None, None, None
        if apply_pp:
            if model_kind != "trocr":
                status_box.write("Releasing the recognition model before the "
                                 "corrector (as in the evaluation) ...")
                models.unload_all()
            resources = _build_postprocess_resources(layout=layout)
            status_box.write("Applying field correction rules ...")
            # Example pages hold hand-entered values, not model output, so
            # the correction rules are not applied to them.
            to_correct = predictions_raw.drop(index=demo_ids)
            predictions_post = pd.concat([
                postprocess_predictions(to_correct, layout, resources),
                predictions_raw.loc[demo_ids],
            ]).sort_index()
            changes = diff_predictions(predictions_raw, predictions_post)
            status_box.write(f"Post-processing changed {len(changes)} cell(s).")
            progress_bar.progress(1.0)

            # Manifest block: htrpipe's manifest_entry supplies thresholds
            # and resource counts; the two schema-derived keys are
            # overwritten with the schema that ACTUALLY ran.
            pp_block = manifest_entry(
                applied=True, layout_name=layout.name, resources=resources,
                changes=changes,
                city_lexicon_csv=CFG["lexicons_dir"],
                comment_llm_path=CFG["comment_llm_path"])
            actual_rules = {f.name: f.effective_postprocess for f in layout.fields}
            pp_block["field_rules"] = actual_rules
            pp_block["model_calibrated_rules"]["fields"] = sorted(
                n for n, r in actual_rules.items()
                if r in MODEL_CALIBRATED_RULES)

        status_box.update(label="Done", state="complete")

        S.run = {
            "run_name": run_name, "run_dir": run_dir,
            "model_kind": model_kind, "model_meta": model_meta,
            "page_preset": page_preset, "field_preset": field_preset,
            "prompt_block": prompt_block,
            "predictions_raw": predictions_raw,
            "predictions_post": predictions_post,
            "changes": changes, "pp_block": pp_block,
            "checkbox_results": checkbox_results,
            "confidences": confidences,
            "demos": demos, "demo_ids": demo_ids,
            "diagnostics": diagnostics, "runlog": runlog,
            "failures": failures, "seconds_total": seconds_total,
        }
        S.review_key = None
        st.success(f"Run `{run_name}` complete -- continue to step 6.")

    run = S.run
    if run is not None:
        if run["failures"]:
            st.error(f"{len(run['failures'])} page(s) failed and produced "
                     f"empty rows:")
            st.dataframe(pd.DataFrame(run["failures"])[["doc_id", "error"]])
        if run["runlog"] is not None:
            with st.expander("Per-page run log"):
                st.dataframe(run["runlog"])
        for name, df in (run.get("diagnostics") or {}).items():
            if df is not None:
                with st.expander(name.replace("_", " ")):
                    st.dataframe(df)

    nav_buttons()


# ===========================================================================
# STEP 6 -- REVIEW & EXPORT
# ===========================================================================

else:
    st.header("Step 6 -- Review, correct, export")
    run = S.run
    layout = S.layout
    if run is None or layout is None:
        st.warning("No extraction results -- the configuration changed since "
                   "the last run. Go back and run step 5 again.")
        nav_buttons()
        st.stop()

    has_post = run["predictions_post"] is not None
    stage = "post-processed" if has_post else "raw"
    if has_post:
        stage = st.radio("Which values to review",
                         ["post-processed", "raw"], horizontal=True)
    active = (run["predictions_post"] if stage == "post-processed"
              else run["predictions_raw"])

    # The edit frame accumulates the user's corrections. It is re-seeded from
    # the extraction whenever the run or the reviewed stage changes, so edits
    # never silently carry over between different result sets.
    review_key = (run["run_name"], stage)
    if S.review_key != review_key or S.edited_frame is None:
        S.edited_frame = active.copy()
        S.review_key = review_key

    # ---- advisory flags (values are never modified) ----------------------
    # Everything the flags need besides the values themselves, collected once
    # so the review and the export compute them identically.
    demo_ids = run.get("demo_ids") or []
    changed_cells = set()
    if stage == "post-processed" and run["changes"] is not None and len(run["changes"]):
        ch = run["changes"].reset_index()
        changed_cells = {(d, f) for d, f in zip(ch["doc_id"], ch["field"])}

    def flag_frame(frame, reviewed_cells=None):
        """(reasons, flags) for a frame: format checks, confidence, bleed."""
        bleed_cells = None
        if run.get("demos"):
            from htrpipe import fewshot
            query = frame.drop(index=[d for d in demo_ids if d in frame.index])
            bl = fewshot.bleeding_documents(query, run["demos"],
                                            _text_columns(layout))
            bleed_cells = bl.to_dict("records") if len(bl) else []
        return plausibility.check_frame(
            frame, layout, checkbox_results=run["checkbox_results"],
            confidences=run.get("confidences"),
            changed_by_postprocess=changed_cells,
            bleed_cells=bleed_cells, skip_docs=demo_ids,
            reviewed_cells=reviewed_cells)

    reasons, flags = flag_frame(active)
    st.subheader(f"Flags for review: {len(reasons)}")
    st.caption("Flags mark cells for human review. They change nothing and "
               "are not post-processing."
               + (f" Low confidence = conf_min below "
                  f"{plausibility.CONF_MIN_THRESHOLD} (a heuristic threshold, "
                  f"not calibrated on validation data)."
                  if run.get("confidences") is not None else ""))
    if demo_ids:
        st.info(f"Example pages (few-shot): doc_id "
                f"{', '.join(str(d) for d in demo_ids)}. Their values were "
                f"entered by hand in step 4 and were not read by the model.")
    if reasons:
        st.dataframe(plausibility.reasons_table(reasons),
                     use_container_width=True)

    # ---- inspect and correct, one form at a time -------------------------
    st.subheader("Inspect and correct")
    st.caption("Pick a form, compare the scan with its values, and correct "
               "any cell directly in the table. Every change is recorded in "
               "the run manifest as a manual edit.")
    image_by_doc = {f.doc_id: f.image for f in (S.forms or [])}
    doc_choice = st.selectbox("Form", list(S.edited_frame.index),
                              format_func=lambda d: f"doc_id {d}")
    img_col, val_col = st.columns([1, 1])
    with img_col:
        if doc_choice in image_by_doc:
            st.image(image_by_doc[doc_choice],
                     caption=f"doc_id {doc_choice}", use_container_width=True)
        else:
            st.info("The scan for this doc_id is no longer in memory "
                    "(reload the input in step 1 to view it).")
    with val_col:
        detail = pd.DataFrame({
            "field": list(S.edited_frame.columns),
            "value": [S.edited_frame.at[doc_choice, c]
                      for c in S.edited_frame.columns],
            "flag": [reasons.get((doc_choice, c), "")
                     for c in S.edited_frame.columns],
        })
        conf = run.get("confidences")
        if conf is not None:
            per_doc = conf.get(doc_choice)
            # Example pages have no score: their values were typed, not read.
            detail.insert(2, "conf_min", [
                (None if per_doc is None else per_doc.get(c))
                for c in S.edited_frame.columns])
        edited_detail = st.data_editor(
            detail, key=f"detail_editor_{doc_choice}",
            use_container_width=True, height=560, hide_index=True,
            num_rows="fixed", disabled=["field", "flag", "conf_min"],
            column_config={
                "value": st.column_config.TextColumn(
                    "value (editable)",
                    help="Correct the recognised value here."),
                "conf_min": st.column_config.NumberColumn(
                    "conf_min", format="%.2f",
                    help="Probability of the least certain token of this "
                         "value (raw model output)."),
            })
        for _, r in edited_detail.iterrows():
            S.edited_frame.at[doc_choice, r["field"]] = str(r["value"])

    # Overview: extraction as it came out of the pipeline, flagged cells
    # highlighted. Both background AND text colour are pinned so the cell
    # stays readable in the dark theme.
    def _highlight(_df):
        return flags.map(lambda f:
                         "background-color: #ffd6d6; color: #111111;"
                         if f else "")
    st.subheader("Extracted values (as recognised)")
    st.dataframe(active.style.apply(lambda _: _highlight(active), axis=None),
                 use_container_width=True)

    edited = S.edited_frame.astype(str)
    manual_edits = []
    for doc_id in active.index:
        for column in active.columns:
            before = str(active.at[doc_id, column])
            after = str(edited.at[doc_id, column])
            if before != after:
                manual_edits.append({"doc_id": doc_id, "field": column,
                                     "before": before, "after": after})
    if manual_edits:
        st.info(f"{len(manual_edits)} manual edit(s) across all forms will "
                f"be recorded in the manifest.")

    if has_post and run["changes"] is not None and len(run["changes"]):
        with st.expander("What post-processing changed (before -> after)"):
            st.dataframe(run["changes"].reset_index())

    # ---- export ----------------------------------------------------------
    st.subheader("Export")
    st.caption(f"The values under review (**{stage}**) are exported, with or "
               f"without the manual corrections applied on top, together "
               f"with their plausibility flags, the run log and the "
               f"manifest.")

    WITH_EDITS = "with manual edits"
    WITHOUT_EDITS = "without manual edits"
    choice = st.radio("Values to export", [WITH_EDITS, WITHOUT_EDITS],
                      horizontal=True)
    frame = edited if choice == WITH_EDITS else active

    if st.button("Write CSV + flags + manifest to the output folder",
                 type="primary"):
        run_dir = run["run_dir"]

        # Flags belong to the exported values, so they are computed on the
        # exported frame -- an edited value that now passes is not flagged.
        # Cells a person corrected no longer carry the model-output flags
        # (confidence, bleed); format checks run on the corrected value.
        reviewed = ({(e["doc_id"], e["field"]) for e in manual_edits}
                    if choice == WITH_EDITS else set())
        exp_reasons, _exp_flags = flag_frame(frame, reviewed_cells=reviewed)

        # The filename states both the base stage and whether edits are in,
        # so several exports from one run cannot overwrite each other.
        base = "post" if stage == "post-processed" else "raw"
        suffix = "_edited" if choice == WITH_EDITS else ""
        csv_name = f"predictions_{base}{suffix}.csv"

        # Written through htrpipe's text-safe writer so leading zeros
        # (postal codes!) survive.
        write_predictions_text(frame, run_dir / csv_name)
        plausibility.reasons_table(exp_reasons).to_csv(
            run_dir / "plausibility_flags.csv", index=False)
        if run["runlog"] is not None:
            run["runlog"].to_csv(run_dir / "runlog.csv")

        extra = {"exported_base": stage,
                 "exported_with_manual_edits": choice == WITH_EDITS,
                 "exported_files": [csv_name, "plausibility_flags.csv"]}
        if run.get("prompt_block"):
            extra["prompt"] = run["prompt_block"]

        # ---- confidence (Qwen) ----------------------------------------------
        conf = run.get("confidences")
        if conf is not None:
            conf_rows = [
                {"doc_id": d, "field": c,
                 "value_raw": run["predictions_raw"].at[d, c],
                 "conf_min": conf[d].get(c)}
                for d in conf for c in run["predictions_raw"].columns]
            conf_df = pd.DataFrame(conf_rows)
            conf_df.to_csv(run_dir / "confidences.csv", index=False,
                           float_format="%.6f")
            extra["exported_files"].append("confidences.csv")
            scores = conf_df["conf_min"]
            extra["confidence"] = {
                "score": "conf_min",
                "definition": "probability of the least certain token of the "
                              "field's value, from the raw logits",
                "threshold": plausibility.CONF_MIN_THRESHOLD,
                "threshold_origin": "heuristic set by the author; NOT "
                                    "calibrated on validation data",
                "nan_treated_as": 0.0,
                "applies_to": "raw model output (post-processed or manually "
                              "edited cells keep the raw value's score)",
                "n_scored_cells": int(scores.notna().sum()),
                "n_unscored_cells": int(scores.isna().sum()),
                "n_below_threshold_or_unscored": int(
                    (scores.fillna(0.0) < plausibility.CONF_MIN_THRESHOLD).sum()),
                "evaluated_condition": not bool(run.get("demos")),
            }

        # ---- few-shot ---------------------------------------------------------
        if run.get("demos"):
            pb = run.get("prompt_block") or {}
            runlog = run["runlog"]
            query = frame.drop(index=[d for d in demo_ids if d in frame.index])
            from htrpipe import fewshot as _fs
            bleed_exported = _fs.bleeding_documents(query, run["demos"],
                                                    _text_columns(layout))
            extra["condition"] = f"vlm_page_fewshot_k{len(run['demos'])}"
            extra["fewshot"] = {
                "evaluated_condition": False,
                "k": len(run["demos"]),
                "selection": "first k pages of the uploaded batch; values "
                             "entered by hand; excluded from inference",
                "demo_max_pixels": engine_qwen_page.DEMO_MAX_PIXELS,
                "demos": [
                    {"doc_id": d.doc_id, "file": d.path.name,
                     "source_size": list(d.source_size),
                     "prompt_size": list(d.prompt_size),
                     "approx_visual_tokens": d.visual_tokens,
                     "values": d.values}
                    for d in run["demos"]],
                "image_verification": pb.get("verify_report"),
                "n_echoed_examples": (int((runlog["n_objects"] > 1).sum())
                                      if "n_objects" in runlog else None),
                "bleed_raw": pb.get("bleed_raw"),
                "bleed_in_exported_values": int(len(bleed_exported)),
            }

        m = manifest_mod.build(
            run_name=run["run_name"],
            model_kind=run["model_kind"],
            model_meta=run["model_meta"],
            layout_name=layout.name,
            layout_source=S.layout_source or "unknown",
            schema_rows=S.schema_rows or [],
            page_preset=run["page_preset"],
            field_preset=(run["field_preset"] if run["model_kind"] == "trocr"
                          else "n/a"),
            config_matches_evaluated=(presets.config_matches_evaluated(
                run["model_kind"], run["page_preset"], run["field_preset"])
                and not run.get("demos")),
            n_forms=len(run["predictions_raw"]),
            n_failures=len(run["failures"]),
            seconds_total=round(run["seconds_total"], 2),
            postprocess_block=run["pp_block"],
            n_plausibility_flags=len(exp_reasons),
            manual_edits=manual_edits,
            stop_labels=(run["prompt_block"]["stop_labels"]
                         if run["model_kind"] == "chandra" else None),
            extra=extra,
        )
        manifest_path = manifest_mod.write(m, run_dir / "manifest.json")
        st.success(f"Written to `{run_dir}`: {csv_name}, flags, run log, "
                   f"manifest.")

        st.download_button(
            f"Download {csv_name}",
            (run_dir / csv_name).read_text(encoding="utf-8"),
            file_name=f"{run['run_name']}_{csv_name}", mime="text/csv")
        st.download_button(
            "Download plausibility_flags.csv",
            (run_dir / "plausibility_flags.csv").read_text(encoding="utf-8"),
            file_name=f"{run['run_name']}_flags.csv", mime="text/csv")
        if (run_dir / "confidences.csv").exists() and conf is not None:
            st.download_button(
                "Download confidences.csv",
                (run_dir / "confidences.csv").read_text(encoding="utf-8"),
                file_name=f"{run['run_name']}_confidences.csv", mime="text/csv")
        st.download_button("Download manifest",
                           manifest_path.read_text(encoding="utf-8"),
                           file_name=f"{run['run_name']}_manifest.json",
                           mime="application/json")

    nav_buttons()
