"""Model lifecycle: cached loading, explicit unloading, VRAM bookkeeping.

Why this module exists: Streamlit re-runs the entire script on every widget
interaction. Without ``st.cache_resource``, ticking a checkbox would reload an
8-billion-parameter model. And because all three recognisers share one 48 GB
card, switching models must *explicitly* drop the old one -- Python holding a
reference is enough to keep gigabytes allocated.

The actual loading code lives in the engine modules (taken from the
evaluation implementation); this module only wraps it in caching and provides the
one switching function the app calls.
"""

from __future__ import annotations

import streamlit as st

from htrpipe import HTRConfig, build_engine
from htrpipe.postprocess_setup import free_vram

from . import engine_chandra_page, engine_qwen_page

#: The three model kinds the app offers, in UI order.
MODEL_KINDS = ("trocr", "qwen", "chandra")


@st.cache_resource(show_spinner="Loading TrOCR ...")
def load_trocr(model_path: str):
    """Fine-tuned TrOCR via htrpipe's engine factory (evaluated settings).

    The full ``PipelineConfig`` is built later by the presets module; the
    engine itself only needs the HTR block, constructed here with the same
    values (greedy decoding, batch size 8).
    """
    cfg = HTRConfig(engine="trocr", model_path=model_path,
                    batch_size=8, num_beams=1, max_length=128, device="auto")
    return build_engine(cfg)


@st.cache_resource(show_spinner="Loading Qwen3-VL (8B) ... this takes a while")
def load_qwen(model_path: str):
    """Qwen model + processor, loaded exactly as in the evaluation."""
    return engine_qwen_page.load_model(model_path)


@st.cache_resource(show_spinner="Loading Chandra ... this takes a while")
def load_chandra(model_path: str):
    """ChandraOCR runner, loaded exactly as in the evaluation."""
    return engine_chandra_page.load_runner(model_path)


def unload_all(except_kind: str = "") -> float:
    """Drop every cached model except ``except_kind`` and empty the CUDA cache.

    Order matters: the cache entries must be cleared FIRST (dropping the last
    Python references), because ``free_vram`` can only release memory that no
    live object still owns. Returns GiB still allocated afterwards (or -1 if
    torch is unavailable), which the sidebar displays.
    """
    if except_kind != "trocr":
        load_trocr.clear()
    if except_kind != "qwen":
        load_qwen.clear()
    if except_kind != "chandra":
        # Chandra's runner holds the model inside an object; clearing the
        # cache drops it, after which empty_cache can reclaim the memory.
        load_chandra.clear()

    allocated = free_vram(verbose=False)
    return -1.0 if allocated is None else allocated


def vram_status() -> str:
    """One-line VRAM readout for the sidebar. Never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available"
        gib = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"{gib:.1f} / {total:.0f} GiB allocated"
    except Exception:
        return "VRAM status unavailable"
