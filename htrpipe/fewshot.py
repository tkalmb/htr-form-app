"""In-context (few-shot) demonstrations for the whole-page Qwen condition.

The zero-shot condition sends one page image and the field schema. This
module builds the *k-shot* variant of the same prompt: k complete
``(form image, correct JSON)`` pairs, followed by the form to transcribe.

In the application, the demonstrations are the first k pages of the uploaded
batch, whose values the user enters by hand. Those k pages are excluded from
inference -- their hand-entered values go straight into the output -- so no
page is ever both a demonstration and a prediction.

Scope, deliberately narrow:

* **Whole pages only.** Demonstrations are page-level because the thing being
  demonstrated is the page-to-schema mapping, which a field crop cannot show.
* **No model, no decoding.** This module builds ``messages`` and hands back the
  image list in matching order. Loading and generation live in the engine
  module, so the few-shot and zero-shot paths cannot drift apart in dtype,
  attention implementation or generation settings.

What this condition is
----------------------
This is **not** the zero-shot condition and must never be reported as one. It
is a separate declared condition, in the same sense as post-processing: same
model, same images, different information in the prompt.

Two properties keep it defensible, and both are enforced here rather than left
to discipline:

1. :func:`assert_disjoint` refuses to proceed when a demonstration is also a
   page being predicted.
2. The demonstration ``doc_id``s and their values go into the run manifest:
   "few-shot with k=2" is not reproducible without knowing *which* two.

Trimmed for the application: the ground-truth loaders, the seeded
demonstration sampler and the attention probes used during the thesis
experiments are not included, because the application takes its
demonstrations from the user instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from PIL import Image

#: Qwen-family processors emit roughly one visual token per 28x28 patch.
#: Used only for printing a budget estimate; nothing depends on it.
PATCH_AREA = 28 * 28


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def fit_to_pixel_budget(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale by area to at most ``max_pixels``, preserving aspect ratio.

    Demonstration images are resized *here*, before the processor sees them,
    rather than by giving the processor a second pixel budget. The processor's
    ``min_pixels``/``max_pixels`` apply to every image in a call, so there is no
    per-image budget available -- pre-resizing is the only way to spend fewer
    visual tokens on a demonstration than on the query page.

    Keep the result above the processor's ``MIN_PIXELS`` or it will be upscaled
    again on the way in, which wastes the tokens this was meant to save.
    """
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / (width * height)) ** 0.5
    # floor, not round: rounding both dimensions up can push the area back over
    # the budget (a 2481x3508 page at a 802,816 px budget lands on 803,010).
    # Flooring guarantees w' * h' <= w * h * scale**2 == max_pixels.
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def approx_visual_tokens(size) -> int:
    """Rough visual-token count for a (width, height). Printing only."""
    width, height = size
    return (width * height) // PATCH_AREA


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

@dataclass
class Demo:
    """One in-context example: a filled form page and its ground truth."""

    doc_id: int
    path: Path
    values: Dict[str, str]
    image: Image.Image = field(repr=False)
    source_size: tuple = ()
    prompt_size: tuple = ()

    @property
    def visual_tokens(self) -> int:
        return approx_visual_tokens(self.prompt_size)


def assert_disjoint(demos: Sequence[Demo], eval_doc_ids: Iterable[int]) -> None:
    """Refuse to proceed if a demonstration is also being predicted.

    This is the leakage guard. A demonstration in the prompt hands the model
    the exact answer for that page; predicting it would measure copying, not
    recognition. The application excludes demonstration pages from inference,
    so an overlap means a bug rather than a choice, and it raises rather than
    warns.
    """
    evaluated = {int(d) for d in eval_doc_ids}
    overlap = sorted({d.doc_id for d in demos} & evaluated)
    if overlap:
        raise ValueError(
            f"LEAKAGE: doc_ids {overlap} are used as in-context demonstrations "
            f"AND are among the pages being predicted. Demonstration pages "
            f"must be excluded from inference."
        )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def demo_json(values: Mapping[str, str], columns: Sequence[str]) -> str:
    """Serialise a demonstration in exactly the target output format.

    Same key order, same indentation, all values quoted strings. Format
    consistency is doing most of the work in a few-shot prompt: if the
    demonstrations look different from what the model is asked to produce, the
    demonstrations are teaching the wrong thing.
    """
    ordered = {c: str(values.get(c, "")) for c in columns}
    return json.dumps(ordered, ensure_ascii=False, indent=2)


QUERY_PREAMBLE_EN = (
    "Now the actual form. It belongs to a DIFFERENT person than the examples "
    "above. Do not carry any value over from the examples -- read only what is "
    "written on this image. If a field is illegible, return \"\"."
)


CLOSING_INSTRUCTION = (
    "Output the JSON object for this form only. No prose, no markdown fences, "
    "no repetition of the examples."
)


def build_messages(
    query_image: Image.Image,
    task_prompt: str,
    demos: Sequence[Demo],
    columns: Sequence[str],
    system_prompt: str,
    query_preamble: str = QUERY_PREAMBLE_EN,
    closing: str = CLOSING_INSTRUCTION,
):
    """Assemble ``(messages, images)`` for one query page.

    Returns the image list alongside the messages **on purpose**: the chat
    template inserts one image placeholder per ``{"type": "image"}`` entry, in
    order, and the processor fills them from its ``images=`` argument in the
    same order. Building both from one function is what guarantees they agree.
    Passing an independently-assembled image list is the way to silently feed
    the model a demonstration where it expects the query page.

    Layout of the prompt, and why:

    1. task + schema + rules  -- the instructions come before the examples, so
       the examples are read as illustrations of a stated task rather than as
       the task definition.
    2. k x (example image, correct JSON) -- interleaved, each image announced
       before it appears. A block of images followed by a block of answers
       makes the model infer the pairing.
    3. query preamble + query image + closing -- the page to transcribe is
       last, adjacent to the generation point.
    """
    content: List[dict] = [{"type": "text", "text": task_prompt}]
    images: List[Image.Image] = []

    if demos:
        content.append({
            "type": "text",
            "text": (
                f"Below are {len(demos)} fully worked examples of the same form"
                f"type, followed by the form to be transcribed."
            ),
        })
        for i, demo in enumerate(demos, 1):
            content.append({
                "type": "text",
                "text": f"--- Example {i} of {len(demos)}: form image ---",
            })
            content.append({"type": "image", "image": demo.image})
            images.append(demo.image)
            content.append({
                "type": "text",
                "text": (
                    f"--- Example {i} if {len(demos)}: correct output ---\n"
                    + demo_json(demo.values, columns)
                ),
            })

    content.append({"type": "text", "text": query_preamble})
    content.append({"type": "image", "image": query_image})
    images.append(query_image)
    content.append({"type": "text", "text": closing})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    return messages, images


def render_messages(
    messages: Sequence[dict],
    processor_max_pixels: Optional[int] = None,
    max_text_chars: Optional[int] = None,
) -> str:
    """Render an assembled ``messages`` list as inspectable text.

    Exists because the few-shot prompt is not a string. The task block is,
    but the demonstrations are PIL images interleaved with text and are
    assembled per page at call time -- so without this there is no way to read
    what the model is actually given before running a whole batch on it.

    ``processor_max_pixels`` should be the processor's ``max_pixels``. Images
    are handed to the processor at full resolution and it downscales them, so
    the raw size of the query page overstates its token cost; passing the cap
    shows the effective count instead. Demonstration images are pre-resized and
    are already under it.
    """
    lines: List[str] = []

    def text_block(role: str, text: str) -> None:
        shown = text
        if max_text_chars is not None and len(text) > max_text_chars:
            shown = text[:max_text_chars] + f"\n... [{len(text) - max_text_chars} more chars]"
        lines.append("-" * 78)
        lines.append(f"{role.upper()}  [text]")
        lines.append("-" * 78)
        lines.append(shown)
        lines.append("")

    image_no = 0
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content", [])
        if isinstance(content, str):
            text_block(role, content)
            continue
        for block in content:
            if block.get("type") == "image":
                image_no += 1
                width, height = block["image"].size
                tokens = approx_visual_tokens((width, height))
                note = ""
                if processor_max_pixels and width * height > processor_max_pixels:
                    note = (f"  -> processor caps this at "
                            f"~{processor_max_pixels // PATCH_AREA} tokens")
                lines.append("=" * 78)
                lines.append(
                    f"{role.upper()}  <<< IMAGE {image_no} >>>  {width}x{height}px  "
                    f"~{tokens} visual tokens{note}"
                )
                lines.append("=" * 78)
                lines.append("")
            else:
                text_block(role, block.get("text", ""))

    return "\n".join(lines)


def _find_image_placeholder(processor, chat: str) -> Optional[str]:
    """Best guess at the chat template's image placeholder token.

    Attribute names differ across processor versions, so this probes several
    and falls back to a list of known literals. Returns ``None`` if none is
    found -- that is not itself a failure, it just means the string-level count
    is unavailable and :func:`verify_prompt_images` has to rely on the tensor
    check, which is the stronger one anyway.
    """
    candidates: List[str] = []
    for holder in (processor, getattr(processor, "tokenizer", None)):
        for attr in ("image_token", "image_pad_token"):
            value = getattr(holder, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
    candidates += ["<|image_pad|>", "<|vision_start|>", "<image>", "<|image|>"]

    for token in candidates:
        if token in chat:
            return token
    return None


def verify_prompt_images(
    processor,
    messages,
    images,
    expect: Optional[int] = None,
    placeholder_token: Optional[str] = None,
    strict_placeholder: bool = True,
) -> dict:
    """Prove that every image in ``images`` actually reaches the model tensors.

    The bug this exists to catch is silent. ``apply_chat_template`` inserts one
    placeholder per image *entry in the messages*, and the processor expands
    those placeholders from the *separate* ``images=`` argument. If the two
    disagree -- a template that emits no placeholder, an image list built
    independently of the messages -- the extra images are dropped and the run
    quietly becomes zero-shot with some extra text in the prompt. No exception
    is raised and the output still looks reasonable.

    Two independent checks:

    * **String level.** Count placeholder tokens in the rendered chat string.
      Advisory, because the token's spelling is version-dependent.
    * **Tensor level.** ``image_grid_thw`` has one row per image the vision
      tower will actually encode. This is the check that matters: it is
      downstream of the template, the processor and the image processor, and it
      counts what the model receives rather than what was requested.

    Returns a report dict and raises ``AssertionError`` on a mismatch.
    """
    expect = len(images) if expect is None else expect

    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    placeholder = placeholder_token or _find_image_placeholder(processor, chat)

    if placeholder is None and expect > 0 and strict_placeholder:
        # A template that emitted no placeholder at all is the silent-zero-shot
        # bug itself, so "cannot find one" must not be treated as "cannot
        # check". The alternative explanation -- this template spells the token
        # differently from every candidate -- is resolved by looking at the
        # string and passing `placeholder_token=` explicitly.
        raise AssertionError(
            "no image placeholder found in the rendered chat string, but "
            f"{expect} image(s) were passed. Either the template emitted none "
            "(the run would silently degrade to zero-shot with extra text), or "
            "it uses a token this helper does not know. Inspect the string and "
            "pass placeholder_token= if the latter:\n"
            f"  {chat[:300]!r}"
        )

    n_placeholders = chat.count(placeholder) if placeholder else None

    inputs = processor(text=[chat], images=images, return_tensors="pt")

    grid = inputs.get("image_grid_thw", None)
    if grid is None:
        raise AssertionError(
            "the processor returned no 'image_grid_thw'. Either no image "
            "reached it, or this processor reports vision inputs under a "
            "different key -- inspect list(inputs.keys()) before trusting this "
            "run."
        )

    n_encoded = int(grid.shape[0])
    merge = getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2
    per_image_tokens = [
        int(row[0] * row[1] * row[2]) // (merge * merge) for row in grid
    ]

    report = {
        "n_images_passed": len(images),
        "n_expected": expect,
        "n_placeholders_in_chat": n_placeholders,
        "placeholder_token": placeholder,
        "n_images_encoded": n_encoded,
        "visual_tokens_per_image": per_image_tokens,
        "visual_tokens_total": sum(per_image_tokens),
        "input_ids_len": int(inputs["input_ids"].shape[1]),
        "input_keys": sorted(inputs.keys()),
    }

    if n_encoded != expect:
        raise AssertionError(
            f"IMAGES DROPPED: {expect} image(s) were passed but the processor "
            f"encoded {n_encoded}. The demonstrations are not reaching the "
            f"model.\n{report}"
        )
    if n_placeholders is not None and n_placeholders != expect:
        raise AssertionError(
            f"PLACEHOLDER MISMATCH: the chat string contains {n_placeholders} "
            f"{placeholder!r} token(s) for {expect} image(s). The template and "
            f"the image list disagree.\n{report}"
        )
    return report


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def demo_bleed_report(predictions, demos: Sequence[Demo], columns: Sequence[str]):
    """Per-field rate at which a prediction exactly equals a demonstration value.

    The diagnostic this condition needs and the zero-shot condition cannot have.
    A model that cannot read a field has a new option once demonstrations are in
    context: emit a value it saw in one. That is retrieval, not recognition, and
    it inflates any metric that rewards plausible output.

    Requires **no ground truth** -- it compares predictions against the prompt,
    not against labels, so it needs no ground truth at all.

    Read it as a flag, not as an error count. Some matches are legitimate: two
    forms can share a city, and ``studies`` is a four-option categorical where a
    match is expected roughly a quarter of the time by chance. The columns worth
    staring at are the high-entropy ones -- ``email``, ``phone``, ``street``,
    ``comment`` -- where a match is close to impossible by coincidence.
    """
    import pandas as pd

    rows = []
    for col in columns:
        demo_values = {d.values.get(col, "") for d in demos}
        demo_values.discard("")
        predicted = [str(v).strip() for v in predictions[col].tolist()]
        non_empty = [v for v in predicted if v]
        hits = [v for v in non_empty if v in demo_values]
        rows.append({
            "field": col,
            "n_predicted": len(non_empty),
            "n_matching_a_demo": len(hits),
            "rate": (len(hits) / len(non_empty)) if non_empty else float("nan"),
            "example": hits[0] if hits else "",
        })
    return pd.DataFrame(rows).set_index("field")


def bleeding_documents(predictions, demos: Sequence[Demo], columns: Sequence[str]):
    """The individual (doc_id, field, value) cells behind the bleed report.

    Use this to check a suspicious field against the actual page: an exact
    match on ``email`` is worth opening the image for.
    """
    import pandas as pd

    lookup = {}
    for col in columns:
        for demo in demos:
            value = demo.values.get(col, "")
            if value:
                lookup.setdefault((col, value), []).append(demo.doc_id)

    rows = []
    for doc_id, row in predictions.iterrows():
        for col in columns:
            value = str(row[col]).strip()
            if value and (col, value) in lookup:
                rows.append({
                    "doc_id": doc_id,
                    "field": col,
                    "value": value,
                    "from_demo": lookup[(col, value)],
                })
    return pd.DataFrame(rows)


