"""
formparse.py -- turn a transcribed form page into a field row.

Design (differs from the previous version):

    raw model output
        |
        |  ADAPTER   -- one per model family. Knows only that model's markup.
        v
    [Segment]        -- uniform: text, optional bbox, optional grid (row, col)
        |
        |  ASSIGNMENT -- ONE cascade, identical for every model and every layout
        v
    ({field: str}, report)

The previous version dispatched whole pages to one of three parsers
(geometric / sequential-tabular / sequential-inline). That dispatch correlated
with layout, so a cross-layout difference in CER was partly a difference in
parser. Here every page runs the same cascade; what varies is which evidence
the model happened to emit.

This does NOT make the confound disappear -- a model that emits a table on
layout B and bboxes on layout A still gets different evidence there. What it
does is make the confound *measurable*: report["strategy"] records, per field
per page, which rule produced the value, so the distribution can be reported
per layout instead of being buried in a page-level branch.

Contract, unchanged: every value is a `str`; a field the model failed to
produce becomes "" rather than a dropped row. Leading zeros survive because
nothing here ever converts to int.
"""

from __future__ import annotations

PARSER_VERSION = "formparse-2.0"

import html as _htmllib
import re
from dataclasses import dataclass, field as _dcfield

# ==========================================================================
# 1. Segment: the one intermediate representation
# ==========================================================================


@dataclass
class Segment:
    """One transcribed line, with whatever positional evidence came with it.

    text   the transcription, markup already removed
    bbox   (x0, y0, x1, y1) if the model grounded it, else None
    table  index of the table it came from, else None
    row    grid row inside that table (colspans already expanded), else None
    col    grid column inside that table, else None
    order  position in document reading order
    """
    text: str
    bbox: tuple[int, int, int, int] | None = None
    table: int | None = None
    row: int | None = None
    col: int | None = None
    order: int = 0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2 if self.bbox else 0.0

    @property
    def x0(self) -> int:
        return self.bbox[0] if self.bbox else 0

    @property
    def x1(self) -> int:
        return self.bbox[2] if self.bbox else 0

    def same_row_as(self, other: "Segment", frac: float = 0.3) -> bool:
        """Do two grounded segments share a text row?

        Measured as vertical OVERLAP, not centre containment. A wrapped comment
        block is several times taller than its one-line label, so its centre
        falls below the label box and a centre test rejects the true pair --
        observed on layout A, where the comment came back empty for exactly
        this reason.
        """
        if not self.bbox or not other.bbox:
            return False
        lo = max(self.bbox[1], other.bbox[1])
        hi = min(self.bbox[3], other.bbox[3])
        if hi <= lo:
            return False
        shorter = min(self.bbox[3] - self.bbox[1], other.bbox[3] - other.bbox[1])
        return shorter <= 0 or (hi - lo) / shorter >= frac


# ==========================================================================
# 2. Shared text cleaning
# ==========================================================================

_STOP_STR = "<\uff5cend\u2581of\u2581sentence\uff5c>"

# LaTeX / math wrappers. DeepSeek returns some numbers as "\[ 68496 \]".
# That is the model's output format, not something it read off the page.
_MATH_SPAN_RE = re.compile(
    r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|\$([^$\n]*?)\$"
    r"|<math[^>]*>(.*?)</math>",
    re.DOTALL | re.IGNORECASE)
_STRAY_MATH_RE = re.compile(r"\\[\[\]()]|</?math[^>]*>", re.IGNORECASE)


def strip_math_markup(text: str) -> str:
    if not text:
        return ""
    def keep(m):
        inner = next((g for g in m.groups() if g is not None), "")
        return f" {inner.strip()} "
    prev = None
    while prev != text:
        prev = text
        text = _MATH_SPAN_RE.sub(keep, text)
    return _STRAY_MATH_RE.sub(" ", text)


_EMPH_RE = re.compile(r"\*{1,3}|_{2,}")     # markdown bold/italic, incl. *value*
_HEADING_RE = re.compile(r"^#+\s*")


def clean_text(text: str) -> str:
    """Collapse whitespace and drop markup that is never part of a value."""
    if not text:
        return ""
    text = text.replace(_STOP_STR, " ")
    text = strip_math_markup(text)
    text = _HEADING_RE.sub("", text.strip())
    text = _EMPH_RE.sub("", text)
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


# ==========================================================================
# 3. HTML table -> grid, with colspan/rowspan expanded
# ==========================================================================
# Expanding spans matters for checkbox alignment. Chandra layout B often emits
#     header:  IM | WIN | BWL | Kunst(colspan=3)
#     markers:  . |  .  |  .  | [x](colspan=3)
# Index-aligning the two rows as written misreads the tick; after expansion the
# ticked cell and "Kunst" share a column and the answer is recoverable.

_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_SPAN_ATTR_RE = re.compile(r'\b(col|row)span\s*=\s*["\']?(\d+)', re.IGNORECASE)
_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


def _span_of(attrs: str, which: str) -> int:
    for kind, n in _SPAN_ATTR_RE.findall(attrs or ""):
        if kind.lower() == which:
            try:
                return max(1, min(int(n), 12))
            except ValueError:
                return 1
    return 1


def table_to_cells(html: str) -> list[tuple[int, int, str]]:
    """<table> -> [(row, col, text)] on an expanded grid.

    A cell spanning N columns is emitted once, at its leftmost column; the
    columns it covers are simply reserved so later cells land in the right
    place. Emitting it N times would create phantom duplicate values.
    """
    occupied: set[tuple[int, int]] = set()
    out: list[tuple[int, int, str]] = []
    for r, tr in enumerate(_TR_RE.findall(html)):
        c = 0
        for attrs, inner in _TD_RE.findall(tr):
            while (r, c) in occupied:
                c += 1
            cs, rs = _span_of(attrs, "col"), _span_of(attrs, "row")
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, c + dc))
            out.append((r, c, inner))
            c += cs
    return out


# ==========================================================================
# 4. HTML inline markup -> text, with checkbox state preserved
# ==========================================================================

_INPUT_RE = re.compile(r"<input\b[^>]*/?>", re.IGNORECASE)
_CHECKED_RE = re.compile(r"\bchecked\b", re.IGNORECASE)
_INPUT_TYPE_RE = re.compile(r"""\btype\s*=\s*["']?([A-Za-z]+)""", re.IGNORECASE)
_INPUT_VALUE_RE = re.compile(
    r"""\bvalue\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_TICKABLE = {"checkbox", "radio"}
_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_DEL_RE = re.compile(r"<del\b[^>]*>(.*?)</del>", re.DOTALL | re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _render_input(match: re.Match) -> str:
    """<input> -> a marker, or the text it carries.

    Chandra uses <input type="text" value="..."> to carry a handwritten value
    (layout A doc 83 does this for every field on the page). Treating those as
    checkboxes destroys the whole page, so a typed non-tickable input emits its
    value attribute instead.
    """
    tag = match.group(0)
    tm = _INPUT_TYPE_RE.search(tag)
    kind = tm.group(1).lower() if tm else None
    vm = _INPUT_VALUE_RE.search(tag)
    value = next((g for g in vm.groups() if g is not None), "") if vm else ""

    if kind in _TICKABLE:
        return " [x] " if _CHECKED_RE.search(tag) else " [ ] "
    if kind is not None:
        return f" {value} " if value else " "
    if _CHECKED_RE.search(tag):
        return " [x] "
    return f" {value} " if value else " [ ] "


def _drop_del(match: re.Match) -> str:
    """<del>...</del> is struck-through text: the model says it was corrected.

    Dropped only if something else survives in the same fragment. On layout B
    doc 71 the ENTIRE Name value is wrapped in <del>; deleting it there would
    turn a recognised value into an empty field, which scores as a recognition
    failure that did not happen.
    """
    return "\u0001DEL\u0001" + match.group(1) + "\u0001/DEL\u0001"


def html_fragment_to_lines(frag: str) -> list[str]:
    """An HTML fragment -> text lines. <br> and block ends split lines."""
    if not frag:
        return []
    frag = _DEL_RE.sub(_drop_del, frag)
    frag = _IMG_RE.sub(" ", frag)          # signature images carry alt text: drop
    frag = _INPUT_RE.sub(_render_input, frag)
    frag = _BR_RE.sub("\n", frag)
    frag = re.sub(r"</(?:p|div|h[1-6]|li)\s*>", "\n", frag, flags=re.IGNORECASE)
    frag = _TAG_RE.sub(" ", frag)
    frag = _htmllib.unescape(frag)

    out = []
    for raw in frag.splitlines():
        line = clean_text(raw)
        if line and clean_text(_plain(line)):
            out.append(line)
    return out


_DEL_MARK_RE = re.compile("\u0001DEL\u0001(.*?)\u0001/DEL\u0001", re.DOTALL)


def _plain(text: str) -> str:
    """Drop the del markers, keeping their content. For matching, not output."""
    return _DEL_MARK_RE.sub(r"\1", text or "")


def _resolve_del(text: str) -> str:
    """Struck-through text is dropped -- unless it is all the value has.

    "Ort <del>99050</del> Dinkelsb\u00fchl" is a correction: the model says 99050
    was crossed out, so the value is Dinkelsb\u00fchl. But on layout B doc 71 the
    ENTIRE Name value is wrapped in <del>; dropping it there turns a value the
    model did read into an empty field, which would be scored as a recognition
    failure that never happened.
    """
    if "\u0001DEL\u0001" not in (text or ""):
        return text
    without = _DEL_MARK_RE.sub(" ", text)
    if without.strip(" \t:.-|*"):
        return without
    return _plain(text)


# ==========================================================================
# 5. Adapters
# ==========================================================================

_GROUNDED_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>(.*?)(?=<\|ref\|>|\Z)",
    re.DOTALL)
_BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3})\s*\]")
_DIV_RE = re.compile(
    r'<div\b([^>]*)>(.*?)</div>', re.DOTALL | re.IGNORECASE)
_BBOX_ATTR_RE = re.compile(
    r'data-bbox\s*=\s*["\']([\d\s.\-]+)["\']', re.IGNORECASE)
_LABEL_ATTR_RE = re.compile(r'data-label\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)


def looks_grounded(text: str) -> bool:
    return bool(_GROUNDED_RE.search(text or ""))


def looks_like_chandra(text: str) -> bool:
    return bool(re.search(r"<div\b[^>]*data-bbox", text or "", re.IGNORECASE))


def _boxes(det: str) -> list[tuple[int, int, int, int]]:
    out = []
    for m in _BOX_RE.finditer(det or ""):
        try:
            out.append(tuple(int(float(v)) for v in m.group(1).split(",")))
        except ValueError:
            continue
    return out


def _union(boxes):
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _emit_lines(lines, bbox, boxes, segs, table_counter):
    """Turn a block's lines into Segments, giving each its own box when possible.

    A DeepSeek det may hold several boxes for one block -- on layout C doc 48
    four address lines arrived under one <|ref|>. When the counts match, each
    line gets its own box, which is what makes geometric matching work there.
    """
    per_line = boxes if len(boxes) == len(lines) and len(boxes) > 1 else None
    for i, line in enumerate(lines):
        segs.append(Segment(text=line,
                            bbox=per_line[i] if per_line else bbox,
                            order=len(segs)))


def adapt_deepseek(raw: str) -> list[Segment]:
    """DeepSeek-OCR-2 grounded output -> Segments."""
    segs: list[Segment] = []
    tables = 0
    for m in _GROUNDED_RE.finditer(raw or ""):
        boxes = _boxes(m.group(2))
        bbox = _union(boxes)
        content = m.group(3)
        if _TABLE_RE.search(content):
            for tm in _TABLE_RE.finditer(content):
                for r, c, cell in table_to_cells(tm.group(0)):
                    for line in html_fragment_to_lines(cell) or [""]:
                        segs.append(Segment(text=line, bbox=bbox, table=tables,
                                            row=r, col=c, order=len(segs)))
                tables += 1
            continue
        lines = [clean_text(ln) for ln in content.splitlines()]
        lines = [ln for ln in lines if ln]
        _emit_lines(lines, bbox, boxes, segs, tables)
    return segs


def adapt_chandra(raw: str) -> list[Segment]:
    """Chandra HTML output -> Segments."""
    segs: list[Segment] = []
    tables = 0
    for m in _DIV_RE.finditer(raw or ""):
        attrs, inner = m.group(1), m.group(2)
        bm = _BBOX_ATTR_RE.search(attrs)
        bbox = None
        if bm:
            parts = bm.group(1).split()
            if len(parts) == 4:
                try:
                    bbox = tuple(int(float(p)) for p in parts)
                except ValueError:
                    bbox = None
        kind = (_LABEL_ATTR_RE.search(attrs).group(1)
                if _LABEL_ATTR_RE.search(attrs) else "")
        if kind.lower() == "page-header":
            continue                          # holds the page id ("B-83")

        # Text that sits outside the table (e.g. "Studiengang:" before it)
        outside = _TABLE_RE.sub(" ", inner)
        for line in html_fragment_to_lines(outside):
            segs.append(Segment(text=line, bbox=bbox, order=len(segs)))

        for tm in _TABLE_RE.finditer(inner):
            for r, c, cell in table_to_cells(tm.group(0)):
                for line in html_fragment_to_lines(cell) or [""]:
                    segs.append(Segment(text=line, bbox=bbox, table=tables,
                                        row=r, col=c, order=len(segs)))
            tables += 1
    return segs


def adapt_plain(raw: str) -> list[Segment]:
    """Fallback: markdown or plain text, no positional evidence."""
    segs = []
    text = raw or ""
    if _TABLE_RE.search(text):
        tables = 0
        rest = _TABLE_RE.sub("\n", text)
        for tm in _TABLE_RE.finditer(text):
            for r, c, cell in table_to_cells(tm.group(0)):
                for line in html_fragment_to_lines(cell) or [""]:
                    segs.append(Segment(text=line, table=tables, row=r, col=c,
                                        order=len(segs)))
            tables += 1
        text = rest
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"\|?[\s:|-]{4,}\|?", line):
            continue
        if "|" in line:                        # markdown table row -> cells
            cells = [c.strip() for c in line.strip("|").split("|")]
            for c, cell in enumerate(cells):
                seg_text = clean_text(cell)
                segs.append(Segment(text=seg_text, table=0,
                                    row=len(segs), col=c, order=len(segs)))
            continue
        line = clean_text(line)
        if line:
            segs.append(Segment(text=line, order=len(segs)))
    return segs


def to_segments(raw: str) -> tuple[list[Segment], str]:
    """(segments, source) -- source names the adapter that ran."""
    if looks_grounded(raw):
        return adapt_deepseek(raw), "deepseek"
    if looks_like_chandra(raw):
        return adapt_chandra(raw), "chandra"
    return adapt_plain(raw), "plain"


# ==========================================================================
# 6. Label matching
# ==========================================================================

_UMLAUT = "\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df"


def _first_label(label) -> str:
    return label[0] if isinstance(label, (list, tuple)) else label


def _variants(label) -> list[str]:
    raw = list(label) if isinstance(label, (list, tuple)) else [label]
    out = []
    for v in raw:
        out.append(v)
        a = (v.replace("\u00df", "ss").replace("\u00e4", "ae")
              .replace("\u00f6", "oe").replace("\u00fc", "ue"))
        if a != v:
            out.append(a)
    return out


def _label_pattern(label) -> re.Pattern:
    """Strict: word-bounded so 'Name' does not match inside 'Vorname'."""
    alts = [r"\s*".join(re.escape(p) for p in v.split())
            for v in sorted(_variants(label), key=len, reverse=True)]
    body = "|".join(alts)
    return re.compile(rf"(?<![A-Za-z{_UMLAUT}])(?:{body})(?![A-Za-z{_UMLAUT}])",
                      re.IGNORECASE)


def _prefix_pattern(label) -> re.Pattern:
    """Relaxed: the label at the START of a line, trailing boundary not required.

    Layout C doc 71 emits "Stra\u00dfeavenue de Loiseau" -- the space between label
    and value was lost. The strict pattern refuses that match and the field is
    lost, so at line start the boundary requirement is dropped. Anchoring at the
    start keeps this from re-introducing the Name-inside-Vorname problem.
    """
    alts = [r"\s*".join(re.escape(p) for p in v.split())
            for v in sorted(_variants(label), key=len, reverse=True)]
    return re.compile(rf"^\s*(?:{'|'.join(alts)})\s*[:.\-]?\s*", re.IGNORECASE)


@dataclass
class LabelSpec:
    name: str
    label: str | list
    strict: re.Pattern = _dcfield(init=False)
    prefix: re.Pattern = _dcfield(init=False)

    def __post_init__(self):
        self.strict = _label_pattern(self.label)
        self.prefix = _prefix_pattern(self.label)


def build_specs(labels: dict, stop_labels=None) -> list[LabelSpec]:
    specs = [LabelSpec(n, l) for n, l in labels.items()]
    for i, extra in enumerate(stop_labels or []):
        specs.append(LabelSpec(f"__stop_{i}", extra))
    return specs


# ==========================================================================
# 7. Value cleaning
# ==========================================================================

_MARKERS = "xX\u2715\u2716\u2717\u2718\u2713\u2714\u2612\u00d7\u2a2f"
_MARKER_CLASS = f"[{_MARKERS}]"
_MARKER_RE = re.compile(
    rf"\[\s*{_MARKER_CLASS}\s*\]|(?<![A-Za-z0-9]){_MARKER_CLASS}(?![A-Za-z0-9])")
_EMPTY_BOX_RE = re.compile(r"[\u2610\u25a1\u2751]|\[\s*\]")
_ANY_MARKER_RE = re.compile(
    rf"\[\s*{_MARKER_CLASS}\s*\]|\[\s*\]|[\u2610\u2611\u2612\u25a1\u2751]")
_ONLY_MARKERS_RE = re.compile(
    rf"^(?:\s|\[\s*{_MARKER_CLASS}?\s*\]|[\u2610\u2611\u2612\u25a1\u2751])+$")
_LEAD_JUNK_RE = re.compile(r"^[\s:.\-\u2013\u2014|_*]+")
_TRAIL_JUNK_RE = re.compile(r"[\s:|_*]+$")


def clean_value(text: str) -> str:
    if not text:
        return ""
    text = _resolve_del(text)
    v = _TRAIL_JUNK_RE.sub("", _LEAD_JUNK_RE.sub("", text.strip()))
    if _ONLY_MARKERS_RE.match(v):
        return ""                    # a stray checkbox is not a transcription
    return v.strip()


# ==========================================================================
# 8. Checkbox resolution
# ==========================================================================


def _is_marker_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    if _EMPTY_BOX_RE.fullmatch(t):
        return False
    return bool(_MARKER_RE.fullmatch(t)) or bool(
        _MARKER_RE.search(t) and _ANY_MARKER_RE.fullmatch(t))


def _option_index(token: str, lower: list[str]):
    t = clean_value(token).lower()
    return lower.index(t) if t in lower else None


def resolve_checkbox(segments: list[Segment], options) -> tuple[str, str]:
    """(value, status). Returns "" unless exactly one option is unambiguously set.

    A wrong option and an empty field score identically, so no guess is made
    when the evidence is ambiguous -- the status records why instead.
    """
    lower = [o.lower() for o in options]
    segments = [Segment(_plain(s.text), s.bbox, s.table, s.row, s.col, s.order)
                for s in segments]

    # -- grid: an options row and a marker row in the same expanded table
    by_cell = {(s.table, s.row, s.col): s for s in segments if s.row is not None}
    opt_rows: dict[tuple[int, int], dict[int, int]] = {}
    for (t, r, c), s in by_cell.items():
        oi = _option_index(s.text, lower)
        if oi is not None:
            opt_rows.setdefault((t, r), {})[c] = oi
    for (t, r), cols in sorted(opt_rows.items()):
        if len(cols) < 2:
            continue
        for dr in (1, 2):
            hits = []
            any_cell = False
            for c, oi in cols.items():
                cell = by_cell.get((t, r + dr, c))
                if cell is None:
                    continue
                any_cell = True
                if _is_marker_token(cell.text):
                    hits.append(oi)
            if not any_cell:
                continue
            if len(hits) == 1:
                return options[hits[0]], "grid"
            if len(hits) > 1:
                return "", "ambiguous_grid"
            break

    # -- geometric: option segments and marker segments with boxes, paired by x
    opt_segs, mark_segs = [], []
    for s in segments:
        if s.bbox is None:
            continue
        oi = _option_index(re.sub(_ANY_MARKER_RE, " ", s.text), lower)
        if oi is not None:
            opt_segs.append((s, oi))
        if _ANY_MARKER_RE.fullmatch(s.text.strip()):
            mark_segs.append(s)
    if opt_segs and mark_segs and len(mark_segs) == len(opt_segs):
        pairs = []
        for s in mark_segs:
            near = min(opt_segs, key=lambda o: (abs(o[0].x0 - s.x0)
                                                + abs(o[0].cy - s.cy)))
            pairs.append((near[1], _is_marker_token(s.text)))
        ticked = [oi for oi, on in pairs if on]
        if len(set(ticked)) == 1:
            return options[ticked[0]], "geometric"
        if len(ticked) > 1:
            return "", "ambiguous_geometric"

    # -- sequence: markers and options in reading order, equal counts, pair by
    # index. Handles "IM [ ] WIN [x]" and "[ ] IM [x] WIN" identically.
    span = "\n".join(s.text for s in segments)
    markers = [(m.start(), not _EMPTY_BOX_RE.fullmatch(m.group(0).strip()))
               for m in _ANY_MARKER_RE.finditer(span)]
    seen, opt_pos = set(), []
    for opt in options:
        m = re.search(rf"(?<![A-Za-z]){re.escape(opt)}(?![A-Za-z])", span,
                      re.IGNORECASE)
        if m and opt not in seen:
            seen.add(opt)
            opt_pos.append((m.start(), opt))
    opt_pos.sort()
    if markers and opt_pos and len(markers) == len(opt_pos):
        ticked = [i for i, (_, on) in enumerate(markers) if on]
        if len(ticked) == 1:
            return opt_pos[ticked[0]][1], "paired"
        if len(ticked) > 1:
            return "", "ambiguous_paired"
        return "", "none_ticked"

    # NOTE: there is deliberately no "the marker after the option belongs to it"
    # rule. Chandra emits BOTH conventions -- "IM [ ] WIN [x]" on some pages and
    # "[ ] IM [x] WIN" on others -- so a same-line positional rule reads the
    # wrong option on half of them. Where the counts match, the index pairing
    # above is convention-independent and handles both. Where they do not match,
    # the output genuinely does not say which option the tick belongs to, and a
    # wrong option scores exactly like an empty one, so nothing is returned.

    if markers:
        # Markers exist but no rule aligned them: e.g. five boxes for four
        # options (layout A doc 83, layout C doc 30). Whether the spare box
        # leads or trails decides the answer and the output does not say.
        return "", "marker_count_mismatch"
    return "", "unresolved"


# ==========================================================================
# 9. The assignment cascade
# ==========================================================================
# Four rules, tried in a fixed order for every field on every page. The order
# is by strength of evidence, not by layout:
#
#   1 inline      label and value in the SAME segment. The model itself paired
#                 them, so nothing is inferred.
#   2 grid        label cell and value cell in the same expanded table row
#                 (or the cell directly below, when the table is stacked).
#   3 geometric   label segment and value segment on the same visual row, value
#                 to the right. Needs boxes.
#   4 sequential  the next non-label segment in reading order. Weakest: it
#                 assumes the model emitted label and value adjacently.
#
# Which rules are AVAILABLE depends on what the model emitted, and that does
# still vary by layout. report["strategy"] makes that variation countable.

STRATEGIES = ("inline", "grid", "geometric", "sequential")


def _label_hits(segments, specs) -> dict[str, list[tuple[int, re.Match]]]:
    """{spec name: [(segment index, match)]} for every label occurrence.

    A label is an anchor when it opens the segment. A label found mid-segment
    counts only if the segment already opens with some label, i.e. it is a
    label-bearing line that may hold several pairs ("Name: X  Vorname: Y").
    Without that guard, the value "Reimor - Dobes - Straße" registers as a
    Straße anchor, is excluded from its own field's candidate set, and the
    street is reported empty although the model read it correctly.
    """
    begins = [any(sp.prefix.match(_plain(s.text)) for sp in specs)
              for s in segments]
    hits: dict[str, list] = {}
    for spec in specs:
        for i, s in enumerate(segments):
            m = spec.prefix.match(s.text) or spec.prefix.match(_plain(s.text))
            if m is None and begins[i]:
                m = spec.strict.search(s.text) or spec.strict.search(_plain(s.text))
            if m:
                hits.setdefault(spec.name, []).append((i, m))
    return hits


def _is_label_only(seg: Segment, specs) -> bool:
    """Does this segment contain a label and nothing else?"""
    for spec in specs:
        m = spec.prefix.match(seg.text)
        if m and not clean_value(seg.text[m.end():]):
            return True
    return False


def _inline_value(seg: Segment, spec, specs) -> str | None:
    """Value from the label's own segment: the text after it, up to the next label."""
    m = spec.prefix.match(seg.text) or spec.strict.search(seg.text)
    if not m:
        return None
    rest = seg.text[m.end():]
    # another label further along the same segment ends this value
    cut = len(rest)
    for other in specs:
        if other.name == spec.name:
            continue
        om = other.strict.search(rest)
        if om:
            cut = min(cut, om.start())
    return clean_value(rest[:cut])


def _grid_value(segments, idx, specs):
    """(value, source index) from the same table row, else the cell below."""
    lab = segments[idx]
    if lab.row is None:
        return None, None
    same = [(i, s) for i, s in enumerate(segments)
            if s.table == lab.table and s.row == lab.row and s.col > lab.col]
    for i, s in sorted(same, key=lambda t: t[1].col):
        if _is_label_only(s, specs):
            break                       # the next label bounds the search
        v = clean_value(s.text)
        if v:
            return v, i
    # stacked table: label row, then value row (layout B docs 15, 27)
    below = [(i, s) for i, s in enumerate(segments)
             if s.table == lab.table and s.row == lab.row + 1 and s.col == lab.col]
    for i, s in below:
        if _is_label_only(s, specs):
            continue
        v = clean_value(s.text)
        if v:
            return v, i
    return None, None


def _geometric_value(segments, idx, specs, label_idx: set):
    """(value, source index) from the segment right of the label, same row."""
    lab = segments[idx]
    if lab.bbox is None or lab.row is not None:
        return None, None
    right_bound = min(
        (segments[j].x0 for j in label_idx
         if j != idx and segments[j].bbox and segments[j].x0 > lab.x1
         and lab.same_row_as(segments[j])),
        default=None)
    cands = [(i, s) for i, s in enumerate(segments)
             if i not in label_idx and s.bbox
             and s.x0 >= lab.x1 - 5 and lab.same_row_as(s)
             and (right_bound is None or s.x0 < right_bound)]
    cands.sort(key=lambda t: t[1].x0)
    for i, s in cands:
        v = clean_value(s.text)
        if v:
            return v, i
    return None, None


def _sequential_value(segments, idx, label_idx: set):
    """(value, source index) from the next non-label segment in reading order."""
    for j in range(idx + 1, len(segments)):
        if j in label_idx:
            return None, None
        v = clean_value(segments[j].text)
        if v:
            return v, j
    return None, None


def _long_text_extra(segments, idx, label_idx, first: str, start_after: int) -> str:
    """Continuation lines for a wrapped comment.

    The value runs until the next label of any kind. A comment that wrapped
    over three lines arrives as three segments and has to be rejoined, or two
    thirds of the field is scored as a deletion.
    """
    parts = [first] if first else []
    lab = segments[idx]
    for j in range(max(idx, start_after) + 1, len(segments)):
        if j in label_idx:
            break
        s = segments[j]
        if lab.row is not None and s.row is not None and s.row != lab.row:
            break
        if lab.bbox and s.bbox and s.x0 < lab.x0 - 5:
            break                     # back in the label column: new field
        v = clean_value(s.text)
        if v:
            parts.append(v)
    return " ".join(parts).strip()


def parse_page(raw: str, target_fields, labels, checkbox_fields,
               stop_labels=None, line_join: str = " "):
    """Transcribed page -> ({field: str}, report). One path for every model."""
    segments, source = to_segments(raw)
    specs = build_specs(labels, stop_labels)
    spec_by_name = {s.name: s for s in specs}
    columns = [f.name for f in target_fields]
    field_by_name = {f.name: f for f in target_fields}
    row = {c: "" for c in columns}

    hits = _label_hits(segments, specs)
    # every segment that is a label anchor, for bounding the searches
    label_idx = {i for name, hl in hits.items() for i, _ in hl}

    strategy = {c: "none" for c in columns}
    checkbox_status: dict[str, str] = {}
    duplicates = []

    for name in columns:
        if name not in hits:
            continue
        if len(hits[name]) > 1:
            duplicates.append(name)
        idx, _ = hits[name][0]
        spec = spec_by_name[name]
        f = field_by_name[name]
        is_long = getattr(f, "type", "") == "long_text"

        if name in checkbox_fields:
            group = _checkbox_segments(segments, idx, label_idx)
            value, status = resolve_checkbox(group, checkbox_fields[name])
            checkbox_status[name] = status
            strategy[name] = status if value else "none"
            row[name] = value
            continue

        value, how, src = None, "none", idx
        inline = _inline_value(segments[idx], spec, specs)
        if inline:
            value, how, src = inline, "inline", idx
        if value is None:
            g, i = _grid_value(segments, idx, specs)
            if g:
                value, how, src = g, "grid", i
        if value is None:
            g, i = _geometric_value(segments, idx, specs, label_idx)
            if g:
                value, how, src = g, "geometric", i
        if value is None:
            g, i = _sequential_value(segments, idx, label_idx)
            if g:
                value, how, src = g, "sequential", i

        if is_long:
            merged = _long_text_extra(segments, idx, label_idx, value or "", src)
            if merged:
                value = merged
                how = how if how != "none" else "sequential"

        row[name] = value or ""
        strategy[name] = how if value else "none"

    anchored = {c: c in hits for c in columns}
    n_by_strategy = {s: sum(1 for v in strategy.values() if v == s)
                     for s in STRATEGIES}
    # `mode` keeps its old meaning -- which assignment rule ran -- but the rule
    # is now chosen per field, so the page-level value is the dominant one.
    # `source` names the adapter. Log both.
    dominant = max(STRATEGIES, key=lambda s: n_by_strategy[s])
    if not n_by_strategy[dominant]:
        dominant = "none"
    return row, {
        "anchored": anchored,
        "checkbox": checkbox_status,
        "duplicates": duplicates,
        "stops_used": sorted(n for n in hits if n.startswith("__stop_")),
        "n_anchored": sum(anchored.values()),
        "n_fields": len(columns),
        "n_segments": len(segments),
        "n_filled": sum(1 for v in row.values() if v),
        "source": source,
        "strategy": strategy,
        "n_by_strategy": n_by_strategy,
        "mode": dominant,
        "parser_version": PARSER_VERSION,
    }


def _checkbox_segments(segments, idx, label_idx) -> list[Segment]:
    """The segments belonging to the checkbox group.

    From the Studiengang anchor up to the next label anchor, plus -- when the
    group lives in a table -- the two rows below it, because the marker row
    carries no label and would otherwise be cut off.
    """
    out = [segments[idx]]
    lab = segments[idx]
    for j in range(idx + 1, len(segments)):
        s = segments[j]
        if j in label_idx:
            if lab.row is not None and s.row is not None \
                    and s.row - lab.row <= 2:
                continue                 # marker row sits under the label row
            break
        out.append(s)
    if lab.row is not None:
        out += [s for s in segments
                if s.table == lab.table and s.row is not None
                and lab.row < s.row <= lab.row + 2 and s not in out]
    return out


# Backwards-compatible alias: existing notebooks call parse_labeled_text.
def parse_labeled_text(text, target_fields, labels, checkbox_fields,
                       line_join: str = " ", stop_labels=None):
    return parse_page(text, target_fields, labels, checkbox_fields,
                      stop_labels=stop_labels, line_join=line_join)


# ==========================================================================
# 10. JSON parser (Qwen schema prompt) -- carried over unchanged
# ==========================================================================

import json  # noqa: E402

_BARE_VALUE_RE = re.compile(
    r'(:\s*)(?!"|null\s*[,\}]|true\s*[,\}]|false\s*[,\}])'
    r'([^",\{\}\[\]\s][^",\{\}\[\]]*?)(\s*[,\}])')
_GROUNDING_TAG_RE = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>|<\|grounding\|>", re.DOTALL)


def _extract_json_span(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in output")
    return text[start:end + 1]


def _quote_bare_values(blob: str) -> str:
    """Wrap unquoted scalars so 01234 survives as a string, not an int."""
    prev = None
    while prev != blob:
        prev = blob
        blob = _BARE_VALUE_RE.sub(
            lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}', blob)
    return blob


def parse_json_fields(text: str, columns, line_join: str = " "):
    """Model text -> ({field: str}, extra_keys). Raises ValueError if unparsable."""
    blob = _extract_json_span(_GROUNDING_TAG_RE.sub(" ", text or ""))
    try:
        obj = json.loads(blob, parse_int=str, parse_float=str)
    except json.JSONDecodeError:
        obj = json.loads(_quote_bare_values(blob), parse_int=str, parse_float=str)
    if not isinstance(obj, dict):
        raise ValueError(f"parsed JSON is a {type(obj).__name__}, not an object")
    row, extra = {}, sorted(set(obj) - set(columns))
    for col in columns:
        val = obj.get(col, "")
        if val is None:
            val = ""
        elif isinstance(val, (list, tuple)):
            val = line_join.join(str(v) for v in val)
        elif isinstance(val, dict):
            val = ""
        row[col] = str(val).strip()
    return row, extra


def to_lines(text: str) -> list[str]:
    """Flat transcription lines. Kept for notebooks that log raw text."""
    return [_plain(s.text) for s in to_segments(text)[0]]


# -- compatibility ---------------------------------------------------------
# Kept because they only ever produced text for logging.
def strip_grounding(text: str) -> str:
    return "\n".join(to_lines(text))


def clean_lines(text: str) -> list[str]:
    return to_lines(text)


def html_to_lines(text: str) -> list[str]:
    return to_lines(text)


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"<(?:div|p|table|tr|td|input)\b", text or "",
                          re.IGNORECASE))


def _removed(name: str, why: str):
    def fail(*_a, **_k):
        raise NotImplementedError(
            f"{name}() was removed in {PARSER_VERSION}. {why}")
    return fail


# These encoded the old page-level dispatch. They are not shimmed on purpose:
# calling them would suggest the three-way branch still exists.
_WHY = ("Pages are no longer routed to one of three parsers; every page runs "
        "the same cascade and report['strategy'] records which rule produced "
        "each field.")
count_label_blocks = _removed("count_label_blocks", _WHY)
grounded_needs_geometry = _removed("grounded_needs_geometry", _WHY)
grounded_is_tabular = _removed("grounded_is_tabular", _WHY)
parse_grounded_layout = _removed("parse_grounded_layout", _WHY)
parse_grounded_blocks = _removed("parse_grounded_blocks", _WHY)


# ==========================================================================
# 11. Self-test -- `python formparse2.py`
# ==========================================================================
# Every case below is a verbatim excerpt from a val page that broke an earlier
# parser. Adding a case here is cheaper than re-discovering it as a CER anomaly.

if __name__ == "__main__":
    class _F:
        def __init__(self, name, type_="text"):
            self.name, self.type = name, type_

    FIELDS = [_F("last_name"), _F("first_name"), _F("monthly_salary"),
              _F("birth_date"), _F("email"), _F("phone"), _F("studies"),
              _F("street"), _F("house_number"), _F("postal_code"),
              _F("city"), _F("comment", "long_text")]
    LABELS = {"last_name": "Name", "first_name": "Vorname",
              "monthly_salary": "Monatliches Einkommen",
              "birth_date": "Geburtsdatum", "email": ["E-Mail-Adresse", "E-Mail"],
              "phone": "Telefonnummer", "studies": "Studiengang",
              "street": "Stra\u00dfe", "house_number": "Hausnummer",
              "postal_code": ["PLZ", "Postleitzahl"], "city": "Ort",
              "comment": "Kommentar", "signature": "Unterschrift"}
    CB = {"studies": ["IM", "WIN", "BWL", "Kunst"]}
    STOPS = ["Anschrift", "bitte ankreuzen"]

    def parse(raw):
        return parse_page(raw, FIELDS, LABELS, CB, stop_labels=STOPS)

    def g(cat, box, txt):
        return f"<|ref|>{cat}<|/ref|><|det|>{box}<|/det|>\n{txt}\n"

    # -- colspan/rowspan expansion ------------------------------------------
    cells = table_to_cells(
        '<table><tr><td>IM</td><td>WIN</td><td colspan="3">Kunst</td></tr>'
        '<tr><td></td><td></td><td colspan="3">X</td></tr></table>')
    assert (0, 2, "Kunst") in [(r, c, t) for r, c, t in cells], cells
    assert (1, 2, "X") in [(r, c, t) for r, c, t in cells], cells

    # -- DeepSeek layout A: labels first, then values (column order) ---------
    ds_col = (g("text", "[[115, 155, 193, 175]]", "Name")
              + g("text", "[[115, 216, 222, 236]]", "Vorname")
              + g("text", "[[113, 818, 245, 838]]", "Kommentar")
              + g("text", "[[410, 163, 603, 195]]", "L\u00f6wer-Sager")
              + g("text", "[[420, 225, 512, 252]]", "Emma")
              + g("text", "[[405, 826, 810, 888]]",
                  "Ich bin aktuell im Urlaub und\nantworte sp\u00e4ter"))
    row, rep = parse(ds_col)
    assert row["last_name"] == "L\u00f6wer-Sager", row
    assert row["first_name"] == "Emma", row
    # a comment block is far taller than its label: overlap, not centre, pairs them
    assert row["comment"] == "Ich bin aktuell im Urlaub und antworte sp\u00e4ter", row
    assert rep["strategy"]["comment"] == "geometric", rep

    # -- a grounded block with no text at all (layout A doc 31 signature) ----
    assert parse(ds_col + "<|ref|>image<|/ref|><|det|>[[370, 906, 600, 970]]<|/det|>"
                 )[0]["last_name"] == "L\u00f6wer-Sager"

    # -- one det holding several boxes, one per line (layout C doc 48) -------
    multi = g("text", "[[110, 589, 377, 625], [111, 620, 320, 645], "
                      "[111, 646, 312, 673], [110, 675, 368, 705]]",
              "Stra\u00dfe Harkflueg\nHausnummer 67\nPLZ 43597\nOrt 2eubwoda")
    row, _ = parse(multi)
    assert row["street"] == "Harkflueg" and row["house_number"] == "67", row
    assert row["postal_code"] == "43597" and row["city"] == "2eubwoda", row

    # -- label and value merged into one table cell (layout B docs 30, 64) ---
    tab = g("table", "[[115, 171, 873, 549]]",
            '<table><tr><td>Stra\u00dfe</td><td>Charles-Mies-Heag</td>'
            '<td colspan="2">Hausnummer 76</td></tr>'
            '<tr><td>PLZ</td><td>99050</td><td colspan="5">Ort D\u00fcrkelsb\u00fcl</td>'
            '</tr></table>')
    row, rep = parse(tab)
    assert row["street"] == "Charles-Mies-Heag", row
    assert row["house_number"] == "76", row      # inline beats grid
    assert row["city"] == "D\u00fcrkelsb\u00fcl", row
    assert rep["strategy"]["house_number"] == "inline", rep

    # -- value in a later cell, position varies by page (layout B 13 vs 31) --
    for html, want in (
            ('<tr><td>Monatliches Einkommen</td><td></td><td>542149</td></tr>',
             "542149"),
            ('<tr><td>Monatliches Einkommen</td><td>1592.01</td></tr>',
             "1592.01")):
        row, _ = parse(g("table", "[[1,2,3,4]]", f"<table>{html}</table>"))
        assert row["monthly_salary"] == want, (row, want)

    # -- layout C: "Label: value", and no colon inside the Anschrift box -----
    ds_inline = (g("text", "[[111, 174, 696, 202]]",
                   "Name: Soubertich Vorname: Doris")
                 + g("sub_title", "[[111, 570, 211, 586]]", "Anschrift:")
                 + g("text", "[[111, 650, 290, 675]]", "PLZ 6 5 4 9 4"))
    row, rep = parse(ds_inline)
    assert row["last_name"] == "Soubertich", row
    assert row["first_name"] == "Doris", row     # both from one segment
    assert row["postal_code"] == "6 5 4 9 4", row
    assert rep["strategy"]["first_name"] == "inline", rep

    # -- layout C doc 71: the space between label and value was lost ---------
    assert parse(g("text", "[[110, 590, 430, 700]]",
                   "Stra\u00dfeavenue de Loiseau")
                 )[0]["street"] == "avenue de Loiseau"

    # -- layout C doc 31: label alone, value in the NEXT block ---------------
    orphan = (g("text", "[[115, 241, 355, 260]]", "Monatliches Einkommen:")
              + g("text", "[[117, 261, 260, 290]]", "1592.04"))
    assert parse(orphan)[0]["monthly_salary"] == "1592.04"

    # -- Chandra doc 83: the value lives in an attribute, not in the text ----
    ch83 = ('<div data-bbox="791 22 969 66" data-label="Page-Header">A-83</div>'
            '<div data-bbox="112 152 888 190" data-label="Text"><p>Name '
            '<input type="text" value="P\u00e4rtzelt"/></p></div>'
            '<div data-bbox="110 714 888 752" data-label="Text"><p>PLZ '
            '<input type="text" value="01650"/></p></div>')
    row, _ = parse(ch83)
    assert row["last_name"] == "P\u00e4rtzelt", row      # not "[ ]", not ""
    assert row["postal_code"] == "01650", row        # leading zero survives

    # the Page-Header div holds the page id and must never reach a field
    assert "A-83" not in "".join(row.values()), row

    # -- <del>: dropped when other text survives, kept when it is all there --
    assert parse('<div data-bbox="1 2 3 4" data-label="Text"><p>Ort '
                 '<del>99050</del> Dinkelsb\u00fchl</p></div>'
                 )[0]["city"] == "Dinkelsb\u00fchl"
    assert parse('<div data-bbox="1 2 3 4" data-label="Text"><p>Name '
                 '<del>Gilles Pruvost-Didier</del></p></div>'
                 )[0]["last_name"] == "Gilles Pruvost-Didier"

    # -- a value that CONTAINS a printed label is still a value -------------
    for markup, want in (
            ('<div data-bbox="110 614 187 632" data-label="Text"><p>Stra\u00dfe</p>'
             '</div><div data-bbox="413 617 875 648" data-label="Text">'
             '<p>Gesdi-Lachmann-Stra\u00dfe</p></div>', "Gesdi-Lachmann-Stra\u00dfe"),
            ('<div data-bbox="110 614 187 632" data-label="Text"><p>Stra\u00dfe</p>'
             '</div><div data-bbox="413 617 875 648" data-label="Text">'
             '<p>Reimor - Dobes - Stra\u00dfe</p></div>', "Reimor - Dobes - Stra\u00dfe")):
        assert parse(markup)[0]["street"] == want, parse(markup)[0]

    # -- Chandra layout B: label row, value row underneath (docs 15, 27) -----
    stacked = ('<div data-bbox="112 174 874 552" data-label="Form"><table>'
               '<tr><td colspan="2">Stra\u00dfe</td><td colspan="2">Hausnummer</td></tr>'
               '<tr><td colspan="2">Saballee</td><td colspan="2">2/3</td></tr>'
               '</table></div>')
    row, rep = parse(stacked)
    assert row["street"] == "Saballee" and row["house_number"] == "2/3", row
    assert rep["strategy"]["street"] == "grid", rep

    # -- a wrapped comment is joined once, not duplicated -------------------
    wrapped = ('<div data-bbox="1 2 3 4" data-label="Form"><table><tr>'
               '<td colspan="2">Kommentar</td><td colspan="6">Heute Abend '
               'kochen wir Pasta mit<br/>frischem Gem\u00fcse und Kr\u00e4utern.</td>'
               '</tr></table></div>')
    got = parse(wrapped)[0]["comment"]
    assert got == ("Heute Abend kochen wir Pasta mit frischem "
                   "Gem\u00fcse und Kr\u00e4utern."), repr(got)
    assert got.count("Heute Abend") == 1, repr(got)

    # -- checkboxes ---------------------------------------------------------
    # colspan on the header must be expanded or the tick aligns to the wrong option
    cb_grid = ('<div data-bbox="1 2 3 4" data-label="Form"><table>'
               '<tr><td colspan="2" rowspan="2">Studiengang<br/>(bitte ankreuzen)'
               '</td><td>IM</td><td>WIN</td><td>BWL</td><td colspan="3">Kunst</td>'
               '</tr><tr><td></td><td></td><td></td>'
               '<td colspan="3"><input checked="" type="checkbox"/></td></tr>'
               '</table></div>')
    row, rep = parse(cb_grid)
    assert row["studies"] == "Kunst", (row, rep)
    assert rep["checkbox"]["studies"] == "grid", rep

    # markers BEFORE the labels pair by index just as well as after
    for markup, want in (
            ('Studiengang: <input type="checkbox"/> IM '
             '<input checked="" type="checkbox"/> WIN '
             '<input type="checkbox"/> BWL <input type="checkbox"/> Kunst', "WIN"),
            ('Studiengang: IM <input type="checkbox"/> WIN '
             '<input type="checkbox"/> BWL <input checked="" type="checkbox"/> '
             'Kunst <input type="checkbox"/>', "BWL")):
        raw = f'<div data-bbox="1 2 3 4" data-label="Text"><p>{markup}</p></div>'
        assert parse(raw)[0]["studies"] == want, (parse(raw)[0], want)

    # five boxes for four options: which one is spurious decides the answer,
    # and the output does not say, so nothing is returned
    five = ('<div data-bbox="1 2 3 4" data-label="Text"><p>Studiengang '
            '<input type="checkbox"/> IM <input type="checkbox"/> BWL '
            '<input type="checkbox"/> WIN <input checked="" type="checkbox"/> '
            'Kunst <input type="checkbox"/></p></div>')
    row, rep = parse(five)
    assert row["studies"] == "" , row
    assert rep["checkbox"]["studies"] == "marker_count_mismatch", rep

    # DeepSeek layout A: option and glyph in the same block, four blocks
    ds_cb = (g("text", "[[113, 519, 258, 539]]", "Studiengang")
             + g("text", "[[395, 519, 476, 543]]", "IM \u2612")
             + g("text", "[[577, 519, 666, 543]]", "BWL \u2610")
             + g("text", "[[395, 555, 476, 579]]", "WIN \u2610")
             + g("text", "[[577, 555, 666, 579]]", "Kunst \u2610"))
    assert parse(ds_cb)[0]["studies"] == "IM", parse(ds_cb)

    # no markers transcribed at all is a different failure from a misaligned one
    ds_none = (g("text", "[[113, 519, 258, 539]]", "Studiengang")
               + g("text", "[[395, 519, 476, 543]]", "IM")
               + g("text", "[[577, 519, 666, 543]]", "BWL"))
    assert parse(ds_none)[1]["checkbox"]["studies"] == "unresolved"

    # -- JSON path still works, including bare leading-zero scalars ---------
    row, extra = parse_json_fields(
        '{"postal_code": 09252, "city": "Hammelburg", "note": null, "x": 1}',
        ["postal_code", "city", "note"])
    assert row["postal_code"] == "09252", row
    assert row["note"] == "", row
    assert extra == ["x"], extra

    print("self-test ok")
