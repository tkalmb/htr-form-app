"""Field-specific correction of raw recognizer output.

Each rule is registered under a name that a layout's ``postprocess`` key can
refer to, so adding a form type is a JSON change and adding a *kind* of
correction is a new function here.

Governing principle throughout: **never force a fix that does not produce a
demonstrably valid result.** Every rule either yields something that passes an
independent check (matches the field's pattern, is a real calendar date, is in
a closed lexicon) or returns the input untouched. Corrections that cannot be
validated are not applied.
"""

from __future__ import annotations

import calendar
import difflib
import re
import warnings
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from .config import PostprocessConfig
from .schema import FieldSpec, LayoutSpec

# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

try:  # rapidfuzz is orders of magnitude faster than a pure-Python DP table,
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein  # noqa

    def edit_distance(a: str, b: str) -> int:
        """Unweighted Levenshtein distance."""
        return _RF_Levenshtein.distance(a, b)

    _HAS_RAPIDFUZZ = True
except ImportError:  # ...but the pipeline should not hard-depend on it.
    def edit_distance(a: str, b: str) -> int:
        """Unweighted Levenshtein distance (pure-Python fallback)."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                current.append(min(previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + (ca != cb)))
            previous = current
        return previous[-1]

    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Character confusion tables
# ---------------------------------------------------------------------------

#: predicted character -> corrected digit.
#:
#: Hand-specified rather than learned. A data-driven confusion matrix would
#: need more annotated forms than a low-data tier is allowed to use, which
#: works against the minimal-annotation premise. Sources:
#:   * observed on this project's train (40% tier) + val errors: H->4, k->4,
#:     G->6, j->9
#:   * general visual-confusability grounds: o/O->0, s/S->5, l/I/i->1, B->8,
#:     g->9, q->9, Z->2
DEFAULT_DIGIT_CONFUSION_MAP: Dict[str, str] = {
    "H": "4", "k": "4",
    "G": "6",
    "j": "9", "g": "9", "q": "9",
    "o": "0", "O": "0",
    "s": "5", "S": "5", "r": "5",
    "l": "1", "I": "1", "i": "1",
    "B": "8",
    "Z": "2",
    "F": "7",
}

#: digit -> most plausible original letter, for cleaning names.
#: Several letters map to the same digit in the forward table, so this makes a
#: single documented best guess. ``3`` and ``7`` are absent from the forward
#: table, so a stray one is deleted rather than replaced with a guess.
REVERSE_DIGIT_MAP: Dict[str, str] = {
    "0": "o", "1": "i", "2": "z", "4": "h",
    "5": "s", "6": "g", "8": "b", "9": "g",
}


def apply_digit_map(text: str, digit_map: Dict[str, str]) -> str:
    return "".join(digit_map.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

#: Expand German diacritics so "München"/"Muenchen" compare as equal. It must
#: be expansion (1 char -> 2), not folding: a character-for-character
#: substitution table cannot express a 1->2 correspondence.
_GERMAN_DIACRITICS = str.maketrans({
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
})


def normalize_for_matching(text: str) -> str:
    """Lowercase + expand diacritics. Used only to compute distances -- the
    original lexicon spelling is what gets returned."""
    return text.lower().translate(_GERMAN_DIACRITICS)


# ---------------------------------------------------------------------------
# Rule: numeric
# ---------------------------------------------------------------------------

def correct_numeric(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    """Apply the digit-confusion map, gated on the field's ``pattern``.

    Three-step logic: if the raw value already matches, leave it; otherwise
    apply the map; keep the result only if it now matches. Without a pattern
    there is nothing to validate against, so this is a no-op.
    """
    pattern = spec.compiled_pattern
    if pattern is None or pattern.match(text):
        return text
    fixed = apply_digit_map(text, res.digit_map)
    return fixed if pattern.match(fixed) else text


# ---------------------------------------------------------------------------
# Rule: date
# ---------------------------------------------------------------------------

GERMAN_MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]
_MONTH_ALIASES = {"maerz": "März"}
_MONTH_LOOKUP = {name.lower(): name for name in GERMAN_MONTHS}
_MONTH_LOOKUP.update(_MONTH_ALIASES)
_MONTH_TO_NUM = {name.lower(): i + 1 for i, name in enumerate(GERMAN_MONTHS)}

_DATE_NUMERIC_RE = re.compile(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})$")

#: Day and year positions accept letters too, so a confusable letter standing
#: in for a digit is still recognised and can be fixed. The month requires 3+
#: pure letters -- the shortest German month name is "Mai", and a numeric date
#: never contains a run of three letters, which is what keeps this pattern
#: from matching numeric dates.
_DATE_TEXT_MONTH_RE = re.compile(
    r"^([\dA-Za-zÄÖÜäöüß]{1,2})\.?\s*([A-Za-zÄÖÜäöüß]{3,})\s*([\dA-Za-zÄÖÜäöüß]{2,4})$"
)

def _is_real_date(day_str: str, month_num: int, year_str: str) -> bool:
    """Leap-year-aware calendar validity check."""
    if not (1 <= month_num <= 12):
        return False
    if not day_str.isdigit() or not year_str.isdigit():
        return False
    # Two-digit years get an assumed century for the leap-year check only.
    # This assumption never leaves this function.
    year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)
    return 1 <= int(day_str) <= calendar.monthrange(year, month_num)[1]


def is_plausible_date(text: str) -> bool:
    """True if ``text`` is a numeric date naming a real calendar day."""
    match = _DATE_NUMERIC_RE.match(text)
    if not match:
        return False
    day_str, month_str, year_str = match.groups()
    return month_str.isdigit() and _is_real_date(day_str, int(month_str), year_str)


def is_plausible_birthday(text: str) -> bool:
    """Plausibility check covering both numeric and spelled-month formats.

    Used as an *error-detection* signal, not a correction: it says whether a
    value could be a real date, not whether it is the right one. A wrong but
    valid date passes -- that blind spot is the point of measuring its recall.
    """
    if is_plausible_date(re.sub(r"\s+", "", text)):
        return True
    match = _DATE_TEXT_MONTH_RE.match(text.strip())
    if not match:
        return False
    day_str, month_word, year_str = match.groups()
    canonical = _MONTH_LOOKUP.get(month_word.lower())
    if canonical is None:
        return False
    return _is_real_date(day_str, _MONTH_TO_NUM[canonical.lower()], year_str)


def correct_month_word(word: str, max_relative_distance: float = 0.34) -> Optional[str]:
    """Nearest German month name, or ``None`` if no confident match."""
    lowered = word.lower()
    if lowered in _MONTH_LOOKUP:
        return _MONTH_LOOKUP[lowered]

    best_name, best_dist = None, float("inf")
    for name in GERMAN_MONTHS:
        dist = edit_distance(lowered, name.lower())
        if dist < best_dist:
            best_name, best_dist = name, dist

    if best_name is not None and best_dist <= max_relative_distance * max(len(lowered), 1):
        return best_name
    return None


def correct_date(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    """Correct a date in either numeric or spelled-out-month form.

    Both branches fix digits with the confusion map and keep the result only
    if it yields a real calendar date. The spelled-month branch additionally
    corrects the month word and reformats to ``D. Monthname YYYY``.
    """
    match = _DATE_TEXT_MONTH_RE.match(text.strip())
    if match:
        day_raw, month_raw, year_raw = match.groups()
        day = apply_digit_map(day_raw, res.digit_map)
        year = apply_digit_map(year_raw, res.digit_map)
        month = correct_month_word(month_raw, res.config.month_max_relative_distance)
        if month is not None and _is_real_date(day, _MONTH_TO_NUM[month.lower()], year):
            return f"{day}. {month} {year}"
        return text  # not confidently correctable -> leave alone

    compact = re.sub(r"\s+", "", text)
    if is_plausible_date(compact):
        return compact
    fixed = apply_digit_map(compact, res.digit_map)
    return fixed if is_plausible_date(fixed) else text


# ---------------------------------------------------------------------------
# Rule: lexicon
# ---------------------------------------------------------------------------

def correct_with_lexicon(
    text: str,
    lexicon: Sequence[str],
    max_relative_distance: float = 0.25,
) -> str:
    """Snap ``text`` to its nearest lexicon entry if close enough.

    Distances are computed on normalised forms; the lexicon's own spelling and
    casing is what gets returned. The ~1/5-of-length threshold follows common
    practice in lexical OCR correction; tighten it to be more conservative.

    Only appropriate for genuinely closed vocabularies. Applying it to an open
    set such as person names corrects valid values into different valid
    values -- see :func:`correct_name`.
    """
    if not text or not lexicon or text in lexicon:
        return text

    normalized = normalize_for_matching(text)
    best_entry, best_dist = None, float("inf")
    for entry in lexicon:
        dist = edit_distance(normalized, normalize_for_matching(entry))
        if dist < best_dist:
            best_entry, best_dist = entry, dist
            if dist == 0:
                break

    if best_entry is not None and best_dist <= max_relative_distance * max(len(normalized), 1):
        return best_entry
    return text


def rule_lexicon(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    lexicon = res.lexicons.get(spec.lexicon)
    if not lexicon:
        warnings.warn(
            f"field {spec.name!r} requests lexicon {spec.lexicon!r}, which is "
            f"empty or missing from the supplied resources -- leaving values "
            f"unchanged.", stacklevel=2,
        )
        return text
    return correct_with_lexicon(text, lexicon, res.config.lexicon_max_relative_distance)


# ---------------------------------------------------------------------------
# Rule: email
# ---------------------------------------------------------------------------

def correct_email(text: str, domains: Set[str], max_relative_distance: float = 0.34) -> str:
    """Normalise whitespace/case and snap the domain to a known one.

    The local part is left untouched: in this dataset it is generated
    independently of the name fields, so there is nothing to cross-check it
    against. Splitting on the *last* ``@`` tolerates a spurious extra one
    introduced by recognition noise.
    """
    text = re.sub(r"\s+", "", text).lower()
    if "@" not in text:
        return text

    local, _, domain = text.rpartition("@")
    if domain in domains:
        return f"{local}@{domain}"

    best, best_dist = None, float("inf")
    for candidate in domains:
        dist = edit_distance(domain, candidate)
        if dist < best_dist:
            best, best_dist = candidate, dist

    if best is not None and best_dist <= max_relative_distance * max(len(domain), 1):
        return f"{local}@{best}"
    return f"{local}@{domain}"


def rule_email(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    return correct_email(text, res.email_domains, res.config.email_max_relative_distance)


# ---------------------------------------------------------------------------
# Rule: street suffix
# ---------------------------------------------------------------------------

DEFAULT_STREET_SUFFIXES = [
    "straße", "strasse", "str.",
    "weg", "allee", "gasse", "platz", "ring",
    "damm", "ufer", "steig", "pfad", "promenade", "chaussee",
]


def _suffix_threshold(target: str, max_relative_distance: float = 0.34) -> int:
    """Per-target threshold, so "promenade" tolerates more edits than "weg"."""
    return max(1, round(max_relative_distance * len(target)))


def _best_suffix(candidate: str, targets: Sequence[str], max_relative_distance: float):
    lowered = candidate.lower()
    best, best_dist = None, float("inf")
    for target in targets:
        dist = edit_distance(lowered, target)
        if dist <= _suffix_threshold(target, max_relative_distance) and dist < best_dist:
            best, best_dist = target, dist
    return best, best_dist


def correct_street_suffix(
    text: str,
    targets: Sequence[str] = tuple(DEFAULT_STREET_SUFFIXES),
    max_relative_distance: float = 0.34,
) -> str:
    """Correct only the trailing street-type word; never the base name.

    Two shapes are handled differently. A hyphenated ``Firstname-Lastname-Weg``
    already marks the boundary, so the last segment is compared whole and
    capitalised if corrected. A plain compound like ``Hauptstraße`` has no
    boundary marker, so trailing windows near each target's length are tried
    and the suffix stays lowercase.

    Abbreviations are independent targets rather than things to expand:
    forcing ``Höfigstr.`` to ``Höfigstraße`` would be wrong whenever the
    source genuinely used the short form.
    """
    stripped = text.strip()
    targets = list(targets)

    if "-" in stripped:
        parts = stripped.split("-")
        target, _ = _best_suffix(parts[-1], targets, max_relative_distance)
        if target is not None:
            return "-".join(parts[:-1] + [target.capitalize()])
        return stripped

    lowered = stripped.lower()
    best_target, best_len, best_dist = None, None, float("inf")
    for target in targets:
        threshold = _suffix_threshold(target, max_relative_distance)
        for length in (len(target), len(target) - 1, len(target) + 1):
            if length < 1 or length > len(lowered):
                continue
            dist = edit_distance(lowered[-length:], target)
            if dist <= threshold and dist < best_dist:
                best_target, best_len, best_dist = target, length, dist

    if best_target is not None:
        return f"{stripped[:len(stripped) - best_len]}{best_target}"
    return stripped


def rule_street_suffix(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    return correct_street_suffix(text, res.street_suffixes,
                                res.config.street_max_relative_distance)


# ---------------------------------------------------------------------------
# Rule: name
# ---------------------------------------------------------------------------

#: A word inside a name: any run not broken by whitespace or a hyphen.
_NAME_WORD_RE = re.compile(r"[^\s\-]+")


def clean_name(text: str, reverse_map: Dict[str, str] = None) -> str:
    """Strip characters that cannot legitimately occur in a name.

    Digits are mapped back to their most plausible letter where the forward
    confusion table justifies one, and dropped otherwise. Hyphen and
    apostrophe are kept (legitimate in names); whitespace is preserved so
    multi-word names can be capitalised word by word afterwards.
    """
    reverse_map = REVERSE_DIGIT_MAP if reverse_map is None else reverse_map
    chars = []
    for ch in text:
        if ch.isalpha() or ch in "-'" or ch.isspace():
            chars.append(ch)
        elif ch in reverse_map:
            chars.append(reverse_map[ch])
    return " ".join("".join(chars).split())


def fix_name_capitalization(text: str) -> str:
    """Title-case each word, treating both hyphen and space as boundaries --
    so ``anne-marie`` and ``osuna milla`` both come out right."""
    return _NAME_WORD_RE.sub(lambda m: m.group(0).capitalize(), text)


def correct_name(text: str, spec: FieldSpec = None, res: "PostprocessResources" = None) -> str:
    """Character cleanup and capitalisation only -- deliberately no lexicon.

    A lexicon step was tested and removed: with a training pool of a handful
    of names, ``Bachmann`` (a genuine, different, valid name) was "corrected"
    to ``Lachmann`` simply because that happened to be in the lexicon. Unlike
    cities, there is no complete list of valid German names to substitute in,
    so nearest-neighbour guessing reproduces that failure at any threshold.

    What remains fixes only what is unconditionally wrong (a digit in a name)
    or purely cosmetic (casing) -- neither guesses at content. The cost is
    that a genuine letter-for-letter error (``rn`` read as ``m``) is left
    uncorrected.
    """
    reverse_map = res.reverse_digit_map if res is not None else REVERSE_DIGIT_MAP
    return fix_name_capitalization(clean_name(text, reverse_map))


# ---------------------------------------------------------------------------
# Rule: LLM (free text)
# ---------------------------------------------------------------------------

DEFAULT_COMMENT_SYSTEM_PROMPT = """You are an expert German text correction system specialized in post-processing handwritten text recognition (HTR) output. Your task is to correct errors introduced by the HTR model while preserving the original intent and content.

**Input:** Raw text extracted from a handwritten "Kommentar" field on a German form, which may contain German or English or Spanish or French text.
**Output:** Corrected text that is readable, grammatically sensible, and preserves meaning

**Correction Guidelines:**
1. Fix common OCR/HTR character substitution errors, for example:
   - 0 <-> O
   - 1 <-> I or l
   - 8 <-> B
   - rn <-> m
2. Correct special symbols that might have been incorrectly interpreted, for example:
   - Hello/ -> Hello!
3. Fix corrupted German umlauts (ä, ö, ü)
   - StraBe -> Straße
4. Ensure grammatical coherence (agreement, word order, sentence structure)
5. Preserve the original language

**Important:**
- Do NOT add information that wasn't in the original text
- Do NOT paraphrase or rewrite sentences
- If the sentence already makes sense, do NOT change it
- Only fix errors; do not "improve" the writing style
- Preserve the language as written
- When correcting a word, consider if will make sense after the change. If not, do not change it.

**Output format:**
Return ONLY the corrected text. No explanations, no quotation marks.

**Examples:**
Input: "Am Wochenende besuchen wirmeine GroBeltern, die einen schönen barten haben."
Output: "Am Wochenende besuchen wir meine Großeltern, die einen schönen Garten haben."

Input: "Ich benötige Support für API-Zuaritte"
Output: "Ich benötige Support für API-Zugriffe."

Input: "No issues found during the installation Process,"
Output: "No issues found during the installation process."
"""


def strip_added_terminal_punctuation(original: str, corrected: str) -> str:
    """Drop a trailing ``.``/``!``/``?`` the model added on its own -- cosmetic,
    so it should not count towards "how much did the model change"."""
    if corrected and corrected[-1] in ".!?" and not (original and original[-1] in ".!?"):
        return corrected[:-1].rstrip()
    return corrected


def accept_llm_correction(
    original: str,
    raw_output: str,
    max_word_edit_distance: int = 2,
    max_word_count_diff: int = 2,
    max_changed_word_fraction: float = 0.3,
    min_changed_word_allowance: int = 2,
    allow_word_splits: bool = False,
) -> str:
    """Accept the model's output, or fall back to the original.

    Separated from generation so it can be unit-tested without the model.

    Alignment is done with ``difflib.SequenceMatcher`` rather than a positional
    ``zip``. Once word counts are allowed to differ at all, ``zip`` misaligns
    every word after an insertion or deletion, comparing pairs that were never
    meant to correspond -- a real bug in an earlier version of this check.

    Four layered conditions, any of which rejects the output:

    1. word count differs by more than ``max_word_count_diff`` (substantial
       rewriting, not a character fix)
    2. an aligned ``replace`` block pairs spans of unequal length
       (structurally a rewording, not a like-for-like swap)
    3. any aligned word pair differs by more than ``max_word_edit_distance``
       characters (a typo fix is 1-2 edits; more suggests the model
       substituted a different word than what was written)
    4. the total fraction of words touched exceeds the cap -- a whole-sentence
       net against many individually-passing edits adding up to a rewrite.
       The minimum allowance matters: comments here are often 2-4 words, where
       one legitimate fix already exceeds any pure fraction.

    Known gaps, measured rather than assumed
    ----------------------------------------
    * A wrong substitution that is short, same-length and similarly-spelled
      passes condition 3.
    * **Condition 2 rejects word-split fixes.** ``"wirmeine"`` ->
      ``["wir", "meine"]`` is an unequal-length replace block, so it is always
      rejected -- even though splitting a run-together word is exactly the
      first example in the system prompt. The prompt teaches a correction the
      guardrail can never accept.
    * Conversely, a *pure insertion* of an unrelated word (``"a b c d"`` ->
      ``"a b XXXXXXXX c d"``) is accepted, because insertions are only counted
      towards condition 4, never inspected character-wise.

    Set ``allow_word_splits=True`` to accept an unequal-length replace block
    when joining the two spans gives nearly the same characters -- which
    admits ``"wirmeine"`` -> ``"wir meine"`` while still rejecting a genuine
    rewording. This is **off by default** so existing tuned behaviour is
    unchanged; turn it on deliberately and re-check a sample of outputs.
    """
    corrected = raw_output.strip().strip('"').strip("'").strip()
    if not corrected:
        return original

    corrected = strip_added_terminal_punctuation(original, corrected)

    original_words = original.split()
    corrected_words = corrected.split()

    if abs(len(original_words) - len(corrected_words)) > max_word_count_diff:
        return original

    matcher = difflib.SequenceMatcher(a=original_words, b=corrected_words, autojunk=False)
    n_touched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            orig_span, corr_span = original_words[i1:i2], corrected_words[j1:j2]
            if len(orig_span) != len(corr_span):
                if not allow_word_splits:
                    return original
                # Same characters, different word boundaries -> a split/merge
                # fix rather than a rewording. Compare the joined spans.
                joined_original = "".join(orig_span).lower()
                joined_corrected = "".join(corr_span).lower()
                if edit_distance(joined_original, joined_corrected) > max_word_edit_distance:
                    return original
                n_touched += max(len(orig_span), len(corr_span))
                continue
            for orig_word, corr_word in zip(orig_span, corr_span):
                if edit_distance(orig_word.lower(), corr_word.lower()) > max_word_edit_distance:
                    return original
            n_touched += len(orig_span)
        elif tag in ("insert", "delete"):
            n_touched += max(i2 - i1, j2 - j1)

    threshold = max(min_changed_word_allowance,
                    round(max_changed_word_fraction * len(original_words)))
    if n_touched > threshold:
        return original

    return corrected


class CommentLLM:
    """Lazily-loaded local instruction-tuned model for free-text correction.

    Local rather than an API: a form-processing pipeline for administrative
    documents should not depend on sending field contents to a third party,
    independent of whether this particular dataset is synthetic.

    The model is loaded on first use, so importing this module -- or running a
    pipeline whose layout has no ``llm`` field -- costs nothing.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        system_prompt: str = DEFAULT_COMMENT_SYSTEM_PROMPT,
        config: Optional[PostprocessConfig] = None,
        max_new_tokens: Optional[int] = None,
        guardrail_kwargs: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.system_prompt = system_prompt
        self.config = config or PostprocessConfig()
        #: Explicit arguments win over the config, so a one-off experiment does
        #: not require editing the shared settings.
        self.max_new_tokens = (max_new_tokens if max_new_tokens is not None
                               else self.config.llm_max_new_tokens)
        self.guardrail_kwargs = (guardrail_kwargs if guardrail_kwargs is not None
                                 else self.config.guardrail_kwargs())
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .recognize import resolve_device

            device = resolve_device(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype="auto"
            ).to(device)
            self._model.eval()
            self._torch = torch
        return self._model, self._tokenizer

    def generate(self, text: str) -> str:
        """Raw model output, before the guardrail. Greedy decoding, because
        this task wants determinism, not variation."""
        model, tokenizer = self._load()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with self._torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def correct(self, text: str) -> str:
        if not text.strip():
            return text
        return accept_llm_correction(text, self.generate(text), **self.guardrail_kwargs)


def rule_llm(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    if res.comment_llm is None:
        warnings.warn(
            f"field {spec.name!r} requests the 'llm' rule but no CommentLLM was "
            f"supplied in the resources -- leaving values unchanged.",
            stacklevel=2,
        )
        return text
    return res.comment_llm.correct(text)


# ---------------------------------------------------------------------------
# Frame dtype safety
# ---------------------------------------------------------------------------
#
# A recognizer only ever emits strings. Any numeric dtype appearing in a
# predictions or ground-truth frame is therefore always the result of a dtype
# *inference* somewhere upstream -- overwhelmingly ``pd.read_csv`` deciding
# that a column of digits is an ``int64``.
#
# That inference is lossy and irreversible. ``"02125"`` becomes ``2125``, and
# a later ``astype(str)`` yields ``"2125"``: the leading zero is gone for good.
# In this project it silently corrupts ``postal_code`` (German PLZ starting
# with 0), ``phone`` (leading 0 in every German area code) and ``house_number``.
#
# Worse, it is *data-dependent* and so intermittent: a single unparseable cell
# anywhere in the column (a garbled recognition such as ``86Zro``) forces the
# column to ``object`` and accidentally protects every other value in it. A run
# can therefore be correct purely because it contained an error, and a later,
# better run silently loses leading zeros. That is exactly the kind of bug that
# must be caught by an assertion rather than by inspection.
#
# The guard below is deliberately loud: by default it *raises*, because a
# corrupted value cannot be repaired downstream and a silently wrong number in
# a thesis table is worse than a failed cell.


def _to_text(value) -> str:
    """One cell -> ``str``, mapping missing values to the empty string.

    ``float`` is handled explicitly: a column that acquired a ``NaN`` becomes
    ``float64``, and a plain ``str()`` on it would render ``3394`` as
    ``"3394.0"``. That branch is a last-resort repair for a frame that already
    lost its dtype; it cannot restore a leading zero and is not a substitute
    for reading the file correctly in the first place.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
    if isinstance(value, bool):
        return str(value)
    return str(value)


def numeric_dtype_columns(df, exclude: Iterable[str] = ()) -> List[str]:
    """Columns of ``df`` whose pandas dtype is numeric.

    Non-empty output means leading zeros have already been destroyed in those
    columns and the frame cannot be trusted.
    """
    import pandas as pd

    excluded = set(exclude)
    return [c for c in df.columns
            if c not in excluded
            and not pd.api.types.is_bool_dtype(df[c])
            and pd.api.types.is_numeric_dtype(df[c])]


def ensure_text_frame(
    df,
    exclude: Iterable[str] = (),
    on_numeric: str = "raise",
    name: str = "predictions",
):
    """Return a copy of ``df`` in which every value is a plain ``str``.

    This is a *guard*, not a repair: if a column already arrived as a numeric
    dtype the damage is done, and the point of this function is to say so.

    ``on_numeric``
        ``"raise"``  -- refuse to continue (default; correct for a pipeline run)
        ``"warn"``   -- report and coerce anyway (for inspecting a legacy CSV)
        ``"ignore"`` -- coerce silently

    ``exclude``
        Column names that are legitimately non-text. Nothing produced by a
        layout belongs here -- even a checkbox group yields an option *label* --
        so this is only for bookkeeping columns a notebook has attached.
    """
    if on_numeric not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_numeric must be 'raise', 'warn' or 'ignore', got {on_numeric!r}")

    offenders = numeric_dtype_columns(df, exclude)
    if offenders and on_numeric != "ignore":
        message = (
            f"{name}: column(s) {offenders} have a numeric dtype. A recognizer "
            f"only emits strings, so this is a dtype inference (almost always "
            f"pd.read_csv without dtype=str) and any leading zeros in them are "
            f"already lost -- '02125' is now 2125 and cannot be recovered. "
            f"Read the CSV with read_predictions_text() (or "
            f"pd.read_csv(..., dtype=str, keep_default_na=False)) and re-run."
        )
        if on_numeric == "raise":
            raise TypeError(message)
        warnings.warn(message, stacklevel=2)

    out = df.copy()
    excluded = set(exclude)
    for column in out.columns:
        if column not in excluded:
            out[column] = out[column].map(_to_text)
    return out


def write_predictions_text(df, path, exclude: Iterable[str] = (), name: str = "predictions"):
    """Write a predictions CSV as text, then verify it reads back unchanged.

    Use in place of a bare ``to_csv``/``save_predictions``. The guard runs
    before the write, so a frame that already lost its leading zeros fails here
    rather than producing a clean-looking file full of wrong numbers.

    ``float_format`` is pinned even though every column is a string by this
    point: it costs nothing and removes the scientific-notation failure mode if
    a numeric column is ever added deliberately.
    """
    import pathlib

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    text = ensure_text_frame(df, exclude=exclude, on_numeric="raise", name=name)
    text.to_csv(target, float_format="%.17g")
    assert_roundtrip_preserved(text, target, exclude=exclude, name=name)
    return target


def read_predictions_text(path, index_col: int = 0, numeric_index: bool = True):
    """Read a predictions CSV with **every column forced to ``str``**.

    This is the counterpart of the guard above and the actual fix: dtype has to
    be pinned at read time, because by the time the frame exists it is too late.

    ``keep_default_na=False`` matters as much as ``dtype=str``: without it an
    empty field, and any value pandas recognises as a null token, comes back as
    ``NaN`` instead of ``""`` -- which re-introduces a float dtype through the
    back door.

    ``numeric_index`` restores an integer ``doc_id`` index, since ``dtype=str``
    applies to the index too and downstream code joins on integer doc ids.
    Leading zeros are irrelevant for a doc id but meaningful for a field, hence
    the asymmetry.
    """
    import pandas as pd

    df = pd.read_csv(
        path,
        index_col=index_col,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    if numeric_index and len(df.index):
        as_str = df.index.astype(str)
        if all(v.lstrip("-").isdigit() for v in as_str):
            df.index = as_str.astype(int)
    return df


def assert_roundtrip_preserved(df, path, exclude: Iterable[str] = (), name: str = "predictions"):
    """Re-read a just-written CSV and assert it still says what the frame said.

    Cheap, and the only thing that actually proves the file on disk is usable.
    A ``to_csv`` never loses a leading zero; a careless ``read_csv`` does, so
    the write must be checked by reading it back the way consumers will.
    """
    reloaded = read_predictions_text(path)
    original = ensure_text_frame(df, exclude=exclude, on_numeric="raise", name=name)

    shared_columns = [c for c in original.columns if c in reloaded.columns]
    missing = [c for c in original.columns if c not in reloaded.columns]
    if missing:
        raise AssertionError(f"{name}: column(s) {missing} did not survive the CSV round-trip")

    mismatches = []
    for doc_id in original.index.intersection(reloaded.index):
        for column in shared_columns:
            before, after = original.at[doc_id, column], reloaded.at[doc_id, column]
            if before != after:
                mismatches.append((doc_id, column, before, after))

    if mismatches:
        preview = "; ".join(f"{d}/{c}: {b!r} -> {a!r}" for d, c, b, a in mismatches[:10])
        raise AssertionError(
            f"{name}: {len(mismatches)} value(s) changed in the CSV round-trip. "
            f"First few: {preview}"
        )
    return len(original.index.intersection(reloaded.index)) * len(shared_columns)


# ---------------------------------------------------------------------------
# Registry and driver
# ---------------------------------------------------------------------------

def rule_none(text: str, spec: FieldSpec, res: "PostprocessResources") -> str:
    return text


RULES: Dict[str, Callable[[str, FieldSpec, "PostprocessResources"], str]] = {
    "none": rule_none,
    "numeric": correct_numeric,
    "date": correct_date,
    "lexicon": rule_lexicon,
    "email": rule_email,
    "street_suffix": rule_street_suffix,
    "name": correct_name,
    "llm": rule_llm,
}


@dataclass
class PostprocessResources:
    """External data the rules need, supplied once by the notebook.

    Keeping these out of the rule functions is what lets the same rule serve
    different forms: a new layout supplies its own lexicons and patterns
    without any rule being rewritten.
    """

    digit_map: Dict[str, str] = dc_field(default_factory=lambda: dict(DEFAULT_DIGIT_CONFUSION_MAP))
    reverse_digit_map: Dict[str, str] = dc_field(default_factory=lambda: dict(REVERSE_DIGIT_MAP))
    #: name -> list of valid values, referenced by a field's ``lexicon`` key.
    lexicons: Dict[str, List[str]] = dc_field(default_factory=dict)
    email_domains: Set[str] = dc_field(default_factory=set)
    street_suffixes: List[str] = dc_field(default_factory=lambda: list(DEFAULT_STREET_SUFFIXES))
    comment_llm: Optional[CommentLLM] = None
    #: Thresholds. Kept separate from the data above so a run's tuning is
    #: visible in one place and recorded in the run manifest.
    config: PostprocessConfig = dc_field(default_factory=PostprocessConfig)

    def load_lexicon_from_csv(self, name: str, path: str, column: str = "name") -> int:
        """Read a lexicon from a CSV column. Returns how many entries loaded."""
        import pandas as pd

        series = pd.read_csv(path)[column].dropna().astype(str)
        values = sorted({v.strip() for v in series if v.strip()})
        self.lexicons[name] = values
        return len(values)

    def load_lexicon_from_lines(self, name: str, path: str) -> int:
        """Read a lexicon from a text file, one entry per line."""
        with open(path, encoding="utf-8") as fh:
            values = sorted({line.strip() for line in fh if line.strip()})
        self.lexicons[name] = values
        return len(values)


def validate_resources(layout: LayoutSpec, res: PostprocessResources) -> List[str]:
    """Return a list of problems that would make post-processing a silent
    no-op. Call this before a run rather than discovering it in the results."""
    problems: List[str] = []
    for spec in layout.fields:
        rule = spec.effective_postprocess
        if rule == "lexicon" and not res.lexicons.get(spec.lexicon):
            problems.append(
                f"field {spec.name!r}: lexicon {spec.lexicon!r} is empty or missing"
            )
        if rule == "email" and not res.email_domains:
            problems.append(f"field {spec.name!r}: no email domains supplied")
        if rule == "llm" and res.comment_llm is None:
            problems.append(f"field {spec.name!r}: no CommentLLM supplied")
        if rule == "numeric" and not spec.pattern:
            problems.append(f"field {spec.name!r}: 'numeric' rule needs a 'pattern'")
    return problems


def postprocess_value(value: str, spec: FieldSpec, res: PostprocessResources) -> str:
    """Apply one field's rule to one value."""
    return RULES[spec.effective_postprocess](value, spec, res)


def postprocess_predictions(df, layout: LayoutSpec, res: PostprocessResources,
                            on_numeric: str = "raise"):
    """Apply each field's rule to the matching column of a predictions frame.

    Columns not described by the layout are passed through untouched, so a
    diagnostic column added to the frame does not need a rule.

    The frame is checked for numeric dtypes before any rule runs. The previous
    version coerced with ``.astype(str)`` inside the loop, which looks like it
    handles the same problem but does not: by then ``2125`` is already an int
    and ``astype(str)`` faithfully produces ``"2125"``, so a lost leading zero
    passed through post-processing invisibly and was scored as a recognition
    error. Pass ``on_numeric="warn"`` to inspect a legacy frame anyway.
    """
    out = ensure_text_frame(df, on_numeric=on_numeric, name="predictions")
    for column in out.columns:
        try:
            spec = layout[column]
        except KeyError:
            continue
        rule = RULES[spec.effective_postprocess]
        out[column] = out[column].apply(lambda value, s=spec: rule(value, s, res))
    return out


def diff_predictions(before, after, fields: Optional[Iterable[str]] = None):
    """Long-format table of every cell post-processing actually changed.

    The most useful artefact of the whole stage: it shows what the rules did,
    which is both a debugging tool and the evidence for whether a rule earns
    its place.
    """
    import pandas as pd

    columns = list(fields) if fields else [c for c in before.columns if c in after.columns]
    rows = []
    for doc_id in before.index.intersection(after.index):
        for column in columns:
            old, new = before.at[doc_id, column], after.at[doc_id, column]
            if old != new:
                rows.append({"doc_id": doc_id, "field": column, "before": old, "after": new})
    if not rows:
        return pd.DataFrame(columns=["doc_id", "field", "before", "after"]).set_index(["doc_id", "field"])
    return pd.DataFrame(rows).set_index(["doc_id", "field"])
