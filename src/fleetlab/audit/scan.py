"""Find the facts of the truth inventory in what a session received and wrote.

Three questions, in this order.

**Does a fact stand in this text?** `Scanner.candidates` reads a text as numbers,
names and encoded blobs and looks each up in the inventory:

- a number by its value, so 14, 14.0 and 1.4e1 are one number, with or without a
  sign, before a full stop, after an underscore, inside JSON;
- a long number also when its digits are separated (thousands separators, spaces,
  a line break), split into two numbers a few words apart, written as a float, in
  hexadecimal, as raw bytes in base64, or rounded;
- a name whatever its case and whatever joins its words (`a_b`, `a b`, `aB`);
- all of that again inside base64, hexadecimal, URL-encoded and escaped text,
  three layers deep.

**Does it count?** A hidden value that the public surface shows as well (the
number 3, the word "plain") counts only where one of its hidden names or
subjects stands beside it; a value the public surface does not show counts
wherever it stands. `Scanner.learn_public` reads the public surface with the
same reader and remembers what it shows; `Scanner.scan` drops exactly that.
So the scan of the public surface itself returns nothing, by construction, and
`selftest` checks that it really does.

That would make any disclosure in the public surface public. So
`Scanner.overlap` lists what the public surface shows of the inventory, each
entry with a status: innocent by a rule that is safe, accepted by a person on
record, or unresolved. An audit with an unresolved entry is not clean
(`verdict.finish`).

**What does it prove?** `judge`: a value no coincidence and no guess can
produce (a seed, a canary, a hidden property name) is a leak wherever it
stands. Anything a tool returned bound to its hidden name is a leak. A short
number that stands alone, or a value the session wrote itself beside its name,
is listed for a person to judge; so is a number of one or two digits or an
ordinary word that the public data does not show and a tool returned alone.
An echo proves nothing, and an echo is only this: what a tool returns of the
request of that same call. What a tool returns is otherwise judged on its own,
whatever the session wrote before and wherever: an earlier call, a note in
memory, a request, a submission.

`scan_session` applies this to the files of one session (answer, tool log, CLI
stream, notes, requests, replies), part by part, so that the host's own
measurements in those files (token counts, durations, clock times) are not
taken for the session's output. One finding per fact and session. A record
that holds nothing is a problem, and that session was not scanned: a call
record that names no tool or holds nothing that came back, an event without
a type, a result record without a status or with a null answer and no reason
for it, other calls than the result record lists.

`NOT_RECOGNISED` says what this scan does not recognise; every verdict
carries it.
"""

from __future__ import annotations

import base64
import binascii
import bisect
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pyarrow.parquet as pq

from fleetlab.audit.truth import Fact, canonical_number, significant_digits

__all__ = [
    "NOT_RECOGNISED",
    "OWN_PLACES",
    "RETURNED_PLACES",
    "TOOL_LOG",
    "Candidate",
    "Finding",
    "Reader",
    "Scanner",
    "SessionScan",
    "fold",
    "judge",
    "says_nothing_alone",
    "scan_session",
    "scan_texts",
    "scan_tool_log",
    "source_of",
    "words_of",
]

# Single words that name a kind of truth. Beside a value they bind it like a hidden name
# does. Any other single word that is a key in a truth file ("exact", "points", "true")
# is an ordinary word, and binds only as a key with its separator: `"points": -2.083`.
NAME_WORDS = frozenset({"level", "seed", "onset", "oracle"})
KEY_FORM = re.compile(r"""^["']?\s*[:=]\s*["'\[({]{0,3}$""")
# How far from a value one of its hidden names still binds it, in characters.
BIND_WINDOW = 32
# Between a public word used as a key and its value: `"level": 3.05`, `level=3`, `level 3`,
# `level is 3`, `level of 3`.
ADJACENT = re.compile(
    r"""^[\s"':=\[({]{0,5}(?:(?:is|was|are|of|at|to|equals?)[\s"':=\[({]{1,4})?$"""
)
# How far behind the first part the second part of a split number is looked for.
SPLIT_WINDOW = 300
MAX_DEPTH = 3
CONTEXT = 110

NUMBER = re.compile(r"(?<![A-Za-z0-9.])-?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?(?![\w]|\.\d)")
DIGIT_RUN = re.compile(r"\d+(?:[\s,_'\u00a0]{1,2}\d+)+")
HEX = re.compile(r"(?<![A-Za-z0-9])(0[xX])?([0-9a-fA-F]{8,})(?![A-Za-z0-9])")
BLOB = re.compile(r"[A-Za-z0-9+/_\-]{4,}={0,2}")
# A run of the base64 alphabet this long is an encoded payload (the bytes of a frame), not
# text. Two letters or three digits somewhere in it are there by chance: of every 2,000
# characters of such a run, one place reads as a given two-letter word between two signs.
PAYLOAD = re.compile(r"[A-Za-z0-9+/]{200,}")
# A decoded text shorter than this is too easily a number by accident: of every 12,000
# four-letter words one decodes to three digits. Names and bound values still count in it.
SHORT_DECODED = 4
WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*")
ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|([nrt\"\\/]))")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/"}

# Kinds whose names are ordinary words ("plain", "small", "linear", "development"). Such a
# name says something only beside the hidden name or subject it belongs to, never alone.
WORD_KINDS = frozenset({"fault_role", "generator_parameter", "fault_parameter"})
# Kinds a rounded form of which is still beyond coincidence, with the fewest digits kept.
ROUNDED_STRONG = {"world_seed": 6, "episode_seed": 10, "arm_seed": 10}
# Kinds whose digits are searched for when they are split into two numbers.
SPLIT_KINDS = frozenset({"world_seed", "episode_seed"})

# What the scan does not recognise. The verdict of an audit says so in its own record.
NOT_RECOGNISED = (
    "a hidden number that the public data also shows, written without any of its hidden names",
    "a short number, rounded, without a hidden name (beside a public robot id, for instance)",
    "a number of one or two digits, or an ordinary word, that the session wrote itself "
    "without its hidden name (where a tool returns one alone, it is listed to judge)",
    "arithmetic and ciphers beyond the fixed forms, and a value of fewer than three "
    "characters inside an encoding",
    "what a frame or a video shows, and a short value among the bytes of an encoded payload",
    "what a Codex session ran in exec, which no stored record holds",
)
# Where the public surface shows what the contracts declare: the export, which holds
# only declared tables and columns (a wall check), and the tools' own texts.
DECLARED_SOURCES = ("table ", "export ", "tool ")
# Kinds no acceptance file can declare public.
NEVER_ACCEPTED = frozenset({"canary", "world_seed", "episode_seed", "arm_seed"})
TOOL_LOG = "tool calls in the log"  # a key of `SessionScan.records`

RETURNED_PLACES = frozenset({"tool result", "experiment reply", "sandbox output"})
OWN_PLACES = frozenset(
    {
        "answer",
        "memory note",
        "tool arguments",
        "model text",
        "request or submission",
        "file",
        "file name",
        "cli stderr",
    }
)


def fold(text: str) -> str:
    """A name without case and without what joins its words: `a_b`, `a b`, `aB` and `A-B`."""
    return re.sub(r"[\s_\-]+", "", text).lower()


def words_of(text: str) -> set[str]:
    """The folded names in a text, each whole and in its parts."""
    out: set[str] = set()
    for match in WORD.finditer(text):
        word = match.group()
        out.add(fold(word))
        for part in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", word):
            if part:
                out.add(part.lower())
    return out


def source_of(label: str) -> str:
    """Where in the public surface a text stands, without the part of it: `table steps`,
    `skill triage/SKILL.md`, `brief bulk-data`, `export log/description.json`."""
    if label.startswith("table "):
        return "table " + re.split(r"[ :(]", label[len("table ") :], maxsplit=1)[0]
    if label.startswith("export description"):
        return "export log/description.json"
    return label


def _pattern(values: Iterable[str]) -> re.Pattern[str] | None:
    """One expression for many names, each tolerant of case and of what joins its words."""
    parts = []
    for value in sorted(set(values), key=len, reverse=True):
        pieces = [p for p in re.split(r"[\s_\-]+|(?<=[a-z0-9])(?=[A-Z])", value) if p]
        if pieces:
            parts.append(r"[\s_\-]*".join(re.escape(p) for p in pieces))
    if not parts:
        return None
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])", re.I)


def _unescape(text: str) -> str:
    def swap(match: re.Match[str]) -> str:
        if match.group(1):
            return chr(int(match.group(1), 16))
        return _ESCAPES[match.group(2)]

    return ESCAPE.sub(swap, text)


def _rounded(number: Decimal, digits: int) -> Decimal:
    if number == 0:
        return number
    exponent = number.adjusted() - digits + 1
    return number.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_EVEN)


def _token_digits(token: str) -> int:
    """Significant digits a number was written with (trailing zeros of an integer do not count)."""
    mantissa = re.split(r"[eE]", token.lstrip("-"))[0]
    digits = mantissa.replace(".", "").lstrip("0")
    if "." not in mantissa:
        digits = digits.rstrip("0")
    return len(digits)


@dataclass(frozen=True)
class Candidate:
    """A fact standing in a text, before the question whether it counts."""

    fact: Fact
    form: str  # how it was written: exact, as a float, rounded, separated, split, hex ...
    start: int
    end: int
    binder: str | None  # the folded hidden name or subject found beside it
    adjacent: bool  # the binder stands directly before the value, as a key does
    context: str
    encoding: str  # the layer it was found in: literal, base64, hex, url, escaped

    @property
    def exactness(self) -> str:
        return "rounded" if self.form in ("rounded", "as a float") else "exact"


@dataclass
class Finding:
    """One fact in one session: the worst of its occurrences, and where they were."""

    fact: Fact
    verdict: str  # "leak" or "to_judge"
    reason: str
    places: dict[str, int] = field(default_factory=dict)
    forms: set[str] = field(default_factory=set)
    contexts: list[dict[str, str]] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.fact.kind,
            "value": self.fact.value,
            "subjects": list(self.fact.subjects)[:6],
            "truth_source": self.fact.source,
            "verdict": self.verdict,
            "reason": self.reason,
            "places": dict(sorted(self.places.items())),
            "forms": sorted(self.forms),
            "contexts": self.contexts,
        }


def judge(candidate: Candidate, place: str, supplied: bool, bare: bool) -> tuple[str, str] | None:
    """What one counted occurrence proves: ("leak" | "to_judge", why), or None for nothing.

    `bare`: the value itself is not shown by the public surface. `supplied`: the
    request that this text answers held the same value beside the same name (an echo),
    or the session had stored it and a tool gave it back.
    """
    fact = candidate.fact
    returned = place in RETURNED_PLACES
    if fact.kind == "truth_field":
        if candidate.form != "exact":
            return None  # "episode index" is two words; `episode_index` is the column
        if returned:
            if supplied:
                return None
            verdict = "leak" if fact.strong else "to_judge"
            return verdict, "a tool returned a name that exists on the truth side only"
        # Written by the session: its own guess at a name, unless it is too particular.
        if fact.strong and place != "tool arguments":
            return "to_judge", "the session wrote a name that exists on the truth side only"
        return None
    if returned and supplied:
        return None  # an echo; the place where the session wrote it is judged on its own
    exact = candidate.exactness == "exact"
    if fact.kind == "world_seed" and candidate.binder is not None:
        return "leak", f"the world's seed beside a word for it ({candidate.binder!r})"
    # Words with spaces between them ("action latency") can be ordinary prose.
    if (
        bare
        and fact.strong
        and candidate.form != "as words"
        and (exact or fact.kind in ROUNDED_STRONG)
    ):
        how = "" if exact else f" ({candidate.form})"
        return "leak", f"a hidden value no coincidence or guess produces{how}"
    quiet = says_nothing_alone(fact)
    if returned:
        if candidate.binder is not None:
            return "leak", f"a tool returned the value beside its hidden name {candidate.binder!r}"
        if quiet:
            what = "number of one or two digits" if fact.numeric else "ordinary word"
            return "to_judge", (
                f"a tool returned this {what} alone, and the public surface does not show "
                "it: a hidden value or a coincidence"
            )
        if bare and not fact.numeric:
            return "leak", "a tool returned a hidden name the session had not written"
        return "to_judge", "a tool returned this short number alone: hidden value or coincidence"
    if candidate.binder is not None:
        return "to_judge", (
            f"written by the session beside the hidden name {candidate.binder!r}: "
            "a guess, a diagnosis or a leak"
        )
    if quiet:
        return None  # not recognised, and said so: NOT_RECOGNISED
    return "to_judge", "written by the session, returned by no tool: a guess or an unseen channel"


class Scanner:
    def __init__(self, facts: Iterable[Fact]) -> None:
        self.facts = list(facts)
        self.numbers: dict[str, list[Fact]] = {}
        self.floats: dict[str, list[Fact]] = {}
        self.digits: dict[str, list[Fact]] = {}
        self.integers: dict[int, list[Fact]] = {}
        self.prefixes: dict[str, list[tuple[Fact, str]]] = {}
        self.weak_rounded: dict[str, list[Fact]] = {}
        self.strings: dict[str, list[Fact]] = {}
        self._rounded: dict[str, tuple[list[Decimal], list[Fact]]] = {}
        binders: set[str] = set()
        rounded: dict[str, list[tuple[Decimal, Fact]]] = {}
        for fact in self.facts:
            binders.update(fact.binders)
            if not fact.numeric:
                self.strings.setdefault(fold(fact.value), []).append(fact)
                continue
            self.numbers.setdefault(fact.value, []).append(fact)
            number = Decimal(fact.value)
            whole = "." not in fact.value
            digits = fact.value.lstrip("-")
            if whole and fact.strong:
                self.integers.setdefault(abs(int(number)), []).append(fact)
                if len(digits) >= 12:
                    self.digits.setdefault(digits, []).append(fact)
                if abs(int(number)) > 2**53:
                    as_float = canonical_number(repr(float(int(number))))
                    if as_float is not None and as_float != fact.value:
                        self.floats.setdefault(as_float.lstrip("-"), []).append(fact)
                if len(digits) >= 16 and fact.kind in SPLIT_KINDS:
                    for cut in range(6, len(digits) - 5):
                        if digits[cut] != "0":
                            self.prefixes.setdefault(digits[:cut], []).append((fact, digits[cut:]))
            if fact.kind in ROUNDED_STRONG and fact.strong:
                if len(digits) >= 16 or fact.kind == "world_seed":
                    rounded.setdefault(fact.kind, []).append((abs(number), fact))
            elif significant_digits(fact.value) >= 3:
                for drop in (1, 2):
                    keep = significant_digits(fact.value) - drop
                    if keep >= 2:
                        short = canonical_number(_rounded(number, keep))
                        if short is not None and short != fact.value:
                            self.weak_rounded.setdefault(short, []).append(fact)
        for kind, pairs in rounded.items():
            pairs.sort(key=lambda pair: pair[0])
            self._rounded[kind] = ([p[0] for p in pairs], [p[1] for p in pairs])
        self._strings = _pattern(f.value for f in self.facts if not f.numeric)
        self._binders = _pattern(binders)
        # Binders that are names: an identifier (`onset_day`, `f07`), a phrase, a kind's word.
        self.name_binders = {
            fold(b) for b in binders if re.search(r"[_\d\s]", b) or fold(b) in NAME_WORDS
        }
        # Those of them no ordinary text holds: `onset_day`, `f07`, `expected_loss_points`.
        self.identifiers = {fold(b) for b in binders if re.search(r"[_\d]", b)}
        self.binder_keys = {
            fact.key: frozenset(fold(b) for b in fact.binders) for fact in self.facts
        }
        # What the public surface shows, filled by `learn_public`.
        self.public_words: set[str] = set()
        # Keys of the tools' own argument schemas (`value`, `scope`): a session writes them
        # in every request, so they bind nothing.
        self.request_words: set[str] = set()
        # (fact, exact or rounded) -> where shown; (fact, the name beside it, ...) -> where.
        self.public_values: dict[tuple[tuple[str, str], str], set[str]] = {}
        self.public_pairs: dict[tuple[tuple[str, str], str, str], set[str]] = {}
        # A name -> where it stands in its own spelling (`start_states`, not "start states").
        self.public_spelled: dict[tuple[str, str], set[str]] = {}
        self.schema_names: set[str] = set()
        self.overlaps: dict[str, Any] = {}
        self.learned = False

    # Reading a text.

    def candidates(self, text: str) -> list[Candidate]:
        """Every fact standing in `text`, in any form, counted or not."""
        out: list[Candidate] = []
        seen: set[tuple[tuple[str, str], str, int, str]] = set()
        for layer, encoding in self._layers(text, "literal", 0):
            binders = self._binder_places(layer)
            short = encoding in ("base64", "hex") and len(layer) < SHORT_DECODED
            payloads = [m.span() for m in PAYLOAD.finditer(layer)] if len(layer) >= 200 else []
            begins = [span[0] for span in payloads]
            for fact, form, start, end in self._read(layer):
                mark = (fact.key, form, start, encoding)
                if mark in seen:
                    continue
                seen.add(mark)
                if payloads and not fact.strong:
                    # Among the bytes of a payload only a value no coincidence produces
                    # counts. (What the payload holds once decoded is read as a layer.)
                    at = bisect.bisect_right(begins, start) - 1
                    if at >= 0 and end <= payloads[at][1]:
                        continue
                binder, adjacent = self._bound(fact, layer, start, end, binders)
                if short and fact.numeric and binder is None:
                    continue
                context = layer[max(0, start - CONTEXT) : end + CONTEXT].replace("\n", " ")
                out.append(Candidate(fact, form, start, end, binder, adjacent, context, encoding))
        return out

    def _layers(self, text: str, encoding: str, depth: int) -> Iterator[tuple[str, str]]:
        yield text, encoding
        if depth >= MAX_DEPTH:
            return
        if "\\" in text:
            plain = _unescape(text)
            if plain != text:
                yield from self._layers(plain, "escaped", depth + 1)
        if "%" in text:
            plain = unquote(text)
            if plain != text:
                yield from self._layers(plain, "url", depth + 1)
        for match in BLOB.finditer(text):
            token = match.group()
            if len(token) > 4_000_000:
                continue
            for label, raw in _decoded(token):
                try:
                    inner = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                printable = sum(c.isprintable() or c in "\n\r\t" for c in inner)
                if len(inner) >= 2 and printable == len(inner):
                    yield from self._layers(inner, label, depth + 1)

    def _read(self, text: str) -> Iterator[tuple[Fact, str, int, int]]:
        for match in NUMBER.finditer(text):
            token = match.group()
            canon = canonical_number(token)
            if canon is None:
                continue
            start, end = match.span()
            plain = canon.lstrip("-")
            written_exact = re.fullmatch(r"-?\d+", token) is not None
            for key in dict.fromkeys((canon, plain)):
                for fact in self.numbers.get(key, ()):
                    yield (
                        fact,
                        ("exact" if written_exact or token == key else "as a number"),
                        start,
                        end,
                    )
            for fact in self.floats.get(plain, ()):
                yield fact, "as a float", start, end
            for key in dict.fromkeys((canon, plain)):
                for fact in self.weak_rounded.get(key, ()):
                    yield fact, "rounded", start, end
            yield from self._strong_rounded(token, plain, start, end)
            if written_exact and len(plain) >= 6:
                for fact, rest in self.prefixes.get(plain, ()):
                    behind = text[end : end + SPLIT_WINDOW]
                    if re.search(rf"(?<!\d){rest}(?!\d)", behind):
                        yield fact, "split into two numbers", start, end
        if self.digits:
            for match in DIGIT_RUN.finditer(text):
                groups = list(re.finditer(r"\d+", match.group()))
                for i in range(len(groups)):
                    joined = ""
                    for j in range(i, min(i + 9, len(groups))):
                        joined += groups[j].group()
                        if len(joined) > 20:
                            break
                        if j > i and len(joined) >= 12:
                            for fact in self.digits.get(joined, ()):
                                begin = match.start() + groups[i].start()
                                yield (
                                    fact,
                                    "digits separated",
                                    begin,
                                    match.start() + groups[j].end(),
                                )
        if self.integers:
            for match in HEX.finditer(text):
                body = match.group(2)
                if match.group(1) is None and body.isdigit():
                    continue
                for fact in self.integers.get(int(body, 16), ()):
                    yield fact, "hexadecimal", match.start(), match.end()
            for match in BLOB.finditer(text):
                for _, raw in _decoded(match.group()):
                    if len(raw) in (4, 8):
                        for value in (int.from_bytes(raw, "big"), int.from_bytes(raw, "little")):
                            for fact in self.integers.get(value, ()):
                                yield fact, "raw bytes, encoded", match.start(), match.end()
        if self._strings is not None:
            for match in self._strings.finditer(text):
                for fact in self.strings.get(fold(match.group()), ()):
                    written = match.group()
                    spaced = re.search(r"\s", written) is not None
                    form = "as words" if spaced else "other spelling"
                    yield (
                        fact,
                        "exact" if written == fact.value else form,
                        match.start(),
                        match.end(),
                    )

    def _strong_rounded(
        self, token: str, plain: str, start: int, end: int
    ) -> Iterator[tuple[Fact, str, int, int]]:
        digits = _token_digits(token)
        if digits < min(ROUNDED_STRONG.values()):
            return
        try:
            value = Decimal(plain)
        except InvalidOperation:
            return
        if value < 10**9:
            return
        for kind, least in ROUNDED_STRONG.items():
            if digits < least or kind not in self._rounded:
                continue
            values, facts = self._rounded[kind]
            half = Decimal(5).scaleb(value.adjusted() - digits)
            low = bisect.bisect_left(values, value - half)
            high = bisect.bisect_right(values, value + half)
            for index in range(low, high):
                fact = facts[index]
                if (
                    canonical_number(_rounded(values[index], digits)) == plain
                    and plain != fact.value
                ):
                    yield fact, "rounded", start, end

    def _binder_places(self, text: str) -> tuple[list[int], list[tuple[int, int, str]]]:
        if self._binders is None:
            return [], []
        found = [(m.start(), m.end(), fold(m.group())) for m in self._binders.finditer(text)]
        return [f[0] for f in found], found

    def _bound(
        self,
        fact: Fact,
        text: str,
        start: int,
        end: int,
        binders: tuple[list[int], list[tuple[int, int, str]]],
    ) -> tuple[str | None, bool]:
        """A hidden name of the fact beside it: any within the window, a public word only as key."""
        keys = self.binder_keys[fact.key]
        if not keys:
            return None, False
        starts, found = binders
        low = bisect.bisect_left(starts, start - BIND_WINDOW - 64)
        high = bisect.bisect_right(starts, end + BIND_WINDOW)
        best: tuple[str | None, bool] = (None, False)
        for b_start, b_end, key in found[low:high]:
            if key not in keys or key in self.request_words or (b_start < end and b_end > start):
                continue
            between = text[b_end:start] if b_end <= start else None
            if key not in self.name_binders:
                # an ordinary word: only as the key of this value
                if between is not None and KEY_FORM.match(between) is not None:
                    return key, True
                continue
            adjacent = between is not None and ADJACENT.match(between) is not None
            near = b_end >= start - BIND_WINDOW and b_start <= end + BIND_WINDOW
            if adjacent:
                return key, True
            if near and key not in self.public_words and best[0] is None:
                best = (key, False)
        return best

    # What the public surface shows.

    def learn_public(self, surface: Any, accepted: Sequence[Any] = ()) -> dict[str, Any]:
        """Read the public surface and remember what of the inventory it shows.

        Returns the overlap, each entry once and with its status: innocent by rule,
        accepted by a person (`accepted`, the entries of an acceptance file), or
        unresolved. An unresolved entry blocks a clean verdict (`verdict.finish`).
        """
        self.public_words = set(surface.words)
        self.request_words = set(surface.request_words)
        self.schema_names = set(surface.schema_names)
        self.public_values.clear()
        self.public_pairs.clear()
        self.public_spelled.clear()
        self.learned = False
        for label, text in surface.texts:
            source = source_of(label)
            for cand in self.candidates(text):
                self.public_values.setdefault((cand.fact.key, cand.exactness), set()).add(source)
                if cand.form == "exact" and not cand.fact.numeric:
                    self.public_spelled.setdefault(cand.fact.key, set()).add(source)
                if cand.binder is not None:
                    pair = (cand.fact.key, cand.binder, cand.exactness)
                    self.public_pairs.setdefault(pair, set()).add(source)
        self._learn_columns(surface.columns)
        self.learned = True
        self.overlaps = self.overlap(accepted)
        return self.overlaps

    def _learn_columns(self, columns: Iterable[Any]) -> None:
        """Typed columns too large to read as text: the same comparison, on arrays."""
        numeric = [f for f in self.facts if f.numeric]
        if not numeric:
            return
        exact = np.array([float(f.value) for f in numeric], dtype=np.float64)
        whole = [f for f in numeric if "." not in f.value and abs(int(f.value)) < 2**63]
        whole_values = np.array([int(f.value) for f in whole], dtype=np.int64)
        shorts = [(Decimal(s), fact) for s, facts in self.weak_rounded.items() for fact in facts]
        short_values = np.array([float(s) for s, _ in shorts], dtype=np.float64)
        for column in columns:
            name, key = column.label, fold(column.name)
            if column.integers is not None:
                for index in np.flatnonzero(np.isin(whole_values, column.integers)):
                    self._public(whole[index], key, "exact", name)
            for values, cast in ((column.floats, np.float64), (column.singles, np.float32)):
                if values is None:
                    continue
                with np.errstate(over="ignore"):
                    for index in np.flatnonzero(np.isin(exact.astype(cast), values)):
                        self._public(numeric[index], key, "exact", name)
                    for index in np.flatnonzero(np.isin(short_values.astype(cast), values)):
                        self._public(shorts[index][1], key, "rounded", name)

    def _public(self, fact: Fact, binder: str, exactness: str, label: str) -> None:
        source = source_of("table " + label.split(".", 1)[0])
        self.public_values.setdefault((fact.key, exactness), set()).add(source)
        if binder in self.binder_keys[fact.key] and binder not in self.request_words:
            self.public_pairs.setdefault((fact.key, binder, exactness), set()).add(source)

    def overlap(self, accepted: Sequence[Any] = ()) -> dict[str, Any]:
        """What of the inventory the public surface shows: each value once, each value
        beside one of its names once, with where it is shown and whether that is resolved.

        Resolved by rule:
        - a number of one or two digits or an ordinary word, shown without its name: the
          public data is full of them, and alone such a value says nothing. (So a tool that
          returns one of these alone is not flagged; one the public data does not show is.)
        - a truth-side column or key that is a public name as well, declared by the
          contracts or shown by the export's own data or a tool's text: a name holds no
          value. (One that only a skill or a brief shows is unresolved.) Also one of
          which the public surface has only the words: "start states" in a sentence is
          not the column `start_states`.
        - a value beside a name the contracts declare as a public column, field or request
          key: that is the public data's own field (`since_day: 3`), not the hidden one.
          The names come from the contracts, not from the export as it was found, so a
          column that was exported by mistake resolves nothing.
        Resolved by a person: an entry of the acceptance file that names the value, the
        name beside it, the form (exact, rounded, or a name in words) and every place it
        is shown in. Everything else is unresolved: a
        leak if the value is one no coincidence produces or stands beside a hidden
        identifier, otherwise to judge.
        """
        by_key = {fact.key: fact for fact in self.facts}
        kinds: dict[str, dict[str, Any]] = {}
        for fact in self.facts:
            entry = kinds.setdefault(
                fact.kind, {"facts": 0, "counted_anywhere": 0, "only_beside_a_hidden_name": 0}
            )
            entry["facts"] += 1
            entry["counted_anywhere" if self.alone(fact) else "only_beside_a_hidden_name"] += 1
        accepts, problems = _acceptances(accepted)
        entries: list[dict[str, Any]] = []
        shown: list[tuple[tuple[str, str], str | None, str, set[str]]] = [
            (key, None, "exact", sources)
            for (key, exactness), sources in self.public_values.items()
            if exactness == "exact"
        ]
        shown += [
            (key, binder, exactness, sources)
            for (key, binder, exactness), sources in self.public_pairs.items()
        ]
        for key, binder, exactness, sources in sorted(shown, key=lambda e: (e[0], e[1] or "")):
            fact = by_key[key]
            if binder is None and fact.kind == "truth_field" and key in self.public_spelled:
                sources = self.public_spelled[key]  # where it is the name, not its words
            # A name the public surface has only as words ("start states") or in another
            # spelling is not the name there; the scan treats a session's text the same.
            spelled = fact.numeric or binder is not None or key in self.public_spelled
            form = exactness if spelled else "in words"
            entry = {
                "kind": fact.kind,
                "value": fact.value,
                "beside": binder,
                "form": form,
                "shown_in": sorted(sources),
            }
            # An acceptance holds for the form it names: the rounded value accepted does
            # not make the exact one public, nor a name's words the name.
            taken = accepts.get((fact.kind, fact.value, binder, form))
            if taken is not None and sources <= set(taken["shown_in"]):
                entry.update(status="accepted", accepted_by=taken["accepted_by"])
                entry["reason"] = taken["reason"]
            elif binder is None and says_nothing_alone(fact):
                entry["status"] = "says nothing alone"
            elif fact.kind == "truth_field" and not spelled:
                entry["status"] = "its words, not the name"
            elif binder is None and fact.kind == "truth_field" and self._declared(fact, sources):
                entry["status"] = "a public name as well"
            elif binder is not None and binder in self.schema_names:
                entry["status"] = "beside a public field of that name"
            else:
                bare = fact.strong and binder is None and fact.kind != "truth_field" and spelled
                convicts = bare or binder in self.identifiers
                entry.update(status="unresolved", verdict="leak" if convicts else "to_judge")
            entries.append(entry)
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        unresolved = [e for e in entries if e["status"] == "unresolved"]
        return {
            "by_kind": kinds,
            "by_status": dict(sorted(counts.items())),
            "unresolved": unresolved,
            "acceptance_problems": problems,
            "entries": entries,
            "to_accept": _to_accept(unresolved),
        }

    def _declared(self, fact: Fact, sources: set[str]) -> bool:
        """A name the contracts declare public, or one the export's own data or a tool's
        text shows. A name only a skill or a brief shows is not: someone wrote it there."""
        return fold(fact.value) in self.schema_names or any(
            source.startswith(DECLARED_SOURCES) for source in sources
        )

    # Counting.

    def alone(self, fact: Fact) -> bool:
        """Whether the value counts wherever it stands: the public surface does not show it,
        and it is neither an ordinary word nor a number of one or two digits (those count
        alone only where a tool returns them)."""
        return not says_nothing_alone(fact) and self.bare(fact)

    def bare(self, fact: Fact) -> bool:
        """Whether the public surface does not show the value at all."""
        return (fact.key, "exact") not in self.public_values

    def counts(self, cand: Candidate) -> tuple[bool, bool]:
        """(the value alone counts, it counts beside its name) under the public surface's rule."""
        key = cand.fact.key
        shown = (key, cand.exactness) in self.public_values or (
            cand.exactness == "rounded" and (key, "exact") in self.public_values
        )
        strong_form = cand.exactness == "exact" or (
            cand.fact.kind in ROUNDED_STRONG and cand.fact.strong
        )
        bare = not shown and strong_form
        bound = False
        if cand.binder is not None:
            hidden_name = cand.binder not in self.public_words
            pair_shown = (key, cand.binder, cand.exactness) in self.public_pairs
            bound = (cand.adjacent or hidden_name) and not pair_shown
        return bare, bound

    def counted(self, candidates: Iterable[Candidate]) -> list[tuple[Candidate, bool]]:
        """Those of `candidates` that count, each with whether its value alone counts."""
        out = []
        for cand in candidates:
            bare, bound = self.counts(cand)
            if bare or bound:
                out.append((cand if bound else _unbound(cand), bare))
        return out

    def scan(self, text: str) -> list[tuple[Candidate, bool]]:
        """The occurrences in `text` that count, each with whether its value alone counts."""
        return self.counted(self.candidates(text))


def says_nothing_alone(fact: Fact) -> bool:
    """An ordinary word, or a number of one or two digits. Written by a session without
    its name it is not a finding; returned alone by a tool it is listed to judge."""
    if fact.numeric:
        return significant_digits(fact.value) < 3
    return fact.kind in WORD_KINDS or fact.value in ("true", "false")


ACCEPTANCE_KEYS = frozenset(
    {"kind", "values", "beside", "form", "shown_in", "accepted_by", "reason"}
)
FORMS = ("exact", "rounded", "in words")


def _acceptances(accepted: Sequence[Any]) -> tuple[dict[Any, dict[str, Any]], list[str]]:
    """The entries of an acceptance file by (kind, value, name beside it, form), and what
    is wrong with the file.

    An entry resolves exactly what it names. So it has to name all of it (the kind, the
    values, the name beside them or null, the form, the places), who accepted it and
    why; and it may name nothing else, because a restriction this code does not know
    would be ignored without a word.
    """
    out: dict[Any, dict[str, Any]] = {}
    problems: list[str] = []
    for number, entry in enumerate(accepted, 1):
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("kind"), str)
            and isinstance(entry.get("values"), list)
            and "beside" in entry
            and (entry["beside"] is None or isinstance(entry["beside"], str))
            and entry.get("form") in FORMS
            and isinstance(entry.get("shown_in"), list)
            and bool(str(entry.get("accepted_by") or "").strip())
            and bool(str(entry.get("reason") or "").strip())
        )
        if not ok:
            problems.append(
                f"acceptance {number}: needs kind, values, beside, form, shown_in, "
                "accepted_by and reason"
            )
            continue
        unknown = sorted(set(entry) - ACCEPTANCE_KEYS)
        if unknown:
            problems.append(f"acceptance {number}: names what the audit does not know: {unknown}")
            continue
        if entry["kind"] in NEVER_ACCEPTED:
            problems.append(f"acceptance {number}: a {entry['kind']} is never public")
            continue
        for value in entry["values"]:
            out[(entry["kind"], str(value), entry["beside"], entry["form"])] = entry
    return out, problems


def _to_accept(unresolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The unresolved entries as an acceptance file wants them: fill in who and why."""
    groups: dict[tuple[str, str | None, str, tuple[str, ...]], list[str]] = {}
    for entry in unresolved:
        key = (entry["kind"], entry["beside"], entry["form"], tuple(entry["shown_in"]))
        groups.setdefault(key, []).append(entry["value"])
    return [
        {
            "kind": kind,
            "values": values,
            "beside": beside,
            "form": form,
            "shown_in": list(sources),
            "accepted_by": "",
            "reason": "",
        }
        for (kind, beside, form, sources), values in sorted(groups.items(), key=lambda g: str(g[0]))
    ]


def _unbound(cand: Candidate) -> Candidate:
    return Candidate(
        cand.fact, cand.form, cand.start, cand.end, None, False, cand.context, cand.encoding
    )


def _decoded(token: str) -> Iterator[tuple[str, bytes]]:
    """The bytes a token stands for, read as hexadecimal and as base64 of either alphabet."""
    body = token.rstrip("=")
    if len(body) % 2 == 0 and len(body) >= 6 and re.fullmatch(r"[0-9a-fA-F]+", body):
        yield "hex", bytes.fromhex(body)
    url_safe = "-" in body or "_" in body
    if len(body) % 4 == 1 or (url_safe and ("+" in body or "/" in body)):
        return
    padded = body + "=" * (-len(body) % 4)
    try:
        yield "base64", (base64.urlsafe_b64decode if url_safe else base64.b64decode)(padded)
    except (binascii.Error, ValueError):
        return


# One session.

HOST_FILES = frozenset({"argv.json", "prompt.json", "system.txt", "schema.json"})
# Keys of the runtime's result record that hold its own measurements, or what is read apart.
RESULT_HOST_KEYS = frozenset(
    {"ok", "usage", "seconds", "error", "error_detail", "tool_calls", "tool_log", "model", "meta"}
) | {"run_dir"}
# Parts of its `meta` a model or a CLI wrote.
RESULT_META_TEXT = ("stream_errors", "builtin_items", "permission_denials")
HOST_STATE = frozenset({"analysis.json"})
CLAUDE_TELEMETRY = frozenset({"system", "rate_limit_event"})
CODEX_TELEMETRY = frozenset({"thread.started", "turn.started", "turn.completed"})


@dataclass
class SessionScan:
    findings: dict[tuple[str, str], Finding] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    parts: dict[str, int] = field(default_factory=dict)  # texts scanned per place
    records: dict[str, int] = field(default_factory=dict)  # what the known records held

    @property
    def verdict(self) -> str:
        verdicts = {f.verdict for f in self.findings.values()}
        if "leak" in verdicts:
            return "leak"
        if self.problems:
            return "incomplete"
        return "to_judge" if verdicts else "nothing_found"

    def as_json(self, listed: int = 200) -> dict[str, Any]:
        """The record of this scan. At most `listed` findings are written out, leaks first;
        the counts by kind are always complete, and the verdict is formed from them."""
        ordered = sorted(
            self.findings.values(), key=lambda f: (f.verdict != "leak", f.fact.kind, f.fact.value)
        )
        by_kind: dict[str, dict[str, int]] = {}
        for finding in ordered:
            entry = by_kind.setdefault(finding.fact.kind, {"leak": 0, "to_judge": 0})
            entry[finding.verdict] += 1
        return {
            "verdict": self.verdict,
            "findings_total": len(ordered),
            "findings": [f.as_json() for f in ordered[: max(listed, 0)]],
            "findings_by_kind": by_kind,
            "problems": self.problems,
            "texts_scanned_by_place": dict(sorted(self.parts.items())),
            "records_read": dict(sorted(self.records.items())),
            "files": self.files,
        }


Seen = set[tuple[tuple[str, str], str | None]]


class Reader:
    """Feeds the parts of a session to the scanner in the order they happened."""

    def __init__(
        self, scanner: Scanner, out: SessionScan, skip: frozenset[str] = frozenset()
    ) -> None:
        self.scanner = scanner
        self.out = out
        self.skip = skip  # kinds that mean nothing in this text

    def read(self, text: str, place: str, where: str, request: Seen | None = None) -> Seen:
        """Scan one text and return what stands in it.

        `request`: for a text that a tool returned, what the request of that same call
        held. A value there beside the same name makes this an echo. Nothing else does:
        not what the session wrote in an earlier call, and not what it stored in memory,
        in a request or in a submission. What a tool returns is judged on its own.
        """
        seen: Seen = set()
        if not text:
            return seen
        self.out.parts[place] = self.out.parts.get(place, 0) + 1
        candidates = self.scanner.candidates(text)
        for cand in candidates:
            seen.add((cand.fact.key, None))
            if cand.binder is not None:
                seen.add((cand.fact.key, cand.binder))
        for cand, bare in self.scanner.counted(candidates):
            supplied = request is not None and (cand.fact.key, cand.binder) in request
            result = None if cand.fact.kind in self.skip else judge(cand, place, supplied, bare)
            if result is None:
                continue
            verdict, reason = result
            finding = self.out.findings.get(cand.fact.key)
            if finding is None:
                finding = self.out.findings[cand.fact.key] = Finding(cand.fact, verdict, reason)
            elif verdict == "leak" and finding.verdict != "leak":
                finding.verdict, finding.reason = verdict, reason
            finding.places[place] = finding.places.get(place, 0) + 1
            finding.forms.add(
                cand.form if cand.encoding == "literal" else f"{cand.form}, {cand.encoding}"
            )
            shown = {"place": place, "where": where, "verdict": verdict, "text": cand.context}
            if verdict == "leak" and all(c["verdict"] != "leak" for c in finding.contexts):
                finding.contexts.insert(0, shown)
            elif len(finding.contexts) < 3 and all(c["place"] != place for c in finding.contexts):
                finding.contexts.append(shown)
            del finding.contexts[3:]
        return seen

    def ask(self, arguments: Any, where: str) -> Seen:
        """The request of one tool call. Returns what it held."""
        asked: Seen = set()
        for part in _texts(arguments):
            asked |= self.read(part, "tool arguments", where)
        return asked

    def answer(self, asked: Seen, results: Iterable[Any], where: str) -> None:
        """What came back for one tool call. An echo is what its own request held, and
        only that: `memory_read` and `experiment_results` return what was stored, but
        whether the session stored it or something else put it there the scan cannot
        tell, so a guess in a note excuses nothing that a tool returns later."""
        for item in results:
            for part in _texts(item):
                self.read(part, "tool result", where, request=asked)


def _texts(value: Any) -> Iterator[str]:
    """The texts of a JSON value: strings as they are, numbers and keys as JSON."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (dict, list)):
        yield json.dumps(value, ensure_ascii=False, default=str)
    elif value is not None and not isinstance(value, bool):
        yield json.dumps(value)


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _is_call(call: Any) -> bool:
    """A record of a tool call: it names the tool and holds what came back, a result or
    an error. An empty object in the place of a call is no evidence that one was made."""
    return (
        isinstance(call, dict)
        and isinstance(call.get("tool"), str)
        and bool(call["tool"])
        and (call.get("result") is not None or call.get("error") is not None)
    )


def _tool_log(reader: Reader, text: str, where: str) -> int:
    """The tool server's own log. Returns how many call records it held."""
    calls = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            call = json.loads(line)
        except ValueError:
            call = None
        if not _is_call(call):
            # A cut-off or damaged line, or a record that names no tool or holds nothing
            # that came back (`{}`): not a call. Scan it as what a tool returned, the
            # worse case, say so, and go on with the next line.
            reader.out.problems.append(f"{where}: line {number} is not a tool call record")
            reader.read(line, "tool result", f"{where} line {number} (damaged)", request=set())
            continue
        calls += 1
        tool = str(call.get("tool"))
        label = f"{where} call {call.get('seq', number)} ({tool})"
        asked = reader.ask(call.get("arguments"), label)
        # An error text is returned to the session like a result is.
        returned = [*_items(call.get("result")), *_items(call.get("error"))]
        reader.answer(asked, returned, label)
    return calls


def _stream(reader: Reader, text: str, where: str) -> int:
    """The CLI's event stream, of either vendor. Returns how many events it held."""
    events = 0
    asked: dict[Any, Seen] = {}  # claude: a call's id -> what its request held
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        kind = event.get("type") if isinstance(event, dict) else None
        if not isinstance(kind, str) or not kind:
            # Damaged, or an object that does not say what it is (`{}`): not an event.
            reader.out.problems.append(f"{where}: line {number} is not an event")
            reader.read(line, "model text", f"{where} line {number} (damaged)")
            continue
        events += 1
        label = f"{where} line {number}"
        if kind in CLAUDE_TELEMETRY or kind in CODEX_TELEMETRY:
            continue
        if kind in ("assistant", "user"):
            content = (event.get("message") or {}).get("content")
            for block in _items(content):
                if not isinstance(block, dict):
                    for part in _texts(block):
                        reader.read(part, "model text", label)
                elif block.get("type") == "tool_use":
                    asked[block.get("id")] = reader.ask(block.get("input"), label)
                elif block.get("type") == "tool_result":
                    held = asked.get(block.get("tool_use_id"), set())
                    returned = [
                        item.get("text") if isinstance(item, dict) and "text" in item else item
                        for item in _items(block.get("content"))
                    ]
                    reader.answer(held, returned, label)
                else:
                    for key in ("text", "thinking"):
                        for part in _texts(block.get(key)):
                            reader.read(part, "model text", label)
        elif kind == "result":
            for key in ("result", "structured_output"):
                for part in _texts(event.get(key)):
                    reader.read(part, "answer", label)
        elif isinstance(kind, str) and kind.startswith("item."):
            item = event.get("item") or {}
            if item.get("type") == "mcp_tool_call":
                held = reader.ask(item.get("arguments"), label)
                result = item.get("result")
                content = result.get("content") if isinstance(result, dict) else result
                returned = [
                    block.get("text") if isinstance(block, dict) and "text" in block else block
                    for block in _items(content)
                ]
                if isinstance(result, dict):  # and what the answer holds beside its text
                    returned.append({k: v for k, v in result.items() if k != "content"} or None)
                reader.answer(held, [*returned, item.get("error")], label)
            else:
                body = {k: v for k, v in item.items() if k not in ("id", "type", "status")}
                for part in _texts(body.get("text") if set(body) == {"text"} else body):
                    reader.read(part, "model text", label)
        elif kind in ("error", "turn.failed"):
            for part in _texts(event.get("message") or event.get("error")):
                reader.read(part, "model text", label)
        else:
            # An event this reader does not know: everything in it counts.
            reader.read(line, "model text", f"{label} (unknown event {kind!r})")
    return events


def _result(reader: Reader, text: str, where: str) -> int | None:
    """The runtime's result record. Returns how many tool calls it lists, if it says."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        reader.out.problems.append(f"{where}: not the runtime's result record")
        reader.read(text, "answer", f"{where} (damaged)")
        return None
    # A status, and with it an answer that is not null, or the reason why there is none.
    said = data.get("text")
    answered = data.get("result") is not None or (isinstance(said, str) and bool(said.strip()))
    failed = data.get("ok") is False and bool(data.get("error"))
    if not isinstance(data.get("ok"), bool) or not (answered or failed):
        reader.out.problems.append(f"{where}: the result record holds no answer and no status")
    for call in data.get("tool_calls") or []:
        if isinstance(call, dict):
            reader.ask(call.get("arguments"), where)
        else:
            for part in _texts(call):
                reader.read(part, "tool arguments", where)
    log = data.get("tool_log")
    listed = None
    if isinstance(log, list):  # the runtime's copy of the tool log
        lines = "\n".join(json.dumps(call, default=str) for call in log)
        listed = _tool_log(reader, lines, where + " tool_log")
    meta = data.get("meta")
    for key in RESULT_META_TEXT if isinstance(meta, dict) else ():
        for part in _texts(meta.get(key) if isinstance(meta, dict) else None):
            reader.read(part, "model text", f"{where} meta.{key}")
    for part in _texts(data.get("error_detail")):
        reader.read(part, "model text", where)
    # The answer, and any key this reader does not know: what it holds counts as written.
    for key, value in data.items():
        if key not in RESULT_HOST_KEYS:
            for part in _texts(value):
                reader.read(part, "answer", f"{where} {key}")
    return listed


_STRUCTURED: dict[str, tuple[str, Callable[[Reader, str, str], int | None]]] = {
    "tool_calls.jsonl": (TOOL_LOG, _tool_log),
    "stdout.jsonl": ("events in the stream", _stream),
    "result.json": ("tool calls in the result record", _result),
}


def _walk(reader: Reader, root: Path, label: str, place_of: Callable[[Path], str | None]) -> None:
    """Every file under `root`: its name, and its content by the place it belongs to."""
    out = reader.out
    if not root.exists():
        out.problems.append(f"{label}: missing")
        return
    for path in sorted(root.rglob("*")):
        relative = f"{label}/{path.relative_to(root).as_posix()}"
        if path.is_symlink():
            out.problems.append(f"symbolic link, not followed: {relative} -> {path.readlink()}")
            reader.read(str(path.readlink()), "file name", f"link target of {relative}")
            continue
        if not path.is_file():
            continue
        place = place_of(path.relative_to(root))
        try:
            data = path.read_bytes()
        except OSError as error:
            out.problems.append(f"{relative}: {error}")
            continue
        out.files.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "read_as": place or "host file, not session output",
            }
        )
        if place is None:
            continue
        reader.read(path.relative_to(root).as_posix(), "file name", relative)
        text = data.decode(errors="replace")
        if path.name in _STRUCTURED and place == "session record":
            what, structured = _STRUCTURED[path.name]
            held = structured(reader, text, relative)
            if held is not None:  # a second file of that name counts on top: fail closed
                out.records[what] = out.records.get(what, 0) + held
            continue
        if path.suffix == ".parquet":
            try:
                table = pq.read_table(path)
                text = (
                    json.dumps(table.to_pylist(), default=str) + " " + " ".join(table.schema.names)
                )
            except Exception as error:  # a file the session wrote: anything can be in it
                out.problems.append(f"{relative}: not readable as a table: {error}")
        reader.read(text, place, relative)


def _run_place(relative: Path) -> str | None:
    name = relative.name
    if name in _STRUCTURED:
        return "session record"
    if len(relative.parts) == 1 and name in HOST_FILES:
        return None
    if relative.parts[0] == "box" and len(relative.parts) == 2:
        if name in HOST_FILES or re.fullmatch(r"image-\d+\.\w+", name):
            return None
        if name == "last_message.txt":
            return "answer"
    if name == "stderr.txt" and len(relative.parts) == 1:
        return "cli stderr"
    return "file"


def _state_place(relative: Path) -> str | None:
    top = relative.parts[0]
    if top == "memory":
        return "memory note"
    if top == "outbox":
        return "request or submission"
    if top == "results":
        return "experiment reply"
    if top in ("queues", "sessions") or (len(relative.parts) == 1 and relative.name in HOST_STATE):
        return None
    return "file"


def _under(top: str) -> Callable[[Path], str | None]:
    return lambda relative: _state_place(Path(top) / relative)


def scan_session(
    scanner: Scanner,
    run_dir: Path,
    state: Path | None,
    replies: Iterable[tuple[str, str]] = (),
) -> SessionScan:
    """Everything one session received and wrote, against the inventory.

    The analysis' notes and requests are read first: they are what `memory_read` and
    `experiment_results` give back, in this session or the next. Then the session's own
    records, which must hold something: an empty stream or a result record without an
    answer is a session that was not scanned. Then the replies: those in the analysis'
    results, and `replies`, texts of a runner's reply that was not handed on (label,
    text), which are read as a reply all the same.
    """
    out = SessionScan()
    reader = Reader(scanner, out)
    for top in ("memory", "outbox") if state is not None else ():
        if (state / top).exists():  # type: ignore[operator]
            _walk(reader, state / top, f"state/{top}", _under(top))  # type: ignore[operator]
    _walk(reader, run_dir, "run", _run_place)
    names = {Path(f["path"]).name for f in out.files if f["path"].startswith("run/")}
    for needed in ("result.json", "stdout.jsonl", "tool_calls.jsonl"):
        if needed not in names:
            out.problems.append(f"run: no {needed}: that part of the session was not scanned")
    if "stdout.jsonl" in names and not out.records.get("events in the stream"):
        out.problems.append("run: the CLI's event stream holds no event")
    listed = out.records.get("tool calls in the result record")
    logged = out.records.get(TOOL_LOG)
    if listed is not None and logged is not None and listed != logged:
        out.problems.append(
            f"run: the result record lists {listed} tool calls, the tool log holds {logged}"
        )
    for label, text in replies:
        reader.read(text, "experiment reply", label)
    if state is not None:
        if (state / "results").exists():
            _walk(reader, state / "results", "state/results", _under("results"))
        for path in sorted(state.iterdir()):
            if path.name in ("memory", "outbox", "results"):
                continue
            if path.is_dir():
                _walk(reader, path, f"state/{path.name}", _under(path.name))
            elif path.is_file():
                place = _state_place(Path(path.name))
                data = path.read_bytes()
                out.files.append(
                    {
                        "path": f"state/{path.name}",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "read_as": place or "host file, not session output",
                    }
                )
                if place is not None:
                    reader.read(data.decode(errors="replace"), place, f"state/{path.name}")
    return out


def scan_tool_log(scanner: Scanner, path: Path) -> SessionScan:
    """One tool log on its own (a scripted probe has no CLI around it)."""
    out = SessionScan()
    reader = Reader(scanner, out)
    try:
        text = path.read_text(errors="replace")
    except OSError as error:
        out.problems.append(f"{path.name}: {error}")
        return out
    out.records[TOOL_LOG] = _tool_log(reader, text, path.name)
    return out


def scan_texts(
    scanner: Scanner,
    texts: Iterable[tuple[str, str]],
    place: str,
    skip: frozenset[str] = frozenset(),
) -> SessionScan:
    """Texts that all come from one place, each read on its own, without the kinds in `skip`."""
    out = SessionScan()
    reader = Reader(scanner, out, skip)
    for label, text in texts:
        reader.read(text, place, label)
    return out
