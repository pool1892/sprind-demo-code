"""Traceability: every number of the answer against the tool results of the call log (plan,
honesty rule 8; docs/design/contracts.md, section 12).
"""

from __future__ import annotations

import bisect
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fleetlab.contracts.tools import Diagnosis, RankedFixSet
from fleetlab.scoring.models import JudgedNumber, TracedNumber, TraceScore

# The calls whose results are sources: the tools that return data about the fleet. Not
# the skills (procedures, whose SQL constants such as 1.96 would trace anything), not the
# submissions and their receipts (their arguments are the answer itself), not the agent's
# own notes, not a request's receipt.
DATA_SOURCES: frozenset[str] = frozenset({"query", "describe", "get_frames", "experiment_results"})
# Whole numbers below this, written without decimals in the answer's text, occur in
# nearly every result; they are counted apart ("small") and neither traced nor untraced.
# The answer's own fields (expected gains and their intervals) are always checked.
SMALL_WHOLE = 10
# A float field is compared at the decimals its shortest form has, at most these.
MAX_DECIMALS = 6

# A number in the answer's text. Not after a letter, a digit, an underscore or a decimal
# point, so that ids (r03, s2, e000123, f01, x0001) and words (day3, k3) carry none;
# a thousands separator is read as one ("1,234"); a sign counts, the minus sign too.
MINUS_SIGN = chr(0x2212)
_TEXT_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<sign>[-+" + MINUS_SIGN + r"]?)"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?![\d,])|\d+)(?P<dec>\.\d+)?(?![\d])"
)
# A number in a tool result or in a call's arguments: also scientific notation.
_DATA_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class AnswerNumber:
    where: str
    text: str
    value: float
    decimals: int


def text_numbers(where: str, text: str) -> Iterator[AnswerNumber]:
    """The numbers written in a text, with their decimals as written.

    An integer below ten glued to the letters after it (1st, 3x, 2h) is part of a
    word and skipped; so is everything glued to a letter before it.
    """
    for match in _TEXT_NUMBER.finditer(text):
        integer, decimals = match["int"], match["dec"] or ""
        value = float(integer.replace(",", "") + decimals)
        after = text[match.end() : match.end() + 1]
        if not decimals and value < 10 and after.isalpha():
            continue
        sign = -1.0 if match["sign"] in ("-", MINUS_SIGN) else 1.0
        yield AnswerNumber(where, match.group(0), sign * value, max(len(decimals) - 1, 0))


def float_number(where: str, value: float) -> AnswerNumber:
    """A number from a float field, at the decimals of its shortest form (at most six)."""
    exponent = Decimal(repr(value)).normalize().as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return AnswerNumber(where, repr(value), value, min(decimals, MAX_DECIMALS))


def answer_numbers(
    diagnosis: Diagnosis | None, fix_set: RankedFixSet | None
) -> tuple[list[AnswerNumber], list[JudgedNumber]]:
    """The numbers to trace, and the confidences, which are judgement and not checked."""
    out: list[AnswerNumber] = []
    judged: list[JudgedNumber] = []
    if fix_set is not None:
        for i, ranked in enumerate(fix_set.actions):
            at = f"fix_set.actions[{i}]"
            out.append(float_number(f"{at}.expected_gain", ranked.expected_gain))
            out.append(float_number(f"{at}.uncertainty.low", ranked.uncertainty.low))
            out.append(float_number(f"{at}.uncertainty.high", ranked.uncertainty.high))
            out.extend(text_numbers(f"{at}.rationale", ranked.rationale))
    if diagnosis is not None:
        for i, cause in enumerate(diagnosis.causes):
            out.extend(text_numbers(f"diagnosis.causes[{i}].failure_mode", cause.failure_mode))
            judged.append(
                JudgedNumber(where=f"diagnosis.causes[{i}].confidence", value=cause.confidence)
            )
        for i, summary in enumerate(diagnosis.mode_summaries):
            out.extend(text_numbers(f"diagnosis.mode_summaries[{i}]", summary))
        for i, warning in enumerate(diagnosis.warnings):
            judged.append(
                JudgedNumber(where=f"diagnosis.warnings[{i}].confidence", value=warning.confidence)
            )
    return out, judged


def result_text(result: Any) -> str:
    """A call's result as text: the text itself, the tool server's list of text parts
    joined, anything else as JSON."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list) and all(isinstance(part, str) for part in result):
        return "\n".join(result)
    return json.dumps(result, default=str)


def data_numbers(text: str) -> list[float]:
    """The absolute values of the numbers in a tool result or in a call's arguments."""
    out = []
    for match in _DATA_NUMBER.finditer(text):
        try:
            value = abs(float(match.group(0)))
        except ValueError:
            continue
        if math.isfinite(value):
            out.append(value)
    return out


@dataclass
class _Call:
    label: str
    arguments: list[float]  # sorted


class Sources:
    """The numbers of every tool result that counts as a source, sorted, with their calls."""

    def __init__(self, tool_log: Sequence[dict[str, Any]]) -> None:
        self.calls: list[_Call] = []
        pairs: list[tuple[float, int]] = []
        for record in tool_log:
            tool = str(record.get("tool", ""))
            if tool not in DATA_SOURCES or record.get("ok") is False:
                continue
            if record.get("delivered") is False:  # the CLI refused the result (too large)
                continue
            arguments = sorted(data_numbers(json.dumps(record.get("arguments") or {}, default=str)))
            session = record.get("session")
            where = (
                f"seq {record.get('seq')}"
                if session is None
                else (f"session {session} seq {record.get('seq')}")
            )
            self.calls.append(_Call(f"{where} ({tool})", arguments))
            index = len(self.calls) - 1
            pairs.extend(
                (value, index) for value in data_numbers(result_text(record.get("result")))
            )
        pairs.sort()
        self.values = [v for v, _ in pairs]
        self.owners = [c for _, c in pairs]

    def find(self, value: float, decimals: int) -> tuple[str | None, bool]:
        """A call whose result shows `value` at `decimals`, searched from the value outward, and
        whether it is only an echo of that call's own arguments.

        A result number shows the value when it lies within half a unit of the
        last decimal. A match in a call whose arguments carry a number in the same
        range is an echo; any other match wins over it.
        """
        target = abs(value)
        tolerance = 0.5 * 10.0 ** (-decimals) * (1 + 1e-9) + 1e-12
        low, high = target - tolerance, target + tolerance
        start = bisect.bisect_left(self.values, low)
        stop = bisect.bisect_right(self.values, high)
        if start == stop:
            return None, False
        middle = min(max(bisect.bisect_left(self.values, target), start), stop - 1)
        echo: str | None = None
        for index in _outward(middle, start, stop):
            call = self.calls[self.owners[index]]
            args = call.arguments
            k = bisect.bisect_left(args, low)
            if k < len(args) and args[k] <= high:
                echo = echo or call.label
                continue
            return call.label, False
        return echo, True


def _outward(middle: int, start: int, stop: int) -> Iterator[int]:
    """Indices from `middle` outward, alternating, within [start, stop)."""
    yield middle
    for step in range(1, stop - start):
        for index in (middle - step, middle + step):
            if start <= index < stop:
                yield index


SCALES: tuple[tuple[str, float, int], ...] = (
    ("as written", 1.0, 0),
    ("x100", 100.0, -2),
    ("/100", 0.01, 2),
)


def trace_number(number: AnswerNumber, sources: Sources) -> TracedNumber:
    """Whether a tool result shows the number as written, times 100 or divided by 100
    (a rate against percentage points). A value that is not finite is untraced."""
    in_text = not number.where.endswith(("expected_gain", ".low", ".high"))
    if in_text and number.decimals == 0 and abs(number.value) < SMALL_WHOLE:
        return TracedNumber(
            where=number.where, text=number.text, value=number.value, status="small"
        )
    echo_source: str | None = None
    scales = SCALES if math.isfinite(number.value) else ()
    for name, factor, shift in scales:
        decimals = max(number.decimals + shift, 0)
        source, is_echo = sources.find(number.value * factor, decimals)
        if source is None:
            continue
        if not is_echo:
            return TracedNumber(
                where=number.where,
                text=number.text,
                value=number.value,
                status="traced",
                source=f"{source}, {name}",
            )
        echo_source = echo_source or f"{source}, {name}"
    if echo_source is not None:
        return TracedNumber(
            where=number.where,
            text=number.text,
            value=number.value,
            status="echo",
            source=echo_source,
        )
    return TracedNumber(where=number.where, text=number.text, value=number.value, status="untraced")


def score_traceability(
    diagnosis: Diagnosis | None, fix_set: RankedFixSet | None, tool_log: Sequence[dict[str, Any]]
) -> TraceScore:
    """Every number of the diagnosis and the fix set, traced to a tool result or not."""
    if diagnosis is None and fix_set is None:
        return TraceScore(missing="no answer submitted", calls=len(tool_log))
    numbers, judged = answer_numbers(diagnosis, fix_set)
    sources = Sources(tool_log)
    traced = [trace_number(n, sources) for n in numbers]
    return TraceScore(
        calls=len(tool_log),
        calls_searched=len(sources.calls),
        traced=sum(t.status == "traced" for t in traced),
        echo=sum(t.status == "echo" for t in traced),
        untraced=sum(t.status == "untraced" for t in traced),
        small=sum(t.status == "small" for t in traced),
        numbers=traced,
        not_checked=judged,
    )
