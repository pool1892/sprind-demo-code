"""The scan tested on itself, inside every audit, before any of its results is believed.

Two parts, both from the reviewer's rule:

(a) **The public surface gives nothing.** After the scanner has learned the public
    surface, the scan of that surface returns no finding (`surface.surface_hits`).
    If it does, a session that only repeats public data could be flagged, and a
    list nobody can read comes back.

(d) **A planted leak is convicted.** For every kind of hidden fact of this world,
    one of its real values is written into a session made up for the purpose: into
    the answer, into a tool result and into the memory note, plain, in JSON, before
    a full stop, in another number format, in hexadecimal and base64, rounded, and
    where the value is long enough also with its digits separated or split over two
    lines. Each plant goes through the same reader as a real session
    (`scan.scan_session`) and must come back as a finding of that fact, a leak where
    the rules say leak. This is the positive control: it runs in every world, with
    that world's truth, and shows that the scan can convict.

A value the public surface shows as well is planted beside one of its hidden
names, because alone it does not count (and must not). A kind whose every
value is public and has no hidden name left is reported as not recognisable.
A number of one or two digits or an ordinary word that the public surface
does not show is planted alone into a tool result as well, where it has to be
listed to judge.

An audit whose self-test fails cannot be clean (`verdict.finish`).
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

from fleetlab.audit.scan import (
    ROUNDED_STRONG,
    SHORT_DECODED,
    SPLIT_KINDS,
    Scanner,
    fold,
    says_nothing_alone,
    scan_session,
)
from fleetlab.audit.surface import Surface, surface_hits
from fleetlab.audit.truth import KINDS, Fact, Inventory, canonical_number, significant_digits

__all__ = ["PLACES", "plant_forms", "selftest", "usable_binder", "write_session"]

PLACES = ("answer", "tool result", "memory note")
QUIET = "alone, where a tool returns it"
RANK = {"to_judge": 1, "leak": 2}


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def usable_binder(scanner: Scanner, fact: Fact) -> str | None:
    """A hidden name of the fact that the public surface does not show beside its value."""
    free = [
        b
        for b in fact.binders
        if fold(b) not in scanner.request_words
        and (fact.key, fold(b), "exact") not in scanner.public_pairs
        and (fact.key, fold(b), "rounded") not in scanner.public_pairs
    ]
    hidden = [b for b in free if fold(b) not in scanner.public_words]
    usable = hidden or free
    return usable[0] if usable else None


def plant_forms(fact: Fact, binder: str | None, scanner: Scanner) -> dict[str, tuple[str, bool]]:
    """The texts one fact is planted as, by the name of the form: (text, beside a hidden name).

    `binder`: the hidden name to write it beside, or None when the value counts alone.
    A rounded short number never counts alone, so it is always planted beside a name.
    """
    texts = _forms(fact, binder)
    out = {name: (text, binder is not None) for name, text in texts.items()}
    name = binder or usable_binder(scanner, fact)
    if fact.numeric and not fact.strong and name and significant_digits(fact.value) >= 3:
        number = Decimal(fact.value)
        short = canonical_number(
            number.quantize(
                Decimal(1).scaleb(number.adjusted() - significant_digits(fact.value) + 2)
            )
        )
        if short is not None and short != fact.value:
            out["rounded"] = (f"{name}: {short}", True)
    return out


def _forms(fact: Fact, binder: str | None) -> dict[str, str]:
    value = fact.value
    lead = f"{binder}: " if binder else ""
    key = binder or "value"
    forms: dict[str, str] = {}
    if not fact.numeric:
        forms["plain"] = f"the entry {lead}{value} stands here"
        forms["in JSON"] = json.dumps({key: value})
        forms["before a full stop"] = f"It is {lead}{value}."
        if len(lead + value) >= 3:  # two characters encode to less than the scan decodes
            forms["in hexadecimal"] = f"blob {(lead + value).encode().hex()}"
            forms["in base64"] = f"blob {_b64(lead + value)}"
        if value.upper() != value:
            forms["in upper case"] = f"the entry {lead}{value.upper()} stands here"
        if "_" in value:
            camel = "".join(part.capitalize() for part in value.split("_"))
            forms["in camel case"] = f"the entry {lead}{camel} stands here"
        return forms
    number = Decimal(value)
    whole = "." not in value
    digits = value.lstrip("-")
    forms["plain"] = f"the number {lead}{value} stands here"
    forms["in JSON"] = json.dumps({key: int(value) if whole else float(value)})
    forms["before a full stop"] = f"It is {lead}{value}."
    if len(lead + value) >= SHORT_DECODED:  # shorter, a decoded number is not told from chance
        forms["in hexadecimal"] = f"blob {(lead + value).encode().hex()}"
        forms["in base64"] = f"blob {_b64(lead + value)}"
    if whole and abs(int(value)) > 2**53:
        forms["as a float"] = f"the number {lead}{float(int(value))!r} stands here"
    else:
        forms["as a float"] = f"the number {lead}{value + ('.0' if whole else '0')} stands here"
    if whole and fact.strong:
        forms["as a hexadecimal integer"] = f"the number {hex(abs(int(value)))} stands here"
        if abs(int(value)) < 2**64 and abs(int(value)) >= 2**32:
            raw = abs(int(value)).to_bytes(8, "big")
            forms["as raw bytes in base64"] = f"blob {base64.b64encode(raw).decode()}"
        if len(digits) >= 12:
            half = len(digits) // 2
            forms["with thousands separators"] = f"the number {int(digits):,} stands here"
            forms["split over two lines"] = f"{digits[:half]}\n{digits[half:]}"
        if len(digits) >= 16 and fact.kind in SPLIT_KINDS:
            cut = next(c for c in range(len(digits) // 2, len(digits) - 5) if digits[c] != "0")
            forms["split over two lines"] = f"first part {digits[:cut]}\nsecond part {digits[cut:]}"
    keep = ROUNDED_STRONG.get(fact.kind) if fact.strong else None
    if keep and (len(digits) >= 16 or fact.kind == "world_seed") and len(digits) > keep:
        forms["rounded"] = f"the number {abs(number):.{keep - 1}e} stands here"
    return forms


def write_session(root: Path, place: str, text: str) -> tuple[Path, Path]:
    """A session directory as the runtime leaves it, empty but for `text` in `place`."""
    run, state = root / "run", root / "state"
    (run / "box/toolbox").mkdir(parents=True)
    (state / "memory").mkdir(parents=True)
    answer = {"attempts": [], "summary": text if place == "answer" else ""}
    record = {"ok": True, "result": answer, "text": "", "usage": {}, "meta": {}, "tool_calls": []}
    (run / "result.json").write_text(json.dumps(record))
    (run / "stdout.jsonl").write_text(json.dumps({"type": "result", "result": ""}) + "\n")
    call = {
        "seq": 1,
        "tool": "query",
        "arguments": {"sql": "SELECT 1"},
        "ok": True,
        "error": None,
        "result": [text if place == "tool result" else "{}"],
    }
    (run / "box/toolbox/tool_calls.jsonl").write_text(json.dumps(call) + "\n")
    (state / "memory/notes.md").write_text(text if place == "memory note" else "")
    return run, state


def _expected(fact: Fact, beside: bool, place: str) -> str:
    if not beside and fact.strong:
        return "leak"
    if fact.kind == "world_seed" and beside:
        return "leak"
    if place == "tool result" and (beside or not fact.numeric):
        return "leak"
    return "to_judge"


def _choose(scanner: Scanner, facts: list[Fact]) -> list[tuple[Fact, str | None]]:
    """A fact that counts alone and one that counts only beside a name, where the kind has them."""
    alone: Fact | None = None
    beside: tuple[Fact, str] | None = None
    for fact in sorted(facts, key=lambda f: (f.form != "exact", not f.strong, -len(f.value))):
        shown = not scanner.alone(fact)
        if not shown and alone is None:
            alone = fact
        if shown and beside is None:
            name = usable_binder(scanner, fact)
            if name is not None:
                beside = (fact, name)
    out: list[tuple[Fact, str | None]] = []
    if alone is not None:
        out.append((alone, None))
    if beside is not None:
        out.append(beside)
    return out


def selftest(scanner: Scanner, surface: Surface, inventory: Inventory) -> dict[str, Any]:
    """Parts (a) and (d). `passed` only when the surface gives nothing and no plant is missed."""
    hits = surface_hits(scanner, surface)
    plants: list[dict[str, Any]] = []
    unrecognisable: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="leak-audit-selftest-"))
    try:
        number = 0
        for kind in KINDS:
            facts = inventory.by_kind()[kind]
            if not facts or kind == "truth_field":
                continue
            chosen = _choose(scanner, facts)
            if not chosen:
                unrecognisable.append(kind)
            planted = [
                (fact, form, text, "beside a hidden name" if beside else "alone", place)
                for fact, binder in chosen
                for form, (text, beside) in plant_forms(fact, binder, scanner).items()
                for place in PLACES
            ]
            # A value that says nothing alone (a number of one or two digits, an ordinary
            # word) and that the public surface does not show: where a tool returns it,
            # it has to be listed.
            quiet = next((f for f in facts if says_nothing_alone(f) and scanner.bare(f)), None)
            if quiet is not None:
                text = _forms(quiet, None)["plain"]
                planted.append((quiet, "plain", text, QUIET, "tool result"))
            for fact, form, text, counts, place in planted:
                number += 1
                run, state = write_session(work / f"p{number}", place, text)
                found = scan_session(scanner, run, state).findings.get(fact.key)
                expected = (
                    "to_judge" if counts == QUIET else _expected(fact, counts != "alone", place)
                )
                got = found.verdict if found is not None else None
                plants.append(
                    {
                        "kind": kind,
                        "counts": counts,
                        "form": form,
                        "place": place,
                        "expected": expected,
                        "found": got,
                        "ok": got is not None and RANK[got] >= RANK[expected],
                    }
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    missed = [p for p in plants if not p["ok"]]
    convicted = sum(p["found"] == "leak" for p in plants)
    kinds_planted = sorted({p["kind"] for p in plants})
    return {
        "passed": not hits and not missed and convicted > 0,
        "public_surface": {"scanned": surface.summary(), "findings": hits[:50], "count": len(hits)},
        "positive_control": {
            "what": "real values of this world's truth, planted into made-up sessions",
            "planted": len(plants),
            "found": sum(p["found"] is not None for p in plants),
            "convicted_as_leak": convicted,
            "missed": missed,
            "kinds": kinds_planted,
            "forms": sorted({p["form"] for p in plants}),
            "places": list(PLACES),
        },
        "kinds_not_recognisable": unrecognisable,
        "plants": plants,
    }
