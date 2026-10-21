"""The wall, checked without a model: what is mounted, and what of the truth is in it.

`check_wall` reuses the runtime's own checks (`fleetlab.agents.checks`): the
export and source audit, the layout of both sandboxes read back from their
command lines, and plain processes inside each sandbox trying the canary, the
truth side, the repository, the home directory and the login stores. It adds a
file-by-file inventory of what each layout mounts.

`seed_checks` adds what an earlier review asked for (finding 5):

- the world seed can be enumerated (a seed of two digits is found by trying);
- a seed or a canary stands in the public surface;
- the world seed stands in a mounted file, as decimal or hexadecimal text or
  as its eight raw bytes: every file of every mount of the tool sandbox,
  interpreter and packages included;
- a seed or canary stands in the mounted source, which no tool shows but a
  process in the sandbox can read.

`export_check` holds the export against the log contract: only declared tables,
columns and description fields.

Whether the world generator's package can be imported from the tool bundle is
`probes.generator_import`.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fleetlab.agents.checks import layout_check, source_audit, static_check
from fleetlab.agents.cli import Backend, agent_root, binary
from fleetlab.agents.run import codex_home_slot
from fleetlab.agents.sandbox import BOX_LOGIN, Layout, ToolMounts, cli_layout, tool_layout
from fleetlab.audit.scan import Scanner
from fleetlab.audit.truth import Inventory
from fleetlab.contracts.log import log_tree_problems
from fleetlab.contracts.model import Fleet

__all__ = ["SYSTEM_MOUNTS", "check_wall", "export_check", "files", "seed_checks"]

# Mounts of system files and of the interpreter: thousands of files that no audit
# of one session needs listed again.
SYSTEM_MOUNTS = frozenset(
    {"/usr", "/etc/ssl", "/etc/ca-certificates", "/opt/cli", "/opt/tools/python", "/opt/tools/site"}
)
HASHED_MOUNTS = frozenset({"/work", "/skills", "/opt/tools/src", "/results"})
# A seed below this many values is found by trying them through the SQL tool: 3.9 million
# SHA-256 a second were measured (an earlier review), 2**40 take about three days.
ENUMERABLE_BELOW = 2**40
SEED_KINDS = frozenset({"world_seed", "episode_seed", "arm_seed", "canary"})


def files(
    layout: Layout, path: Path, *, nested: bool = False, summarise: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """File-by-file inventory; credentials are names/metadata only, never read or hashed.

    Mounts named in `summarise` are counted, not listed.
    """
    count = 0
    problems: list[str] = []
    with path.open("w") as output:
        for mount in layout.mounts:
            if mount.kind not in {"bind", "ro-bind"} or mount.src is None:
                output.write(json.dumps({**asdict(mount), "type": "virtual"}) + "\n")
                continue
            if nested:
                output.write(json.dumps({**asdict(mount), "type": "outer mount reference"}) + "\n")
                continue
            root = Path(mount.src)
            credential = mount.dst in BOX_LOGIN.values()
            # Do not even inspect the credential file. Its individual mount is
            # validated by layout_check; CLI authentication belongs to the runtime.
            if credential:
                output.write(
                    json.dumps({**asdict(mount), "type": "credential; not inspected"}) + "\n"
                )
                continue
            if mount.dst in summarise:
                inside = sum(1 for _ in root.rglob("*")) if root.is_dir() else 1
                counted = {**asdict(mount), "type": "system mount, counted", "entries": inside}
                output.write(json.dumps(counted) + "\n")
                count += inside
                continue
            paths: Sequence[Path] = [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]
            for item in paths:
                dst = mount.dst + ("/" + item.relative_to(root).as_posix() if item != root else "")
                try:
                    metadata = item.lstat()
                    row: dict[str, Any] = {
                        "src": str(item),
                        "dst": dst,
                        "mount": mount.kind,
                        "bytes": metadata.st_size,
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "allowed_by": mount.dst,
                    }
                    if item.is_symlink():
                        row.update(type="symlink", target=str(item.readlink()))
                    elif item.is_file():
                        row["type"] = "file"
                        if mount.dst in HASHED_MOUNTS:
                            row["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
                    elif item.is_dir():
                        row["type"] = "directory"
                    else:
                        row["type"] = "special"
                    output.write(json.dumps(row) + "\n")
                    count += 1
                except OSError as error:
                    problems.append(f"{dst}: {error}")
    return {"record": str(path), "entries": count, "problems": problems}


def check_wall(
    mounts: ToolMounts, out: Path, world: Path, runs: list[Path], fleet: Fleet
) -> dict[str, Any]:
    reports = [source_audit(mounts.export, runs, mounts.env, fleet=fleet).as_json()]
    inventories = []
    out.mkdir()
    for backend in ("claude", "codex"):
        typed: Backend = backend
        hidden = [world / "files", world / "oracle", *[r / "truth" for r in runs]]
        reports += [
            layout_check(typed, mounts.env, mounts.export, hidden).as_json(),
            static_check(typed, mounts.env, mounts.export, hidden).as_json(),
        ]
        with codex_home_slot(agent_root()) as home:
            box = mounts.log_dir.parent / backend / "box"
            (box / "cwd").mkdir(parents=True)
            outer = cli_layout(
                typed, binary(typed), box, mounts, subscription=True, codex_home=home
            )
            inventories.append(
                files(outer, out / f"{backend}-files.jsonl", summarise=SYSTEM_MOUNTS)
            )
    inner = tool_layout(mounts, nested=False)
    inventories.append(files(inner, out / "inner-files.jsonl", summarise=frozenset({"/usr"})))
    return {
        "passed": all(r["passed"] for r in reports) and not any(i["problems"] for i in inventories),
        "checks": reports,
        "inventories": inventories,
        "notes": [
            "System mounts (/usr, certificates, the CLI's directory) are counted; in the tool "
            "sandbox the interpreter and packages are listed by name, type and size; export, "
            "source, skills and results with SHA-256. Credential mounts are never inspected.",
            "Virtual /proc, /dev and tmpfs entries are declared, not host directory inventories.",
        ],
    }


def export_check(export: Path) -> dict[str, Any]:
    """Whether the export holds only what the log contract declares visible: its tables,
    their columns, the fields of the description (`contracts.log.log_tree_problems`).

    The scan takes a hidden value beside a declared public field for that field's own
    value, and a truth-side name that the export shows for a public name. Both rest on
    this: an export with a column of its own would make its content public by being there.
    """
    problems = log_tree_problems(export)
    return {
        "name": "the export holds only what the log contract declares visible",
        "passed": not problems,
        "count": len(problems),
        "problems": problems[:50],
    }


def _patterns(seed: int) -> dict[str, bytes]:
    out = {
        "decimal text": str(seed).encode(),
        "hexadecimal text": format(seed, "x").encode(),
        "hexadecimal text, upper case": format(seed, "X").encode(),
    }
    if seed < 2**64:
        out["eight bytes, big end first"] = seed.to_bytes(8, "big")
        out["eight bytes, little end first"] = seed.to_bytes(8, "little")
    return out


def seed_checks(inventory: Inventory, scanner: Scanner, mounts: ToolMounts) -> list[dict[str, Any]]:
    """Finding 5: the seeds must be neither guessable nor anywhere in what is mounted."""
    by_kind = inventory.by_kind()
    seeds = sorted(int(f.value) for f in by_kind["world_seed"] if f.form == "exact" and f.numeric)
    rows: list[dict[str, Any]] = [
        {
            "name": "the world seed cannot be found by trying",
            "passed": bool(seeds) and all(seed >= ENUMERABLE_BELOW for seed in seeds),
            "seeds": len(seeds),
            "below_2_to_the_40": sum(seed < ENUMERABLE_BELOW for seed in seeds),
        }
    ]
    shown = []
    for fact in inventory.facts:
        if fact.kind not in SEED_KINDS or (fact.kind != "world_seed" and not fact.strong):
            continue
        where = scanner.public_values.get((fact.key, "exact"))
        if where is not None:
            shown.append({"kind": fact.kind, "form": fact.form, "shown_in": sorted(where)})
    rows.append(
        {
            "name": "no seed and no canary in the public surface",
            "passed": not shown,
            "found": shown[:50],
            "count": len(shown),
        }
    )
    hits: list[dict[str, str]] = []
    searched = 0
    strong = [seed for seed in seeds if seed >= ENUMERABLE_BELOW]
    roots = [Path(host) for _, host, _ in mounts.pairs() if host is not None]
    for root in roots if strong else []:
        for path in [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError as error:
                hits.append({"file": str(path), "form": f"unreadable: {error}"})
                continue
            searched += 1
            for seed in strong:
                for form, pattern in _patterns(seed).items():
                    if pattern in data:
                        hits.append({"file": str(path), "form": form})
    rows.append(
        {
            "name": "the world seed stands in no mounted file",
            "passed": bool(strong) and not hits,
            "files_searched": searched,
            "found": hits[:50],
            "note": "" if strong else "not searched: a seed this short matches by chance",
        }
    )
    in_source = []
    for tree in (mounts.env.src, mounts.env.skills):
        for path in sorted(tree.rglob("*")) if tree is not None else []:
            if not path.is_file():
                continue
            for cand, bare in scanner.scan(path.read_text(errors="replace")):
                if bare and cand.fact.kind in SEED_KINDS and cand.fact.strong:
                    in_source.append({"file": path.name, "kind": cand.fact.kind, "form": cand.form})
    rows.append(
        {
            "name": "no seed and no canary in the mounted source and skills",
            "passed": not in_source,
            "found": in_source[:50],
        }
    )
    return rows
