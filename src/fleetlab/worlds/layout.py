"""Where a world's files and runs live (agreed with the oracle).

One root per world, `~/.cache/robots-probes/worlds/<world id>/` by default:

```
files/                  the world files (contracts section 9): spec.json, model.json,
                        description.json, manifest.jsonl, hash; and the truth side beside them:
                        fault_table.json, detectability.json, health.jsonl, task_texts.json,
                        world_truth.json, generator.json
runs/<wid>-observed/    the engine run of the observed window as run, finalized, with the
                        fleet's records (description, change and maintenance logs, health)
                        and the derived events written into it
runs/<wid>-later/       the same for the experiment and scoring windows
export/                 the agent-visible export of the observed run (log/, data_dictionary.md),
                        leak audit passed
runner.json             the runner's steps with their times; "ready" once everything above is
                        done; nothing under runs/ changes after that
oracle/                 the oracle's; the runner never writes there
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.contracts.ids import IdKind, check_neutral

DEFAULT_ROOT = Path.home() / ".cache" / "robots-probes" / "worlds"
RUN_WINDOWS: dict[str, tuple[str, ...]] = {
    "observed": ("observed",),
    "later": ("experiment", "scoring"),
}


@dataclass(frozen=True)
class WorldPaths:
    root: Path
    world_id: str

    @classmethod
    def from_root(cls, root: Path | str) -> WorldPaths:
        """Read identity from records, never from the root's directory name.

        files/spec.json is authoritative: it identifies the composed world. The
        runner's world_id must agree when present; it can identify a root whose
        spec has not been written yet. Neither the root name nor the fleet id
        identifies the world (a public fleet id can deliberately differ).
        """
        root = Path(root).expanduser()
        spec_file = root / "files/spec.json"
        # Retain the miniature fleet's and standalone spec's legacy layouts.
        if not spec_file.is_file():
            for candidate in (root / "world/spec.json", root / "spec.json"):
                if candidate.is_file():
                    spec_file = candidate
                    break
        runner_file = root / "runner.json"
        identities: dict[Path, str] = {}
        for path in (spec_file, runner_file):
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise ValueError(f"{path}: cannot read world identity: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}: world record must be a JSON object")
            value = record.get("world_id")
            if value is not None:
                try:
                    identities[path] = check_neutral(IdKind.world, value)
                except ValueError as error:
                    raise ValueError(f"{path}: invalid world_id: {error}") from error
        if not identities:
            raise ValueError(f"no world_id in {spec_file} or {runner_file}")
        if len(set(identities.values())) != 1:
            raise ValueError(
                f"conflicting world_id: {spec_file} names {identities[spec_file]!r}; "
                f"{runner_file} names {identities[runner_file]!r}"
            )
        return cls(root, next(iter(identities.values())))

    @classmethod
    def default(cls, world_id: str) -> WorldPaths:
        return cls(DEFAULT_ROOT / world_id, world_id)

    @property
    def files(self) -> Path:
        return self.root / "files"

    def run(self, name: str) -> Path:
        """The run directory of `observed` or `later`."""
        if name not in RUN_WINDOWS:
            raise KeyError(f"no run {name!r}; runs are {sorted(RUN_WINDOWS)}")
        return self.root / "runs" / f"{self.world_id}-{name}"

    @property
    def observed(self) -> Path:
        return self.run("observed")

    @property
    def later(self) -> Path:
        return self.run("later")

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def record(self) -> Path:
        return self.root / "runner.json"

    @property
    def oracle(self) -> Path:
        return self.root / "oracle"

    def runs(self) -> list[Path]:
        """The run directories that exist, observed first."""
        return [p for p in (self.observed, self.later) if (p / "run.json").is_file()]

    def read_record(self) -> dict[str, Any]:
        if not self.record.is_file():
            return {}
        data: dict[str, Any] = json.loads(self.record.read_text())
        return data

    @property
    def ready(self) -> bool:
        from fleetlab.worlds.runner import is_ready

        return is_ready(self.read_record(), self)

    @property
    def readiness_problem(self) -> str | None:
        from fleetlab.worlds.runner import readiness_problem

        return readiness_problem(self.read_record(), self)
