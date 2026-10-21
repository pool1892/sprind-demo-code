"""Cold-import boundaries, including access to the lazy public entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FORBIDDEN = (
    "fleetlab.engine",
    "fleetlab.oracle",
    "fleetlab.worlds",
    "fleetlab.faults",
    "fleetlab.scoring",
    "fleetlab.sim",
)


@pytest.mark.parametrize(
    "statement",
    [
        "import fleetlab.baselines",
        "import fleetlab.baselines.classical",
        "import fleetlab.baselines.classical.experiments",
        "from fleetlab.baselines.models import no_data_ranker",
        "from fleetlab.baselines.simple import worst_health",
        "from fleetlab.baselines.video import dramatic_videos",
        "from fleetlab.baselines import no_data_ranker",
        "from fleetlab.baselines import worst_health",
        "from fleetlab.baselines import dramatic_videos",
        "from fleetlab.baselines import random_fixes",
        "from fleetlab.baselines import fixed_prior",
        "import fleetlab.baselines.__main__",
    ],
)
def test_blind_entry_point_imports_no_truth_or_executor(statement: str) -> None:
    code = f"""
import sys
from importlib.abc import MetaPathFinder
forbidden = {FORBIDDEN!r}
class RefuseTruth(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(forbidden), fullname
sys.meta_path.insert(0, RefuseTruth())
{statement}
assert not [n for n in sys.modules if n.startswith(forbidden)]
assert 'fleetlab.baselines._seed' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr


def test_seed_reader_adds_only_its_own_module_and_standard_library() -> None:
    code = """
import sys
import fleetlab.baselines
before = set(sys.modules)
import fleetlab.baselines._seed
added = set(sys.modules) - before
unexpected = [n for n in added if n != 'fleetlab.baselines._seed'
              and n.split('.')[0] not in sys.stdlib_module_names]
assert not unexpected, unexpected
assert 'fleetlab.baselines.common' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr


def test_random_entry_point_reads_seed_only_on_implicit_seed_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.baselines import _seed
    from fleetlab.baselines.random import random_fixes
    from fleetlab.contracts.actions import catalogue
    from fleetlab.contracts.model import Fleet, Robot, Site

    entries = catalogue(
        Fleet(
            id="w01", sites=[Site(id="s1", robots=[Robot(id="r01", site_id="s1", components={})])]
        )
    )
    entries = [a for a in entries if a.is_complete]
    monkeypatch.setattr("fleetlab.baselines.random.export_catalogue", lambda export: entries)
    reads = []

    def reader(export: Path, chance_table: Path | None = None) -> int:
        reads.append((export, chance_table))
        return 197

    monkeypatch.setattr(_seed, "default_seed", reader)
    explicit = random_fixes(tmp_path, seed=197, draws=1)
    assert reads == []
    chance = tmp_path / "specific-chance-table.json"
    implicit = random_fixes(tmp_path, draws=1, chance_table=chance)
    assert reads == [(tmp_path, chance)]
    assert implicit == explicit
