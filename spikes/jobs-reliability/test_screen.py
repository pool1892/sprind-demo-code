"""The screen's CPU checks: the committed batches validate, and the report reads a run dir.

tools/heavy-run --light -m 2G uv run --locked pytest spikes/jobs-reliability -p no:cacheprovider
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import screen  # noqa: E402


@pytest.mark.parametrize("stage", ["screen-01", "hold-01", "confirm-01", "expanded-screen-01"])
def test_committed_batches_validate(stage: str) -> None:
    folder = HERE / "batches" / stage
    manifest = json.loads((folder / screen.MANIFEST).read_text())
    for f in sorted(folder.glob("*.batch.json")):
        assert screen.batch_problems(json.loads(f.read_text()), manifest) == []


def test_screen_size_and_starts() -> None:
    manifest = json.loads((HERE / "batches/screen-01" / screen.MANIFEST).read_text())
    assert manifest["executions"] == 170
    assert manifest["init_states"] == [0, 9]
    arms = sum(1 for r in manifest["jobs"] if r["variant"] not in ("naive", "single"))
    assert arms == 9  # at most two beside naive per candidate


def test_expanded_screen() -> None:
    """150 executions of five new pairs, under the expanded label, on the same suite."""
    folder = HERE / "batches/expanded-screen-01"
    manifest = json.loads((folder / screen.MANIFEST).read_text())
    first = json.loads((HERE / "batches/screen-01" / screen.MANIFEST).read_text())
    assert manifest["executions"] == 150 and manifest["init_states"] == [0, 9]
    assert manifest["tasks"] == first["tasks"]  # the canonical goals and instructions
    for words in ("HOST-AUTHORED", "task-selection", "withheld from engineering agents"):
        assert words in manifest["label"]
    assert "robots-qzfk.44.7" in manifest["label"]
    assert manifest["candidates"] == {c: list(t) for c, t in screen.EXPANDED_CANDIDATES.items()}
    arms = {(r["batch"], r["variant"]) for r in manifest["jobs"]}
    assert arms == {(c, v) for c in screen.EXPANDED_CANDIDATES for v in screen.EXPANDED_ARMS[c]}
    assert {r["digest"] for r in manifest["jobs"]}.isdisjoint(r["digest"] for r in first["jobs"])
    for cand, tasks in screen.EXPANDED_CANDIDATES.items():
        batch = json.loads((folder / f"{cand}.batch.json").read_text())
        assert batch["host_screen"]["bead"] == "robots-qzfk.44.7"
        assert batch["state"]["requested_goals"] == [first["tasks"][t]["goal"] for t in tasks]
        assert batch["state"]["order_rules"] == []


def test_validate_rebuilds_programs_and_labels() -> None:
    """A changed transition is caught even with the manifest's digest updated to match, and a
    batch under the other screen's label is caught."""
    from fleetlab.contracts.jobs import JobProgram

    folder = HERE / "batches/expanded-screen-01"
    manifest = json.loads((folder / screen.MANIFEST).read_text())
    batch = json.loads((folder / "c6.batch.json").read_text())
    program = batch["jobs"][2]["request"]["program"]  # finish-home
    program["steps"][1]["until"]["steps"] = program["steps"][1]["max_steps"] = 60
    row = next(r for r in manifest["jobs"] if (r["batch"], r["job_id"]) == ("c6", "j0003"))
    row["digest"] = JobProgram.model_validate(program).digest
    assert screen.batch_problems(batch, manifest) == [
        "c6/j0003: program is not the screen's finish-home program"
    ]
    manifest = json.loads((folder / screen.MANIFEST).read_text())
    batch = json.loads((folder / "c6.batch.json").read_text())
    batch["host_screen"]["bead"] = "robots-qzfk.44.3"
    assert screen.batch_problems(batch, manifest) == [
        "c6: host_screen is not the robots-qzfk.44.7 screen's label and bead"
    ]


@pytest.mark.parametrize(
    "picks",
    [
        ["c6:home", "c7:home", "c8:home", "--with-naive"],  # more than two picks
        ["c6:home"],  # without the naive chains
        ["c6:lift-home", "--with-naive"],  # an arm the expanded screen did not run
        ["c6:home", "c4:lift-home", "--with-naive"],  # two screens in one stage
    ],
)
def test_expanded_confirm_refuses(picks: list[str]) -> None:
    argv = ["confirm"]
    for p in picks:
        argv += [p] if p.startswith("--") else ["--pick", p]
    with pytest.raises(SystemExit):
        screen.main(argv)


def test_write_never_rewrites_a_stage(tmp_path: Path) -> None:
    screen._write(tmp_path, {"c6": {"a": 1}}, {"m": 1})
    screen._write(tmp_path, {"c6": {"a": 1}}, {"m": 1})  # identical: allowed
    with pytest.raises(SystemExit):
        screen._write(tmp_path, {"c7": {"a": 1}}, {"m": 2})
    assert not (tmp_path / "c7.batch.json").exists()


def _episode(job: str, start: int, delivered: bool, first: list, end: list, **delivery) -> dict:
    return {
        "job_id": job,
        "episode_id": f"e{start:06d}",
        "start": {"init_state": start},
        "delivered": delivered,
        "goals": [
            {"goal": g, "first_held_step": f, "held_at_end": e}
            for g, (f, e) in enumerate(zip(first, end, strict=True))
        ],
        "program_steps": [{"outcome": "reached"}, {"outcome": "timeout"}],
        "control_steps": 400,
        "delivery": {
            "reason": "x",
            "robot_touches": delivery.get("touches", []),
            "not_at_rest": delivery.get("moving", []),
        },
    }


def test_report_counts_mechanisms_and_pairs(tmp_path: Path) -> None:
    run = tmp_path / "c9"
    run.mkdir()
    steps = [{"kind": "policy"}, {"kind": "policy"}]
    record = {
        "jobs": [
            {
                "job_id": "j0001",
                "program": {"name": "naive-chain", "steps": steps},
                "program_digest": "a",
            },
            {
                "job_id": "j0002",
                "program": {"name": "host-c9-lift-home", "steps": steps},
                "program_digest": "b",
            },
        ]
    }
    (run / "run.json").write_text(json.dumps(record))
    (run / "batch_result.json").write_text(
        json.dumps(
            {"jobs": [{"job_id": "j0001", "status": "done"}, {"job_id": "j0002", "status": "done"}]}
        )
    )
    episodes = [
        _episode("j0001", 0, True, [10, 50], [True, True]),
        _episode("j0001", 1, False, [10, None], [True, False]),
        _episode("j0001", 2, False, [10, 90], [True, True], touches=["plate_1"]),
        _episode("j0002", 0, False, [10, 70], [False, True]),
        _episode("j0002", 1, True, [10, 80], [True, True]),
        _episode("j0002", 2, True, [10, 95], [True, True]),
    ]
    (run / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    jobs = screen.summarise_run(run)
    naive, lift = jobs
    assert (naive["variant"], naive["delivered"], naive["n"]) == ("naive", 1, 3)
    assert naive["mechanisms"] == {
        "delivered": 1,
        "goal1_never_held": 1,
        "goals_hold_robot_touching": 1,
    }
    assert lift["variant"] == "lift-home"
    assert lift["mechanisms"] == {"goal0_lost_after_holding": 1, "delivered": 2}
    assert lift["goal_reached_any"] == [3, 3] and lift["goal_held_at_end"] == [2, 3]
    (pair,) = screen.paired(jobs)
    assert (pair["rescued"], pair["broken"]) == ([1, 2], [0])


def test_report_takes_the_stage_label(tmp_path: Path) -> None:
    """Expanded results are attributed to the expanded screen, a folder without a copied
    manifest to the first screen."""
    assert screen.report_label(tmp_path) == screen.LABEL
    (tmp_path / "batches").mkdir()
    copied = HERE / "batches/expanded-screen-01" / screen.MANIFEST
    (tmp_path / "batches" / screen.MANIFEST).write_text(copied.read_text())
    screen.main(["report", str(tmp_path)])
    for name in ("screen_report.md", "screen_report.json"):
        text = (tmp_path / name).read_text()
        assert "robots-qzfk.44.7" in text and "robots-qzfk.44.3" not in text


def test_wilson() -> None:
    lo, hi = screen.wilson(19, 20)
    assert round(lo, 3) == 0.764 and round(hi, 3) == 0.991


FAKE_UV = """#!/bin/sh
case "$*" in
    *" validate "*) exit 0 ;;
    *" -c "*) echo single-drawer ;;
    *fleetlab.jobs.runner*) echo "runner: {'done': 1, 'refused': 0, 'failed': 0} in 0 s" ;;
    *" report "*) exit "$FAKE_REPORT_RC" ;;
    *) exit 3 ;;
esac
"""
FAKE_HEAVY = """#!/bin/sh
[ "$1" = --light ] && shift 3
exec "$@"
"""


@pytest.mark.parametrize(("report_rc", "want"), [(0, 0), (1, 1)])
def test_run_stage_exit_follows_the_report(tmp_path: Path, report_rc: int, want: int) -> None:
    """run_stage.sh fails when the report cannot be written (it used to exit 0)."""
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("uv", FAKE_UV), ("heavy", FAKE_HEAVY)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RELIABILITY_ROOT": str(tmp_path / "out"),
        "RELIABILITY_HEAVY_RUN": str(bin_dir / "heavy"),
        "FAKE_REPORT_RC": str(report_rc),
    }
    done = subprocess.run(
        ["sh", str(HERE / "run_stage.sh"), "hold-01", "hold"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == want, done.stdout + done.stderr
    assert "single-drawer: exit 0" in done.stdout


def test_goal_losses_tell_transient_from_final() -> None:
    """A loss that is regained is not the final loss, and the first loss is not the last."""
    import traces

    def trace(*held: int) -> dict[int, bool]:
        return {s: bool(h) for s, h in enumerate(held)}

    # Held at 1, lost at 3, regained at 5, lost again at 7 for good.
    found = traces.goal_losses(trace(0, 1, 1, 0, 0, 1, 1, 0, 0))
    assert found["losses"] == [(3, 5), (7, None)]
    assert (found["first_loss"], found["final_loss"], found["held_end"]) == (3, 7, False)
    # Lost and regained: one transient loss, no final loss.
    found = traces.goal_losses(trace(0, 1, 0, 1, 1))
    assert found["losses"] == [(2, 3)]
    assert (found["first_loss"], found["final_loss"], found["held_end"]) == (2, None, True)
    # Never held: no losses at all.
    assert traces.goal_losses(trace(0, 0, 0))["losses"] == []
