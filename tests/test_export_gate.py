"""Gate texts against real-world names, without a dependency on stored cache data."""

import ast
import inspect
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import write_toy_run

from fleetlab.agents import measure
from fleetlab.agents.checks import tool_texts
from fleetlab.contracts.live import Decision
from fleetlab.contracts.tools import TOOLS
from fleetlab.engine import episode
from fleetlab.integration import chain
from fleetlab.log import audit_export, leaks
from fleetlab.worlds import events


def event_texts(node: ast.AST) -> list[str]:
    """Literal templates and both branches; formatted scalar slots stand in as 1."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return event_texts(node.body) + event_texts(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [a + b for a in event_texts(node.left) for b in event_texts(node.right)]
    if isinstance(node, ast.JoinedStr):
        return ["".join(str(v.value) if isinstance(v, ast.Constant) else "1" for v in node.values)]
    return []


def test_real_world_names_accept_gate_texts_and_engine_details(tmp_path: Path) -> None:
    root = tmp_path / "w01"
    run = root / "runs/w01-observed"
    write_toy_run(run)
    files = root / "files"
    files.mkdir()
    (files / "spec.json").write_text("{}")
    fixture = json.loads((Path(__file__).parent / "data/export-w41-truth-keys.json").read_text())
    (files / "record.json").write_text(json.dumps(dict.fromkeys(fixture["keys"])))
    out = tmp_path / "export"
    out.mkdir()
    texts = tool_texts()
    # Every tool the agent can read is in the gate, including the loop's (docs/design/loop.md).
    assert len(texts) == len(TOOLS)
    assert {
        "tool write_learned_skill",
        "tool experiment_budget",
        "tool earlier_decisions",
        "tool submit_decision",
    } <= set(texts)
    texts.update(
        chain_system=chain.SYSTEM,
        chain_task=chain.TASK,
        measure_system=measure.SYSTEM,
        measure_task=measure.TASK,
        frames=measure.FRAMES_PROMPT,
        prose='has run; direction "degraded"; serviced by the vendor; two rows of bowls',
    )
    # Read the engine's event producers, so a new detail is included automatically.
    for node in ast.walk(ast.parse(inspect.getsource(episode))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "events"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            detail = node.args[0].elts[-1]
            for i, rendered in enumerate(event_texts(detail)):
                texts[f"engine:{node.lineno}:{i}"] = rendered
    texts.update({f"decision:{d.value}": f"decided {d.value}" for d in Decision})
    assert "handed to a teleoperator" in texts.values()
    assert {"gripper commanded open", "gripper commanded closed"} <= set(texts.values())
    for producer in (
        events._drop_events,
        events._stall_events,
        events._collision_events,
        events._near_miss_event,
        events.scene_contact_events,
    ):
        for node in ast.walk(ast.parse(inspect.getsource(producer))):
            if isinstance(node, ast.JoinedStr):
                for i, rendered in enumerate(event_texts(node)):
                    texts[f"detector:{producer.__name__}:{node.lineno}:{i}"] = rendered
    report = audit_export(out, [run], texts=texts)
    assert report.clean, report.summary()


def test_bookkeeping_is_excluded_and_future_truth_names_come_from_code(tmp_path: Path) -> None:
    root = tmp_path / "w01"
    run = root / "runs/w01-observed"
    write_toy_run(run)
    (root / "files").mkdir()
    (root / "files/spec.json").write_text("{}")
    before = set(leaks.truth_terms([run]))
    (root / "runner.json").write_text('{"runner_bookkeeping_added": 1, "receipt": "x"}')
    (run / "run.json").write_text('{"engine_bookkeeping_added": 2}')
    assert set(leaks.truth_terms([run])) == before
    assert {"jobs_sha256", "fix_sets", "as_run", "all_removed_success"} <= set(
        leaks.hidden_field_terms()
    )
    assert {
        "observed_carriers",
        "detectable_causes",
        "patterns_present",
        "expected_for_them",
        "confirmed_by_the_oracle",
        "qualified_by_observed",
        "expected_effect_points",
        "counts_for_gate",
        "without_any_fault",
    } <= set(leaks.hidden_field_terms())


def test_discovered_single_word_keys_are_names_not_prose(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_toy_run(run)
    (run / "truth/new.json").write_text('{"by": 1, "teleoperator": 2, "bookkeeper": 3}')
    out = tmp_path / "out"
    out.mkdir()
    assert audit_export(out, [run], texts={"ordinary": "receipt signed by a teleoperator"}).clean
    (out / "values.json").write_text('{"by": 2}')
    (out / "bookkeeper.txt").write_text("ordinary")
    pq.write_table(pa.table({"teleoperator": [2]}), out / "values.parquet")
    (out / "label.txt").write_text("by=2")
    assert {"by", "teleoperator", "bookkeeper"} <= {h.term for h in audit_export(out, [run]).hits}


def test_role_words_are_prose_terms_and_ordinary_keys_are_names(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_toy_run(run)
    (run / "truth/roles.json").write_text(
        '{"drift": 1, "level": 2, "outcome": 3, "interaction": 4}'
    )
    out = tmp_path / "export"
    out.mkdir()
    assert audit_export(
        out,
        [run],
        texts={"ordinary": "drift away; liquid level low; outcome success; interaction; verdicts"},
    ).clean
    words = (
        "decoy",
        "inert",
        "confounder",
        "superadditive",
        "additive",
        "detectable",
        "planted",
        "injected",
        "counterfactual",
        "oracle",
        "verdict",
    )
    report = audit_export(out, [run], texts={word: f"this is {word}" for word in words})
    assert set(words) <= {h.where for h in report.hits}
    for key in ("drift", "level", "interaction"):
        (out / "values.json").write_text(json.dumps({key: 2}))
        assert key in {h.term for h in audit_export(out, [run]).hits}
    (out / "values.json").write_text('{"outcome": "success"}')
    assert audit_export(out, [run]).clean  # outcome is also a visible contract field.


def test_separator_names_have_both_boundaries(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text(
        'has run; direction "degraded"; prefixGripperPadAgeDaysSuffix'
    )
    assert leaks.leak_search(tmp_path, ["as_run", "direction_deg", "gripper_pad_age_days"]).clean
    (tmp_path / "note.txt").write_text("asRun; directionDeg; gripper-pad-age-days")
    assert (
        len(leaks.leak_search(tmp_path, ["as_run", "direction_deg", "gripper_pad_age_days"]).hits)
        == 3
    )


def test_main_literal_terms_keep_plural_and_digit_suffix_matches(tmp_path: Path) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "data/export-main-cfedf66-terms.json").read_text()
    )
    assert len(fixture["terms"]) == 84
    texts = {
        f"{term}:{suffix}": term + suffix
        for term in fixture["terms"]
        for suffix in ("s", "es", "2")
    }
    expected = {(where, where.split(":")[0]) for where in texts}
    report = leaks.leak_search(tmp_path, fixture["terms"], texts=texts)
    assert expected <= {(h.where, h.term) for h in report.hits}
    # Literal underscores retain main's substring behavior; joined forms do not.
    assert leaks.leak_search(
        tmp_path, ["gripper_pad_age_days"], texts={"prefix": "xgripper_pad_age_days2"}
    ).hits


def test_declared_public_fields_are_public_even_when_also_hidden_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import Field, create_model

    from fleetlab.contracts import tools

    public = create_model("PublicProbe", scene_seed=(int, 0))
    monkeypatch.setattr(tools, "PublicProbe", public, raising=False)
    monkeypatch.setattr(tools, "__all__", [*tools.__all__, "PublicProbe"])
    assert "scene_seed" not in leaks.hidden_field_terms()
    run = tmp_path / "run"
    write_toy_run(run)
    assert "scene_seed" not in leaks._file_truth_names([run])
    assert "scene_seed" not in leaks.truth_terms([run])
    assert "scene_seed" not in leaks.hidden_column_terms()
    alias = create_model("PublicAlias", seed_value=(int, Field(default=0, alias="sceneSeed")))
    monkeypatch.setattr(tools, "PublicProbe", alias)
    assert "scene_seed" not in leaks.hidden_column_terms()
    assert "scene_seed" not in leaks.hidden_field_terms()


def test_checker_source_changes_the_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.contracts import _base

    source = tmp_path / "base.py"
    source.write_bytes(Path(_base.__file__).read_bytes())
    monkeypatch.setattr(_base, "__file__", str(source))
    before = leaks.audit_identifier(())
    with source.open("a") as handle:
        handle.write("\n# a future checker dependency change\n")
    assert leaks.audit_identifier(()) != before


def test_undecodable_column_is_a_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Opaque(pa.ExtensionType):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(pa.int32(), "test.opaque")

        def __arrow_ext_serialize__(self) -> bytes:
            return b""

        @classmethod
        def __arrow_ext_deserialize__(
            cls, storage_type: pa.DataType, serialized: bytes
        ) -> "Opaque":
            return cls()

    path = tmp_path / "values.parquet"
    pq.write_table(pa.table({"x": [0]}), path)
    data = pa.table({"x": pa.ExtensionArray.from_storage(Opaque(), pa.array([0], pa.int32()))})
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: data)
    assert any(h.term == "(undecodable column)" for h in leaks.leak_search(tmp_path, []).hits)


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_seeds_in_floating_columns_and_nested_values(tmp_path: Path, dtype: pa.DataType) -> None:
    seed = 3858582668
    noise = 6247533203330922350
    pq.write_table(
        pa.table(
            {
                "duration_s": pa.array([float(seed), float(noise)], dtype),
                "nested": pa.array([[float(seed)], [float(noise)]], pa.list_(dtype)),
            }
        ),
        tmp_path / "values.parquet",
    )
    (tmp_path / "note.json").write_text(json.dumps({"value": float(seed), "another": float(noise)}))
    hits = leaks.leak_search(tmp_path, [str(seed), str(noise)], seeds=[seed, noise]).hits
    assert {str(seed), str(noise)} <= {h.term for h in hits}
    assert {
        "value of values.parquet.duration_s",
        "value of values.parquet.nested row 0 at /0",
        "note.json at /value",
    } <= {h.where for h in hits}
