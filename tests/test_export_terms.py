"""Matching policy against code definitions and frozen, actual agent inputs."""

import ast
import gzip
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import write_toy_run

from fleetlab.contracts.model import MODEL_CLASSES, Fleet
from fleetlab.log import leaks


def _agent_texts(root: Path) -> dict[str, str]:
    """Supply only explicitly authored state text, with source filenames."""
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".md", ".json", ".jsonl"}
    }


@pytest.fixture(scope="module")
def real_inputs() -> dict[str, Any]:
    path = Path(__file__).parent / "data/export-skills-e0d1954.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))  # type: ignore[no-any-return]


@pytest.mark.parametrize("plain", [False, True])
def test_real_w42_terms_accept_the_batch4_live_launch_texts(
    tmp_path: Path, real_inputs: dict[str, Any], plain: bool
) -> None:
    fixture = json.loads(
        gzip.decompress(
            (Path(__file__).parent / "data/export-supervisor-batch4.json.gz").read_bytes()
        )
    )
    sources = fixture["sources"]
    for path, source in sources.items():
        assert hashlib.sha256(source.encode()).hexdigest() == fixture["sha256"][path]
    registry = ast.parse(sources["src/fleetlab/skill_library.py"])
    roles = next(
        node
        for node in registry.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SUPERVISOR_SKILLS"
    )
    assert roles.value is not None
    skills = ast.literal_eval(roles.value)
    for name in skills:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(sources[f"skills/{name}/SKILL.md"])

    def skill_dirs(role: str) -> list[Path]:
        assert role == "supervisor"
        return [tmp_path / name for name in skills]

    # Build the real instructions and the gate's texts= expression, without
    # importing the live engine or invoking Batch.after_gate/model sessions.
    supervisor = ast.parse(sources["src/fleetlab/live/vlm/supervisor.py"])
    instructions = next(
        node
        for node in supervisor.body
        if isinstance(node, ast.FunctionDef) and node.name == "instructions"
    )
    namespace: dict[str, Any] = {"skill_dirs": skill_dirs}
    exec(compile(ast.Module(body=[instructions], type_ignores=[]), "supervisor", "exec"), namespace)
    measure = ast.parse(sources["src/fleetlab/live/vlm/measure.py"])
    gate = next(
        node for node in measure.body if isinstance(node, ast.FunctionDef) and node.name == "gate"
    )
    call = next(
        node
        for node in ast.walk(gate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "after_gate"
    )
    expression = next(keyword.value for keyword in call.keywords if keyword.arg == "texts")
    texts = eval(compile(ast.Expression(expression), "live gate texts", "eval"), namespace)
    assert all(word in texts["system"] for word in ("attempts", "frames", "ratio", "verdicts"))
    world = real_inputs["worlds"]["w42"]
    terms = leaks.classify_truth_terms(
        world["terms"],
        fleet=Fleet.model_validate(world["fleet"]),
        kinds=world["kinds"],
        fault_ids=world["fault_ids"],
        identity_names=world["identity_names"],
    )
    assert len(terms) > 3900
    report = leaks.search_texts(
        list(terms) if plain else terms,
        texts,
        kinds=world["kinds"],
        fault_ids=world["fault_ids"],
        seeds=world["seeds"],
    )
    assert report.clean, report.summary()
    # Check each skill on its own too; the system builder includes them both.
    report = leaks.search_texts(
        terms,
        {path: source for path, source in sources.items() if path.startswith("skills/")},
    )
    assert report.clean, report.summary()


@pytest.mark.parametrize("plain", [False, True])
def test_ordinary_single_words_are_name_only_whatever_their_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain: bool
) -> None:
    from fleetlab.worlds import world

    def future_record() -> dict[str, int]:
        return {"recordmarker": 2}

    monkeypatch.setattr(world, "generator_record", future_record)
    fleet = write_toy_run(tmp_path / "run")
    fleet.robots()[0].traits["propertymarker"] = 2
    names = {"recordmarker", "propertymarker", "kindmarker", "unknownmarker", "verdicts"}
    terms = leaks.classify_truth_terms(names, fleet=fleet, kinds=["kindmarker"])
    assert "recordmarker" in terms.structural
    assert {"propertymarker", "kindmarker", "unknownmarker"} <= terms.planted
    vocabulary = list(terms) if plain else terms
    assert leaks.search_texts(
        vocabulary,
        {"system": "We describe " + " and ".join(names) + ": these are ordinary words."},
        kinds=["kindmarker"],
    ).clean
    for word in names:
        for text in (
            f"{word}=2",
            f"{word}: 2",
            f'{{"{word}": 2}}',
            f"read `{word}`",
            f"obj.{word}",
        ):
            report = leaks.search_texts(vocabulary, {"system": text}, kinds=["kindmarker"])
            assert word in {hit.term for hit in report.hits}, (word, text)
        search = leaks._Search(vocabulary, surface="ours", kinds=["kindmarker"])
        for label in ("column", "file", "map key"):
            search.scan_text(label, word, name=True)
        assert len(search.report.hits) == 3
    # Our policy change cannot enable a single-word tripwire on agent text.
    assert leaks.search_texts(
        vocabulary, agent_texts={"notes": json.dumps(dict.fromkeys(names, 2))}
    ).clean


def test_real_ours_prose_still_refuses_deliberately_listed_role_words(
    real_inputs: dict[str, Any],
) -> None:
    world = real_inputs["worlds"]["w42"]
    terms = leaks.classify_truth_terms(world["terms"], fleet=Fleet.model_validate(world["fleet"]))
    for word in leaks._PROSE_TRUTH_NAMES & set(terms):
        report = leaks.search_texts(terms, {"system": f"A {word} explains the behavior."})
        assert word in {hit.term for hit in report.hits}, word


def test_structural_keys_use_exact_names_from_the_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.worlds import world

    run = tmp_path / "run"
    write_toy_run(run)
    (run / "truth/new.json").write_text('{"new_queue_s": 1, "new_count": 2}')

    def future_record() -> dict[str, int]:
        return {"new_queue_s": 1, "new_count": 2}

    monkeypatch.setattr(world, "generator_record", future_record)
    terms = leaks.truth_terms([run])
    # Classification depends on the defining contract/producer, not a list of
    # queue_s or today's other false alarms. The producer establishes structure.
    assert {"new_queue_s", "new_count"} <= terms.structural
    record_names = leaks._record_keys(world.generator_record) & set(terms)
    record_names -= leaks._planted_field_names() | leaks._PROSE_TRUTH_NAMES
    assert record_names <= terms.structural
    out = tmp_path / "out"
    out.mkdir()
    prose = "new queues an experiment; newQueueS; new queue s; a new count; new_count"
    assert leaks.leak_search(out, terms, texts={"prose": prose}).clean
    for surface in ("json", "column", "file", "label"):
        path = out / "values.txt"
        if surface == "json":
            path = out / "values.json"
            path.write_text('{"new_queue_s": 2}')
        elif surface == "column":
            path = out / "values.parquet"
            pq.write_table(pa.table({"new_queue_s": [2]}), path)
        elif surface == "file":
            path = out / "new_queue_s.txt"
            path.write_text("ordinary")
        else:
            path.write_text("new_queue_s=2")
        assert "new_queue_s" in {h.term for h in leaks.leak_search(out, terms).hits}
        path.unlink()


def test_public_answer_schema_changes_are_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import sessions

    class FutureAnswer(sessions.SessionReport):
        future_answer_field: str = ""

    run = tmp_path / "run"
    write_toy_run(run)
    (run / "truth/new.json").write_text('{"future_answer_field": 2, "summary": 1}')
    monkeypatch.setattr(sessions, "SessionReport", FutureAnswer)
    terms = leaks.truth_terms([run])
    assert not {"summary", "future_answer_field"} & set(terms)


def test_untyped_public_reply_fields_come_from_the_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.toolserver.server import ToolBox

    def future_query(self: object, args: object) -> dict[str, int]:
        return {"future_reply_field": 1}

    run = tmp_path / "run"
    write_toy_run(run)
    (run / "truth/new.json").write_text('{"future_reply_field": 2}')
    monkeypatch.setattr(ToolBox, "_query", future_query)
    assert "future_reply_field" not in leaks.truth_terms([run])


def test_checker_identifier_hashes_classification_without_changing_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = leaks.truth_terms()
    identifier = leaks.audit_identifier(())
    original = leaks._record_structure_names
    monkeypatch.setattr(leaks, "_record_structure_names", lambda: original() | {"worn_robots"})
    after = leaks.truth_terms()
    assert before == after
    assert "worn_robots" not in before.structural
    assert "worn_robots" in after.structural
    assert leaks.audit_identifier(()) != identifier


def test_public_schema_changes_invalidate_receipts_without_static_term_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.agents import sessions

    before = leaks.truth_terms()
    identifier = leaks.audit_identifier(())

    class FutureAnswer(sessions.SessionReport):
        fresh_public_record_field: str = ""

    monkeypatch.setattr(sessions, "SessionReport", FutureAnswer)
    assert before == leaks.truth_terms()
    assert leaks.audit_identifier(()) != identifier


def test_task_identity_names_allow_public_prose_but_remain_names(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_toy_run(run)
    path = run / "truth/identities.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("suite"), "suite", pa.array(["new_task_suite"] * len(table))
    )
    pq.write_table(table, path)
    terms = leaks.truth_terms([run])
    assert "new_task_suite" in terms.structural
    out = tmp_path / "out"
    out.mkdir()
    assert leaks.leak_search(out, terms, {"public task": "task new_task_suite 0"}).clean
    assert leaks.leak_search(out, terms, {"prose": "new task suite"}).clean
    (out / "plant.json").write_text('{"new_task_suite": 2}')
    assert not leaks.leak_search(out, terms).clean


def test_description_rejects_undeclared_fields_without_secret_word_matching(tmp_path: Path) -> None:
    from fleetlab.contracts.model import describe_for_agents

    run = tmp_path / "run"
    fleet = write_toy_run(run)
    fleet.robots()[0].components[next(iter(fleet.robots()[0].components))].health[
        "future_public_sensor"
    ] = 0.3
    out = tmp_path / "out"
    out.mkdir()
    (out / "log").mkdir()
    path = out / "log/description.json"
    description = describe_for_agents(fleet, task_classes=["task libero_spatial 0"])
    terms = leaks.truth_terms([run], fleet=fleet)
    path.write_text(json.dumps(description))
    assert leaks.leak_search(out, terms).clean
    description["fleet"]["sites"][0]["new_private_block"] = {"ordinary": 41}
    path.write_text(json.dumps(description))
    report = leaks.leak_search(out, terms)
    assert "new_private_block" not in terms
    assert "new_private_block" in {h.term for h in report.hits}


def test_agent_description_filename_does_not_acquire_export_schema_rules(tmp_path: Path) -> None:
    terms = leaks.truth_terms()
    path = tmp_path / "description.json"
    path.write_text('{"summary": "ordinary"}')
    assert leaks.leak_search(tmp_path, terms).clean
    path.write_text('{"summary": "gripper-wear"}')
    assert "gripper_wear" in {h.term for h in leaks.leak_search(tmp_path, terms).hits}


def test_structural_alias_keeps_its_defined_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.worlds import world

    def future_record() -> dict[str, int]:
        return {"newElapsedMS": 2}

    monkeypatch.setattr(world, "generator_record", future_record)
    terms = leaks.classify_truth_terms(["newElapsedMS"])
    (tmp_path / "name.json").write_text('{"NEWELAPSEDMS": 2}')
    assert leaks.leak_search(tmp_path, terms).hits
    (tmp_path / "name.json").unlink()
    assert leaks.leak_search(tmp_path, terms, {"prose": "new elapsed ms=2; new_elapsed_ms=2"}).clean


LABELS = ("plain", "plural", "digit", "prefix", "sentence", "upper", "key")


def test_main_588_variants_keep_the_same_576_detections(tmp_path: Path) -> None:
    original = json.loads(
        (Path(__file__).parent / "data/export-main-cfedf66-terms.json").read_text()
    )["terms"]
    texts = {
        f"{term}|{label}": value
        for term in original
        for label, value in (
            ("plain", term),
            ("plural", term + "s"),
            ("digit", term + "2"),
            ("prefix", "x" + term),
            ("sentence", f"the {term}s were logged"),
            ("upper", term.upper()),
            ("key", f'{{"{term}s": 1}}'),
        )
    }
    assert len(texts) == 588
    report = leaks.leak_search(tmp_path, leaks.truth_terms(), texts)
    missed = set(texts) - {hit.where for hit in report.hits}
    # Since 2026-10-19 max_steps is a public word of the job vocabulary (JobStep.max_steps).
    public = {f"max_steps|{label}" for label in LABELS}
    # main's left word boundary rejects only these twelve alphabetic prefixes.
    assert missed == {f"{term}|prefix" for term in original if term.isalpha()} | public
    assert len(texts) - len(missed) == 569


@pytest.mark.parametrize("name", ["w41", "w42", "w43"])
def test_real_terms_accept_frozen_pipeline_schema_and_both_libraries(
    tmp_path: Path, real_inputs: dict[str, Any], name: str
) -> None:
    world = real_inputs["worlds"][name]
    fleet = Fleet.model_validate(world["fleet"])
    terms = leaks.classify_truth_terms(
        world["terms"],
        fleet=fleet,
        fault_ids=world["fault_ids"],
        identity_names=world["identity_names"],
    )
    assert len(world["terms"]) > 3900  # actual world, not the 537-term toy vocabulary
    assert {"queue_s", "by_day", "healthy_robots", "needed_for"} <= terms.structural
    assert "summary" not in terms
    for texts in world["texts"].values():
        report = leaks.leak_search(
            tmp_path, terms, texts, seeds=world["seeds"], fault_ids=world["fault_ids"]
        )
        assert report.clean, report.summary()
    # source_audit supplies mounted files as code. Preserve that narrower
    # policy: generic contract vocabulary is allowed, planted mechanisms are not.
    protected = set(world["kinds"]) | set(world["traits"]) | set(world["fault_ids"])
    protected.update(map(str, world["seeds"]))
    protected.add(world["canary"])
    protected -= leaks._known_code_exceptions()
    protected.discard("traits")
    for files in [real_inputs["bundle"], *real_inputs["libraries"].values()]:
        search = leaks._Search(
            protected, seeds=world["seeds"], fault_ids=world["fault_ids"], surface="ours"
        )
        for where, text in files.items():
            search.scan_text(where, text, mask=False)
        assert search.report.clean, search.report.summary()
    for value in (
        world["canary"],
        world["kinds"][0],
        world["traits"][0],
        world["fault_ids"][0],
        str(world["seeds"][0]),
    ):
        report = leaks.leak_search(
            tmp_path, terms, {"plant": value}, seeds=world["seeds"], fault_ids=world["fault_ids"]
        )
        assert not report.clean, value


def test_frozen_agent_notes_allow_natural_kind_forms_but_our_text_does_not(
    tmp_path: Path, real_inputs: dict[str, Any]
) -> None:
    """Use the unchanged notes that required the surface-policy decision."""
    assert len(real_inputs["sessions"]) == 26
    expected = {
        "20261009-r09/w41-logs-1/run.json",
        "20261009-r13/w42-logs-1/run.json",
    }
    caught = set()
    for session in real_inputs["sessions"]:
        for surface, files in session["files"].items():
            for name, text in files.items():
                assert hashlib.sha256(text.encode()).hexdigest() == session["sha256"][surface][name]
        world = real_inputs["worlds"][session["world"]]
        fleet = Fleet.model_validate(world["fleet"])
        terms = leaks.classify_truth_terms(world["terms"], fleet=fleet)
        # These are unchanged agent-authored files, not injected test strings.
        report = leaks.leak_search(tmp_path, terms, session["files"]["memory"])
        if any(h.term == "gripper_wear" for h in report.hits):
            caught.add(session["source"].split("/skills/")[1])
        report = leaks.search_texts(terms, agent_texts=session["files"]["memory"])
        assert not any(h.term == "gripper_wear" for h in report.hits)
    assert caught == expected


def test_internal_search_requires_a_surface() -> None:
    function: Any = leaks._Search
    with pytest.raises(TypeError, match="surface"):
        function([])
    with pytest.raises(ValueError, match="surface"):
        function([], surface="unknown")


@pytest.mark.parametrize("function", ["leak_search", "audit_export"])
def test_tree_scanners_have_no_surface_selector(tmp_path: Path, function: str) -> None:
    with pytest.raises(TypeError, match="surface"):
        getattr(leaks, function)(tmp_path, [], surface="agent")


@pytest.mark.parametrize(
    "natural",
    [
        "gripper wear",
        "gripper-wear",
        "gripperwear",
        "GRIPPERWEAR",
        "calibration update",
        "calibration-update",
        "calibrationupdate",
        "CALIBRATIONUPDATE",
        "gripper pad age days",
        "gripper-pad-age-days",
        "gripperpadagedays",
    ],
)
def test_agent_natural_forms_pass(tmp_path: Path, natural: str) -> None:
    terms = leaks.truth_terms()
    report = leaks.search_texts(terms, agent_texts={"note": natural})
    assert report.clean, report.summary()


@pytest.mark.parametrize("surface", ["ours", "agent"])
@pytest.mark.parametrize(
    "identifier",
    [
        "gripper_wear",
        "gripperWear",
        "GripperWear",
        "GRIPPER_WEAR",
        "gripper.wear",
        "calibration_update",
        "calibrationUpdate",
        "CALIBRATION_UPDATE",
        "gripper_pad_age_days",
        "gripperPadAgeDays",
        "GripperPadAgeDays",
        "gripper.pad_age.days",
    ],
)
def test_both_surfaces_refuse_planted_identifiers(
    tmp_path: Path, surface: Any, identifier: str
) -> None:
    texts = {"note": identifier}
    report = (
        leaks.search_texts(leaks.truth_terms(), agent_texts=texts)
        if surface == "agent"
        else leaks.search_texts(leaks.truth_terms(), texts)
    )
    assert not report.clean, identifier


@pytest.mark.parametrize("name", ["w41", "w42", "w43"])
@pytest.mark.parametrize("surface", ["ours", "agent"])
def test_real_world_protected_probes_on_both_surfaces(
    tmp_path: Path, real_inputs: dict[str, Any], name: str, surface: Any
) -> None:
    world = real_inputs["worlds"][name]
    terms = leaks.classify_truth_terms(
        world["terms"],
        fleet=Fleet.model_validate(world["fleet"]),
        fault_ids=world["fault_ids"],
        identity_names=world["identity_names"],
    )
    assert len(terms) > 3900
    for value in (
        world["canary"],
        world["kinds"][0],
        world["traits"][0],
        "x9e52a18c" if surface == "agent" else world["fault_ids"][0],
        str(world["seeds"][0]),
    ):
        report = leaks.search_texts(
            terms,
            seeds=world["seeds"],
            fault_ids=[*world["fault_ids"], "x9e52a18c"],
            **{"agent_texts" if surface == "agent" else "texts": {"probe": value}},
        )
        assert not report.clean, value


def test_agent_case_is_preserved_in_files_json_names_and_binary(tmp_path: Path) -> None:
    terms = leaks.truth_terms()
    (tmp_path / "notes.md").write_text("gripperwear and gripper-wear")
    assert leaks.search_texts(terms, agent_texts=_agent_texts(tmp_path)).clean
    (tmp_path / "notes.md").write_text("gripperWear")
    (tmp_path / "answer.json").write_text('{"rationale": "gripperWear"}')
    (tmp_path / "gripper_wear.txt").write_text("ordinary")
    search = leaks._Search(terms, surface="agent")
    search.scan_bytes("opaque bytes", b"\x00 gripperWear \x00")
    assert "gripper_wear" in {h.term for h in search.report.hits}
    report = leaks.search_texts(terms, agent_texts=_agent_texts(tmp_path))
    assert {"notes.md", "answer.json at /rationale"} <= {
        h.where for h in report.hits if h.term == "gripper_wear"
    }


def test_agent_fault_id_overrides_generic_word_label_policy(tmp_path: Path) -> None:
    report = leaks.search_texts(
        ["fault"], fault_ids=["fault"], agent_texts={"note": "the fault is recorded"}
    )
    assert "fault" in {h.term for h in report.hits}


@pytest.mark.parametrize("fault_id", ["f01", "F14", "a00", "z99"])
def test_countable_fault_ids_pass_agent_text_only(fault_id: str) -> None:
    # IDs belong to the actual world, even when also present in its vocabulary.
    terms = leaks.classify_truth_terms([fault_id], fault_ids=[fault_id])
    report = leaks.search_texts(
        terms,
        agent_texts={
            "notes.md": f"{fault_id} {fault_id}_level cause_{fault_id} of{fault_id}",
            "answer.json": json.dumps({fault_id: fault_id}),
        },
        fault_ids=[fault_id],
    )
    assert report.clean, report.summary()
    assert leaks.search_texts(terms, {"prompt": fault_id}, fault_ids=[fault_id]).hits


@pytest.mark.parametrize("fault_id", ["f001", "f1", "ff01", "1f01", "x9e52a18c"])
def test_noncountable_fault_ids_remain_independent_agent_tripwires(fault_id: str) -> None:
    # No term vocabulary is needed, and an ordinary alias is no exemption.
    report = leaks.search_texts(
        [],
        agent_texts={"query": f"SELECT ok AS cause_{fault_id}_level"},
        fault_ids=[fault_id],
    )
    assert {h.term for h in report.hits} == {fault_id}


@pytest.mark.parametrize("fault_id", ["name", "sampling_steps", "publicWord"])
def test_agent_fault_id_overrides_public_names_even_without_a_term(
    tmp_path: Path, fault_id: str
) -> None:
    report = leaks.search_texts([], fault_ids=[fault_id], agent_texts={"note": fault_id})
    assert {h.term for h in report.hits} == {fault_id}


def test_agent_float_seed_and_literal_hidden_json_key(tmp_path: Path) -> None:
    seed = 3858582668
    (tmp_path / "answer.json").write_text(
        json.dumps({"rationale": {"gripper_pad_age_days": 3, "values": [float(seed)]}})
    )
    report = leaks.search_texts(
        leaks.truth_terms(), seeds=[seed], agent_texts=_agent_texts(tmp_path)
    )
    assert {"gripper_pad_age_days", str(seed)} <= {h.term for h in report.hits}


@pytest.mark.parametrize("name", ["w41", "w42", "w43"])
def test_agent_single_words_and_record_structure_do_not_refuse(
    tmp_path: Path, real_inputs: dict[str, Any], name: str
) -> None:
    world = real_inputs["worlds"][name]
    terms = leaks.classify_truth_terms(world["terms"], fleet=Fleet.model_validate(world["fleet"]))
    words = {t for t in terms if t.isalpha()} | set(terms.structural)
    text = "nominal success rate; additive effects; a detectable difference; expected gain"
    (tmp_path / "answer.json").write_text(json.dumps(dict.fromkeys(words, text)))
    (tmp_path / "queue_s.txt").write_text("summary=1; nominal=2; mode: normal")
    pq.write_table(pa.table({w: [text] for w in words}), tmp_path / "notes.parquet")
    search = leaks._Search(terms, surface="agent")
    search.scan_bytes("bytes", " ".join(words).encode())
    assert search.report.clean, search.report.summary()
    report = leaks.search_texts(terms, agent_texts=_agent_texts(tmp_path))
    assert report.clean, report.summary()


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("name", ["w41", "w42", "w43"])
def test_agent_role_word_aliases_pass_with_real_world_terms(
    real_inputs: dict[str, Any], name: str, plain: bool
) -> None:
    world = real_inputs["worlds"][name]
    terms = leaks.classify_truth_terms(
        world["terms"], fleet=Fleet.model_validate(world["fleet"]), kinds=world["kinds"]
    )
    vocabulary = list(terms) if plain else terms
    aliases = ["min_detectable", "n_additive", "is_nominal"]
    report = leaks.search_texts(
        vocabulary,
        agent_texts={
            "request /query": "SELECT 2 AS min_detectable, 1 AS n_additive, 1 AS is_nominal",
            "answer.json": json.dumps(dict.fromkeys(aliases, "nominal additive detectable")),
        },
        fault_ids=world["fault_ids"],
        seeds=world["seeds"],
    )
    assert report.clean, report.summary()
    for identifier in ("illumination_change_level", "gripper_pad_age_days_mean"):
        report = leaks.search_texts(vocabulary, agent_texts={"request /query": identifier})
        assert not report.clean, identifier


def test_agent_single_words_never_become_multipart_through_empty_parts() -> None:
    terms = leaks.classify_truth_terms(
        ["nominal", "_nominal", "nominal_", ".additive", "detectable."]
    )
    report = leaks.search_texts(
        terms, agent_texts={"note": "nominal _nominal nominal_ n_additive min_detectable.rate"}
    )
    assert report.clean, report.summary()


def test_all_26_saved_states_pass_without_caller_word_masks(real_inputs: dict[str, Any]) -> None:
    worlds = {
        name: leaks.classify_truth_terms(
            world["terms"],
            fleet=Fleet.model_validate(world["fleet"]),
            kinds=world["kinds"],
            fault_ids=world["fault_ids"],
            identity_names=world["identity_names"],
        )
        for name, world in real_inputs["worlds"].items()
    }
    assert len(real_inputs["sessions"]) == 26
    for session in real_inputs["sessions"]:
        texts = {}
        for area, files in session["files"].items():
            for name, text in files.items():
                assert hashlib.sha256(text.encode()).hexdigest() == session["sha256"][area][name]
                texts[f"{area}/{name}"] = text
        world = real_inputs["worlds"][session["world"]]
        report = leaks.search_texts(
            worlds[session["world"]],
            agent_texts=texts,
            seeds=world["seeds"],
            fault_ids=world["fault_ids"],
        )
        assert report.clean, f"{session['source']}: {report.summary()}"


def test_agent_planted_provenance_and_public_fields(tmp_path: Path) -> None:
    from fleetlab.contracts.model import Fleet

    fleet = write_toy_run(tmp_path / "run")
    fleet.robots()[0].traits["future_hidden_trait"] = 2
    fleet.robots()[0].components[next(iter(fleet.robots()[0].components))].health[
        "future_public_sensor"
    ] = 3
    terms = leaks.classify_truth_terms(
        ["future_hidden_trait", "future_report_key", "future_public_sensor", "private_extra_key"],
        fleet=Fleet.model_validate(fleet.model_dump()),
        extra=["private_extra_key"],
    )
    assert terms.planted == {"future_hidden_trait", "future_report_key", "private_extra_key"}
    assert "future_report_key" not in terms.structural
    assert "future_public_sensor" not in terms
    search = leaks._Search(terms, surface="agent")
    search.scan_json("answer", dict.fromkeys(terms, "future public sensor"))
    assert {h.term for h in search.report.hits} == set(terms)
    # The visible vocabulary names control.sampling_steps; its leaf spelling
    # remains public in an agent's notes even though Job also uses the field.
    report = leaks.search_texts(leaks.truth_terms(), agent_texts={"note": "sampling_steps=12"})
    assert not any(h.term == "sampling_steps" for h in report.hits)


def test_frozen_state_reader_accepts_all_26_saved_sessions(
    tmp_path: Path, real_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.log as log

    frozen = real_inputs["state_reader"]
    assert hashlib.sha256(frozen["source"].encode()).hexdigest() == frozen["sha256"]
    module = types.ModuleType("frozen_export_state_reader")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(frozen["source"], module.__dict__)
    current: dict[str, Any] = {}

    def state_texts(root: Path, runs: object, *, fleet: Fleet) -> leaks.LeakReport:
        # Adapt only the old caller's API: explicitly supply agent state texts.
        # Its JSON prose parsing/public alias masks remain the unchanged e0d1954 code.
        return leaks.search_texts(
            current["terms"],
            agent_texts=_agent_texts(root),
            seeds=current["world"]["seeds"],
            fault_ids=current["world"]["fault_ids"],
        )

    monkeypatch.setattr(log, "audit_export", state_texts)
    refused = {}
    assert len(real_inputs["sessions"]) == 26
    for i, session in enumerate(real_inputs["sessions"]):
        world = real_inputs["worlds"][session["world"]]
        terms = leaks.classify_truth_terms(
            world["terms"],
            fleet=Fleet.model_validate(world["fleet"]),
            kinds=world["kinds"],
            fault_ids=world["fault_ids"],
            identity_names=world["identity_names"],
        )
        root = tmp_path / str(i)
        root.mkdir()
        for area, files in session["files"].items():
            for name, text in files.items():
                assert hashlib.sha256(text.encode()).hexdigest() == session["sha256"][area][name]
                path = root / area / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        current.update(terms=terms, world=world)
        hits = []
        for area in session["files"]:
            report = module.audit_state(root / area, (), Fleet.model_validate(world["fleet"]), area)
            hits.extend(report.hits)
        if hits:
            refused[session["source"].split("/skills/")[1]] = {h.term for h in hits}
    assert refused == {}


@pytest.mark.parametrize("plain", [False, True])
def test_agent_execution_record_names_never_refuse(tmp_path: Path, plain: bool) -> None:
    terms = leaks.truth_terms()
    record_names = leaks._planted_field_names() - leaks._agent_planted_names(
        None, leaks._registry_kinds()
    )
    # Since 2026-10-19 max_steps is a public word of the job vocabulary (JobStep.max_steps).
    assert "max_steps" not in record_names
    assert {"true_success", "image_size", "onset_day"} <= record_names
    names = {t for t in terms if t in record_names}
    vocabulary = list(terms) if plain else terms
    values = "max_steps maxSteps max.steps true_success trueSuccess true.success"
    (tmp_path / "max_steps.txt").write_text(values)
    (tmp_path / "answer.json").write_text(json.dumps(dict.fromkeys(names, values)))
    pq.write_table(pa.table({n: [values] for n in names}), tmp_path / "notes.parquet")
    report = leaks.search_texts(vocabulary, agent_texts=_agent_texts(tmp_path))
    assert report.clean, report.summary()
    # Our own prose still forbids these truth-side execution/outcome identifiers.
    report = leaks.leak_search(tmp_path, terms, {"prose": values})
    assert "true_success" in {h.term for h in report.hits}


def test_agent_hidden_fleet_attributes_and_pattern_roles_follow_their_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Annotated, Literal

    from pydantic import Field

    from fleetlab.contracts._base import HIDDEN
    from fleetlab.contracts.model import Robot
    from fleetlab.worlds import truth

    class FutureRobot(Robot):
        latent_motor_bias: Annotated[float, HIDDEN] = Field(default=0)

    monkeypatch.setattr(leaks, "MODEL_CLASSES", (*MODEL_CLASSES, FutureRobot))
    monkeypatch.setattr(truth, "Role", Literal["future_pattern_role"])
    terms = leaks.classify_truth_terms(
        ["latent_motor_bias", "future_pattern_role", "future_record_layout"]
    )
    assert terms.planted == {"latent_motor_bias", "future_pattern_role", "future_record_layout"}
    report = leaks.search_texts(
        terms, agent_texts={"note": "latentMotorBias future.pattern_role future_record_layout"}
    )
    assert {h.term for h in report.hits} == set(terms)


@pytest.mark.parametrize(
    "identifier",
    [
        "illumination_change_level = 2.0",
        "SELECT ok AS illumination_change_rate",
        "kind_illumination_change",
        "illumination_changes: 2",
        "illumination_change2",
        "kindIlluminationChangeRate",
        "illuminationChangeMean",
        '{"object_batch_variant_level": 6.0}',
        "gripper_pad_age_days_mean",
        "mean_gripperPadAgeDays2",
        "x9e52a18c_level",
        "cause_x9e52a18c",
    ],
)
def test_agent_identifiers_trip_inside_longer_identifiers(identifier: str) -> None:
    terms = leaks.truth_terms()
    report = leaks.search_texts(
        terms, agent_texts={"answer /query": identifier}, fault_ids=["x9e52a18c"]
    )
    assert not report.clean, identifier


@pytest.mark.parametrize(
    "notation", ["3,858,582,668", "3_858_582_668", hex(3858582668), "3.858582668e9"]
)
def test_seeds_refuse_in_text_notation_on_both_authorships(notation: str) -> None:
    for source in ("texts", "agent_texts"):
        report = leaks.search_texts([], **{source: {"note /value": notation}}, seeds=[3858582668])
        assert {h.term for h in report.hits} == {"3858582668"}


def test_tree_and_mounted_code_stay_strict_with_agent_notes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    fleet = write_toy_run(run)
    (run / "truth/summary.json").write_text('{"healthy_robots": 7}')
    out = tmp_path / "export"
    out.mkdir()
    pq.write_table(pa.table({"true_success": [True], "onset_day": [3]}), out / "rows.parquet")
    (out / "events.json").write_text('{"detail": "decoy inert level=2.0 healthy_robots=7"}')
    report = leaks.audit_export(
        out,
        [run],
        fleet=fleet,
        agent_texts={"notes.md": "gripper wear; nominal success rate"},
        code={"tools.py": "illuminationChange"},
    )
    assert {
        "true_success",
        "onset_day",
        "decoy",
        "inert",
        "level",
        "healthy_robots",
        "illumination_change",
    } <= {hit.term for hit in report.hits}
    assert not any(hit.where == "notes.md" for hit in report.hits)


def test_public_change_kinds_pass_agent_text_only() -> None:
    from fleetlab.contracts.actions import ADDED_CHANGE_KINDS

    terms = leaks.truth_terms()
    text = " ".join(ADDED_CHANGE_KINDS)
    assert leaks.search_texts(terms, agent_texts={"diagnosis /label": text}).clean
    assert set(ADDED_CHANGE_KINDS) <= {
        hit.term for hit in leaks.search_texts(terms, {"prompt": text}).hits
    }


def test_unknown_names_stay_planted_until_the_producer_declares_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.worlds import world

    names = {"unrecognized_truth_name"}
    terms = leaks.classify_truth_terms(names)
    assert names <= terms.planted
    assert leaks.search_texts(terms, {"prompt": "unrecognized truth name"}).hits
    assert leaks.search_texts(terms, agent_texts={"note": "unrecognized_truth_name"}).hits

    def future_record() -> dict[str, int]:
        return {"unrecognized_truth_name": 2}

    monkeypatch.setattr(world, "generator_record", future_record)
    classified = leaks.classify_truth_terms(names)
    assert "unrecognized_truth_name" in classified.structural
    assert leaks.search_texts(classified, agent_texts={"note": "unrecognized_truth_name"}).clean


def test_recipe_configuration_is_known_without_changing_strict_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.worlds import recipe

    class FutureRecipe(recipe.WorldRecipe):
        future_recipe_threshold: int = 0

    FutureRecipe.__module__ = recipe.__name__
    names = {"future_recipe_threshold", "min_detectable", "worn_robots", "min_clean_robots"}
    before = leaks.classify_truth_terms(names)
    assert before.planted == {"future_recipe_threshold"}
    monkeypatch.setattr(recipe, "FutureRecipe", FutureRecipe, raising=False)
    after = leaks.classify_truth_terms(names)
    assert before == after and before.structural == after.structural and not after.planted
    assert leaks.search_texts(after, agent_texts={"note": " ".join(names)}).clean
    assert {h.term for h in leaks.search_texts(after, {"prompt": " ".join(names)}).hits} == names
    assert leaks.search_texts(after, {"prompt": "future recipe threshold"}).hits
    # A genuine property with the same spelling still wins over configuration.
    fleet = write_toy_run(tmp_path / "run")
    fleet.robots()[0].traits["future_recipe_threshold"] = 2
    collision = leaks.classify_truth_terms(names, fleet=fleet)
    assert "future_recipe_threshold" in collision.planted
    assert leaks.search_texts(collision, agent_texts={"note": "future_recipe_threshold=2"}).hits


def test_new_hidden_execution_models_are_discovered_from_their_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel

    from fleetlab.contracts import identity

    class FutureIdentity(BaseModel):
        future_execution_field: int = 0

    FutureIdentity.__module__ = identity.__name__
    monkeypatch.setattr(identity, "FutureIdentity", FutureIdentity, raising=False)
    assert "future_execution_field" in leaks._planted_field_names()
    terms = leaks.classify_truth_terms(["future_execution_field"])
    assert leaks.search_texts(terms, agent_texts={"note": "future_execution_field"}).clean
    assert leaks.search_texts(terms, {"prompt": "future execution field"}).hits


def test_agent_text_requires_source_label() -> None:
    with pytest.raises(ValueError, match="source"):
        leaks.search_texts([], agent_texts={"": "ordinary notes"})
