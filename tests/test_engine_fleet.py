"""The worker layout on the toy cell (`make check`): scheduling, exactness across layouts,
resumability, the chain sample, refusals. Spawns a few CPU processes, no GPU."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from engine_helpers import SOFTWARE, config, ident, jobs

from fleetlab.contracts.log import log_tree_problems
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SupervisorSpec
from fleetlab.engine import miopen, toy
from fleetlab.engine.chain import Chain, compare
from fleetlab.engine.episode import ChunkRequest, EngineError, call_policy
from fleetlab.engine.fleet import check_jobs, run, run_in_process, sample_replays
from fleetlab.engine.server import Batcher

# ---- the scheduling rule --------------------------------------------------------------------


def request(key: str, *, steps: int = 2, row: int | None = None, batch: int = 4) -> ChunkRequest:
    env = toy.ToyEnv("toy", 0, 32)
    obs = env.reset(ident(1), [])
    spec = toy.TOY_SPEC.model_copy(update={"sampling_steps": steps})
    return ChunkRequest(
        key=key, observation=obs, instruction="x", noise=None, spec=spec, row=row, batch_size=batch
    )


def test_the_longest_waiting_requests_go_first_in_free_rows() -> None:
    batcher: Batcher[int] = Batcher(4)
    for w in range(6):
        batcher.add(w, request(f"j{w}"))
    assert [w for w, _ in batcher.next_call()] == [0, 1, 2, 3]
    assert [w for w, _ in batcher.next_call()] == [4, 5]
    assert batcher.next_call() == []
    assert batcher.rows_in_use() == 6 / 8


def test_requests_of_another_effective_spec_wait_for_their_own_call() -> None:
    batcher: Batcher[int] = Batcher(4)
    batcher.add(0, request("a", steps=2))
    batcher.add(1, request("b", steps=5))
    batcher.add(2, request("c", steps=2))
    assert [w for w, _ in batcher.next_call()] == [0, 2]
    assert [w for w, _ in batcher.next_call()] == [1]


def test_fixed_rows_do_not_share_a_call() -> None:
    batcher: Batcher[int] = Batcher(4)
    batcher.add(0, request("a", row=1))
    batcher.add(1, request("b", row=1))
    batcher.add(2, request("c"))
    assert [w for w, _ in batcher.next_call()] == [0, 2]
    assert [w for w, _ in batcher.next_call()] == [1]
    with pytest.raises(ValueError, match="batch size"):
        batcher.add(3, request("d", batch=8))


def test_empty_rows_are_padded_with_row_zero() -> None:
    policy = toy.ToyPolicy()
    seen: list[list[str]] = []
    original = policy.act

    def spy(batch, instructions, noise, spec):  # type: ignore[no-untyped-def]
        seen.append(list(instructions))
        return original(batch, instructions, noise, spec)

    policy.act = spy  # type: ignore[method-assign]
    one = request("a")
    two = ChunkRequest(**{**one.__dict__, "key": "b", "instruction": "y"})
    out = call_policy(policy, [one, two])
    assert seen == [["x", "y", "x", "x"]]
    assert policy.calls == [4] and len(out) == 2
    assert out[0].shape == (toy.TOY_SPEC.actions_executed_per_chunk, 7)


# ---- runs -----------------------------------------------------------------------------------


def chains(layout: RunLayout, manifest: list[Job]) -> dict[str, Chain]:
    return {j.key: Chain.load(layout.chain(j)) for j in manifest}


def test_the_layout_does_not_change_an_episode(tmp_path: Path) -> None:
    manifest = jobs(24, video_every=3)
    inline = run_in_process(config(tmp_path / "inline", chain_sample=False), manifest)
    fleet = run(config(tmp_path / "fleet", workers=3), manifest)
    no_video = [j.model_copy(update={"record_video": False}) for j in manifest]
    other = run(config(tmp_path / "other", workers=5, life=3, stagger=False), no_video)
    a, b, c = (chains(r.layout, manifest) for r in (inline, fleet, other))
    for key in a:
        assert compare(b[key], a[key]).identical, key
        assert compare(c[key], a[key]).identical, key
    assert fleet.record.chain_ok is True and len(fleet.record.chain_sample) == 24
    assert log_tree_problems(fleet.layout.root) == []
    videos = sorted(p.name for p in (fleet.layout.log_dir / "video").iterdir())
    assert len(videos) == 2 * 8
    for name in ("episodes", "identities", "verdicts", "faults_active", "effective_policy"):
        assert fleet.layout.table(name).is_file(), name
    record = json.loads(fleet.layout.run_record.read_text())
    assert record["extra"]["share_of_rows_in_use"] > 0
    assert "pinned_conv_database" not in record["replay_rules"]


def test_first_worker_lives_are_staggered(tmp_path: Path) -> None:
    result = run(config(tmp_path / "r", workers=3, life=6), jobs(30))
    lives = result.record.extra["worker_lives"]
    assert isinstance(lives, list)
    first = sorted(
        (v["worker"], v["jobs"]) for v in lives if isinstance(v, dict) and v["life"] == 0
    )
    assert first == [(0, 2), (1, 4), (2, 6)]


def test_a_run_resumes_where_it_stopped(tmp_path: Path) -> None:
    manifest = jobs(12)
    cfg = config(tmp_path / "r", workers=2)
    first = run(cfg, manifest)
    again = run(cfg, manifest)
    assert again.record.extra["jobs_run_now"] == 0 and again.record.chain_ok
    victim = manifest[5]
    before = Chain.load(first.layout.chain(victim))
    first.layout.chain(victim).unlink()  # as if the run had stopped before it finished
    third = run(cfg, manifest)
    assert third.record.extra["jobs_run_now"] == 1
    assert compare(Chain.load(first.layout.chain(victim)), before).identical


def test_a_broken_chain_invalidates_the_run(tmp_path: Path) -> None:
    manifest = jobs(6)
    cfg = config(tmp_path / "r", workers=2)
    first = run(cfg, manifest)
    replay = sample_replays(manifest)[0]
    path = first.layout.chain(replay)
    chain = Chain.load(path)
    chain.state[3] = "0" * 16
    chain.save(path)
    again = run(cfg, manifest)
    assert again.record.chain_ok is False


def test_a_hand_over_in_the_fleet_runs_the_teleoperator(tmp_path: Path) -> None:
    supervised = SupervisorSpec(name="timeout", params={"after": 6, "decision": "hand_over"})
    manifest = [j.model_copy(update={"supervisor": supervised}) for j in jobs(4, faulted=False)]
    result = run(config(tmp_path / "r", workers=2, chain_sample=False), manifest)
    continued = [s for s in result.summaries if s["mode"] == "continue"]
    assert len(continued) == 4
    assert all(s["first_step"] == 17 for s in continued)  # decided at 7, held 10 steps
    decisions = result.layout.table("decisions")
    assert decisions.is_file()


def test_a_replay_with_a_fault_removed(tmp_path: Path) -> None:
    manifest = jobs(6)
    faulted = [j for j in manifest if j.identity.faults]
    removed = [
        Job(identity=j.identity, mode=JobMode.replay, remove=[j.identity.faults[0].fault_id])
        for j in faulted
    ]
    result = run(config(tmp_path / "r", workers=2), manifest + removed)
    assert result.layout.table("replays").is_file()
    healthy = run_in_process(
        config(tmp_path / "h", chain_sample=False),
        [Job(identity=j.identity.model_copy(update={"faults": []})) for j in faulted],
    )
    for job, plain in zip(removed, faulted, strict=True):
        mine = Chain.load(result.layout.chain(job))
        bare = Chain.load(
            healthy.layout.chain(Job(identity=plain.identity.model_copy(update={"faults": []})))
        )
        assert compare(mine, bare).identical


# ---- refusals -------------------------------------------------------------------------------


def test_jobs_of_another_batch_size_precision_or_software_are_refused(tmp_path: Path) -> None:
    cfg = config(tmp_path / "r")
    good = jobs(2)
    check_jobs(cfg, good, SOFTWARE)
    with pytest.raises(EngineError, match="batch size"):
        check_jobs(replace(cfg, batch_size=8), good, SOFTWARE)
    with pytest.raises(EngineError, match="precision"):
        check_jobs(replace(cfg, precision="fp16_res32"), good, SOFTWARE)
    with pytest.raises(EngineError, match="other software"):
        check_jobs(cfg, good, SOFTWARE.model_copy(update={"miopen_db_hash": "abc"}))
    with pytest.raises(EngineError, match="unique"):
        check_jobs(cfg, good + good[:1], SOFTWARE)


def test_an_unpinned_setting_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEETLAB_MIOPEN_DIR", str(tmp_path))
    with pytest.raises(miopen.UnpinnedError, match="no pinned"):
        miopen.require(4, "fp16_res32")
    base = miopen.root()
    (base / "db").mkdir(parents=True)
    db_file = base / "db" / "gfx1151.udb.txt"
    db_file.write_text("conv-key=algorithm\n")
    record = {
        "entries": [{"batch_size": 4, "vision_format": "fp16"}],
        "files": miopen.file_hashes(base),
        "hash": miopen.database_hash(base),
    }
    (base / miopen.MANIFEST).write_text(json.dumps(record))
    assert miopen.require(4, "fp16_res32") == record["hash"]
    assert miopen.require(4, "fp16") == record["hash"]  # the same vision tower format
    with pytest.raises(miopen.UnpinnedError, match="batch size 8"):
        miopen.require(8, "fp16_res32")
    with pytest.raises(miopen.UnpinnedError, match="fp32"):
        miopen.require(4, "fp32")
    db_file.chmod(db_file.stat().st_mode | stat.S_IWUSR)
    db_file.write_text("conv-key=another algorithm\n")  # MIOpen rewrote its choice
    with pytest.raises(miopen.UnpinnedError, match="differs"):
        miopen.require(4, "fp16_res32")
    monkeypatch.setenv("MIOPEN_USER_DB_PATH", "elsewhere")
    assert miopen.activate() == base and os.environ["MIOPEN_USER_DB_PATH"] == str(base / "db")


def test_the_run_refuses_an_unpinned_gpu_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLEETLAB_MIOPEN_DIR", str(tmp_path / "nothing"))
    cfg = config(tmp_path / "r", device="cuda", miopen="pinned")
    with pytest.raises(miopen.UnpinnedError):
        run(cfg, jobs(2))
    assert not (tmp_path / "r" / "run.json").exists()


def test_the_hash_chain_rows_are_the_schema_s(tmp_path: Path) -> None:
    result = run_in_process(config(tmp_path / "r", chain_sample=False), jobs(1))
    with np.load(result.layout.chain(jobs(1)[0])) as data:
        names = set(data.files)
    from fleetlab.contracts.log import SCHEMA

    assert names == {c.name for c in SCHEMA["chain"].columns}


# ---- the review's fixes ---------------------------------------------------------------------


def supervised_jobs(n: int, decision: str = "hand_over") -> list[Job]:
    spec = SupervisorSpec(name="timeout", params={"after": 6, "decision": decision})
    return [j.model_copy(update={"supervisor": spec}) for j in jobs(n, faulted=False)]


def test_the_chain_sample_proves_supervised_jobs_by_scripted_replays(tmp_path: Path) -> None:
    manifest = supervised_jobs(6)
    result = run(config(tmp_path / "r", workers=2), manifest)
    assert result.record.chain_ok is True and len(result.record.chain_sample) == 6
    modes = [s["mode"] for s in result.summaries]
    assert modes.count("scripted") == 6  # every sample replay replays the decisions
    assert modes.count("continue") == 6  # and the scripted replays spawned no second one


def test_a_resumed_run_finishes_its_hand_overs(tmp_path: Path) -> None:
    manifest = supervised_jobs(4)
    cfg = config(tmp_path / "r", workers=2, chain_sample=False)
    first = run(cfg, manifest)
    layout = first.layout
    continuations = layout.continuations()
    assert len(continuations) == 4
    lost = continuations[1]
    before = Chain.load(layout.chain(lost))
    layout.chain(lost).unlink()  # as if killed before the continuation finished
    (layout.root / "truth" / "parts" / f"{lost.key}.json").unlink()
    again = run(cfg, manifest)
    assert again.record.extra["jobs_run_now"] == 1  # the continuation, found on disk
    assert compare(Chain.load(layout.chain(lost)), before).identical
    assert [s["key"] for s in again.summaries] == [lost.key]


def test_a_replay_from_a_snapshot_may_not_change_a_cell_hook_fault() -> None:
    from engine_helpers import context, fault, ident

    from fleetlab.engine.episode import MemoryStore, run_job

    store = MemoryStore()
    cam = fault("toy_camera_shift", 1, pixels=2)
    delay = fault("toy_delay", 2, steps=2)
    env = toy.ToyEnv("toy", 0, 32)
    original = run_job(
        Job(identity=ident(2, faults=[cam, delay])), env, context(store=store, steps=(8,))
    )
    ref = original.snapshots[0]
    with pytest.raises(EngineError, match="cell hook"):
        run_job(
            Job(identity=original.identity, mode=JobMode.replay, remove=["f01"], from_snapshot=ref),
            env,
            context(store=store),
        )
    without_delay = run_job(  # an action hook may go: the snapshot does not carry it
        Job(identity=original.identity, mode=JobMode.replay, remove=["f02"], from_snapshot=ref),
        env,
        context(store=store),
    )
    assert without_delay.chain.first_step == 8


@pytest.mark.parametrize("losses", [1, 2])
def test_a_lost_worker_s_jobs_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, losses: int
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ONCE", str(tmp_path / "crashed"))
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_COUNT", str(losses))
    # One slot for two losses ensures they are the same run job, rather than a
    # run and its already-dispatched sample both resetting that episode.
    cfg = config(
        tmp_path / "r",
        workers=1 if losses == 2 else 2,
        env_factory="engine_helpers:CrashingToyEnv",
    )
    result = run(cfg, jobs(8))
    assert (tmp_path / "crashed").exists()
    incidents = cast(list[dict[str, Any]], result.record.extra["incidents"])
    assert isinstance(incidents, list) and len(incidents) == losses
    assert all(i["recovered"] for i in incidents)
    assert [i["attempts"]["e000003"] for i in incidents] == list(range(1, losses + 1))
    assert all(i["replay_checks"]["e000003"] == {"checked": True, "verdict": "identical"}
               for i in incidents)  # fmt: skip
    assert result.record.chain_ok is True
    assert all(result.layout.is_done(j) for j in jobs(8))


def test_a_job_lost_three_times_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ALWAYS", "1")
    cfg = config(tmp_path / "r", workers=2, env_factory="engine_helpers:CrashingToyEnv")
    with pytest.raises(EngineError, match=r"job \S+ failed 3 times; last:\nexit code 3") as error:
        run(cfg, jobs(8))
    # Deferred writes can lose the preceding job with the crashing job. Either
    # may reach the limit first; the error must name a job with three losses.
    failed_key = str(error.value).split()[1]
    incidents = json.loads((tmp_path / "r" / "run.json").read_text())["extra"]["incidents"]
    assert incidents[-1]["attempts"][failed_key] == 3
    assert all(not i["recovered"] for i in incidents)


@pytest.mark.parametrize("supervised", [False, True])
def test_a_repeated_run_joins_the_chain_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervised: bool
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_EPISODE", "e000002")
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ONCE", str(tmp_path / "crashed"))
    manifest = supervised_jobs(40) if supervised else jobs(40)
    assert "e000002" not in {j.identity.episode_id for j in sample_replays(jobs(40))}
    cfg = config(tmp_path / "r", workers=2, life=10, env_factory="engine_helpers:CrashingToyEnv")
    result = run(cfg, manifest)
    assert "e000002" in result.record.chain_sample
    assert result.record.chain_ok is True
    incident = cast(list[dict[str, Any]], result.record.extra["incidents"])[0]
    assert incident["recovered"]
    assert incident["replay_checks"]["e000002"] == {"checked": True, "verdict": "identical"}
    if supervised:
        assert any(s["episode_id"] == "e000002" and s["mode"] == "scripted"
                   for s in result.summaries)  # fmt: skip


def test_a_repeat_without_a_chain_sample_is_recorded_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ONCE", str(tmp_path / "crashed"))
    result = run(config(tmp_path / "r", workers=2, chain_sample=False,
                        env_factory="engine_helpers:CrashingToyEnv"), jobs(8))  # fmt: skip
    incident = cast(list[dict[str, Any]], result.record.extra["incidents"])[0]
    assert incident["recovered"] and result.record.chain_sample == []
    assert all(c == {"checked": False, "verdict": None} for c in incident["replay_checks"].values())


def test_a_repeated_continuation_is_recovered_without_a_replay_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ONCE", str(tmp_path / "crashed"))
    result = run(config(tmp_path / "r", workers=1,
                        env_factory="engine_helpers:CrashingContinuationToyEnv"),
                 supervised_jobs(2))  # fmt: skip
    incident = cast(list[dict[str, Any]], result.record.extra["incidents"])[0]
    assert incident["recovered"] and result.record.chain_ok is True
    continuations = [k for k in incident["jobs"] if ".continue." in k]
    assert continuations
    assert all(incident["replay_checks"][k] == {"checked": False, "verdict": None}
               for k in continuations)  # fmt: skip


def run_worker_probe(tmp_path: Path, script: str, *, timeout: float = 120) -> str:
    """Bound the whole interpreter, including multiprocessing's exit finalizers."""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path / "r")],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate(timeout=10)
        pytest.fail(f"worker-loss probe stalled:\n{output}")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    assert proc.returncode == 0, output
    return output


@pytest.mark.parametrize("target", ["closed_ask_worker", "closed_request_worker"])
def test_a_lost_worker_with_a_closed_pipe_recovers(tmp_path: Path, target: str) -> None:
    script = f"""
import multiprocessing as mp
import sys
from pathlib import Path
from engine_helpers import config, jobs, {target}
from fleetlab.engine import fleet

fleet.worker_main = {target}
manifest = jobs(2)
result = fleet.run(config(Path(sys.argv[1]), workers=1), manifest)
incidents = result.record.extra['incidents']
assert len(incidents) == 1, incidents
assert incidents[0]['recovered'], incidents
assert 'BrokenPipeError' in incidents[0]['why'], incidents
assert incidents[0]['session'] == 1
assert all(result.layout.is_done(job) for job in manifest)
assert result.record.chain_ok is True
assert not mp.active_children(), mp.active_children()
print('closed pipe recovered', flush=True)
"""
    assert "closed pipe recovered" in run_worker_probe(tmp_path, script)


def test_a_dispatcher_error_stops_the_run(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp
import sys
from pathlib import Path
from engine_helpers import config, jobs, dispatcher_error_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError

fleet.worker_main = dispatcher_error_worker
try:
    fleet.run(config(Path(sys.argv[1]), workers=1), jobs(2))
except EngineError as error:
    assert 'dispatcher failed:' in str(error), str(error)
    assert 'Invalid JSON' in str(error), str(error)
else:
    raise AssertionError('the dispatcher error must end the run')
assert not mp.active_children(), mp.active_children()
print('dispatcher error reported', flush=True)
"""
    assert "dispatcher error reported" in run_worker_probe(tmp_path, script)


def test_a_monitor_error_stops_the_run(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp
import sys
from pathlib import Path
from engine_helpers import config, jobs, idle_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError

def fail_memory_read(pid='self'):
    raise ValueError('monitor memory probe failed')

fleet.worker_main = idle_worker
fleet.pss_mib = fail_memory_read
try:
    fleet.run(config(Path(sys.argv[1]), workers=1), jobs(1))
except EngineError as error:
    assert 'monitor failed:' in str(error), str(error)
    assert 'monitor memory probe failed' in str(error), str(error)
else:
    raise AssertionError('the monitor error must end the run')
assert not mp.active_children(), mp.active_children()
print('monitor error reported', flush=True)
"""
    assert "monitor error reported" in run_worker_probe(tmp_path, script)


def test_every_test_has_a_deadline(tmp_path: Path) -> None:
    path = tmp_path / "test_stall.py"
    path.write_text("import time\ndef test_stall():\n    time.sleep(3600)\n")
    script = f"""
import pytest
import conftest
conftest.TEST_DEADLINE_S = 0.1
assert pytest.main(['-q', '-p', 'conftest', {str(path)!r}]) == 1
print('test deadline enforced', flush=True)
"""
    output = run_worker_probe(tmp_path, script)
    assert "exceeded its 0.1s test deadline" in output
    assert "test deadline enforced" in output


def test_a_lost_worker_cannot_hang_shutdown_with_a_queue_lock(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp
import sys
from pathlib import Path
from engine_helpers import config, jobs, poisoned_queue_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError

fleet.worker_main = poisoned_queue_worker
try:
    fleet.run(config(Path(sys.argv[1]), workers=1), jobs(1))
except EngineError as error:
    assert 'job e000001 failed 3 times' in str(error), str(error)
else:
    raise AssertionError('the run must abort')
assert not mp.active_children(), mp.active_children()
print('shutdown completed', flush=True)
"""
    assert "shutdown completed" in run_worker_probe(tmp_path, script)


@pytest.mark.parametrize("ending", ["policy", "dispatcher", "third", "interrupt"])
def test_shutdown_with_a_request_larger_than_the_pipe_is_bounded(
    tmp_path: Path, ending: str
) -> None:
    script = f"""
import array, faulthandler, fcntl, json, multiprocessing as mp, os, signal, sys
import termios, threading, time
from pathlib import Path
from engine_helpers import config, jobs, large_shutdown_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError

ending = {ending!r}
faulthandler.dump_traceback_later(45)
os.environ['FLEETLAB_TEST_LARGE_SHUTDOWN'] = ending
out = Path(sys.argv[1])
fleet.worker_main = large_shutdown_worker
ctx = mp.get_context('spawn')
queues = []
original_queue = ctx.Queue
def observed_queue(*args, **kwargs):
    q = original_queue(*args, **kwargs)
    queues.append(q)
    return q
ctx.Queue = observed_queue
fleet.mp.get_context = lambda method: ctx

def fill_pipe():
    global shutdown_start
    q = queues[0]
    capacity = fcntl.fcntl(q._reader.fileno(), fcntl.F_GETPIPE_SZ)
    assert 777_600 > capacity, capacity
    (out / 'fill-request-pipe').write_text('ready')
    deadline = time.monotonic() + 30
    while True:
        waiting = array.array('i', [0])
        fcntl.ioctl(q._reader.fileno(), termios.FIONREAD, waiting, True)
        # A partially used page can exhaust pipe buffer slots before the
        # number of readable bytes reaches the nominal byte capacity.
        if waiting[0] >= capacity - os.sysconf('SC_PAGESIZE'):
            assert not q._wlock.acquire(False), 'the feeder must own the writer lock'
            print(f'partial world-size frame: {{waiting[0]}}/{{777_600}} bytes', flush=True)
            shutdown_start = time.monotonic()
            return
        assert time.monotonic() < deadline, (waiting[0], capacity)
        time.sleep(0.01)

original_process = ctx.Process
def observed_process(*args, **kwargs):
    process = original_process(*args, **kwargs)
    wid, generation = kwargs['args'][:2]
    class ObservedProcess:
        def __getattr__(self, name):
            return getattr(process, name)
        def join(self, timeout=None):
            terminal = ending == 'third' and wid == 0 and generation == 2
            if terminal and not (out / 'fill-request-pipe').exists():
                fill_pipe()
            return process.join(timeout)
    return ObservedProcess()
ctx.Process = observed_process

dispatcher = []
class ObservedThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get('args', ()) and kwargs['args'][0] == 'dispatcher':
            dispatcher.append(self)
fleet.threading.Thread = ObservedThread
original_call = fleet.call_policy
shutdown_start = None
def call(policy, requests):
    global shutdown_start
    fill_pipe()
    if ending == 'dispatcher':
        deadline = time.monotonic() + 30
        while dispatcher[0].is_alive():
            assert time.monotonic() < deadline, 'the dispatcher did not report its error'
            time.sleep(0.01)
        shutdown_start = time.monotonic()
        return original_call(policy, requests)
    shutdown_start = time.monotonic()
    if ending == 'interrupt':
        os.kill(os.getpid(), signal.SIGINT)
    raise EngineError('deliberate policy abort with a partial frame')
fleet.call_policy = call
manifest = [j.model_copy(update={{'identity': j.identity.model_copy(update={{'image_size': 360}})}})
            for j in jobs(2, faulted=False)]
try:
    fleet.run(config(out, workers=2, stagger=False, life=8, chain_sample=False), manifest)
except KeyboardInterrupt:
    assert ending == 'interrupt'
except EngineError as error:
    expected = {{'policy': 'deliberate policy abort', 'dispatcher': 'dispatcher failed:',
                'third': 'failed 3 times'}}
    assert expected[ending] in str(error), str(error)
else:
    raise AssertionError('the run must raise its original ending')
if shutdown_start is not None:
    duration = time.monotonic() - shutdown_start
    assert duration < 20, 'shutdown exceeded its bound'
    print(f'shutdown returned in {{duration:.3f}} seconds', flush=True)
record = json.loads((out / 'run.json').read_text())
incidents = record['extra']['incidents']
assert record['extra']['session_count'] == 1
if ending == 'third':
    assert len(incidents) == 3 and incidents[-1]['ended_run'], incidents
    assert not incidents[-1]['recovered'], incidents
assert not mp.active_children(), mp.active_children()
print('large request shutdown completed', flush=True)
"""
    output = run_worker_probe(tmp_path, script)
    (tmp_path / "probe.log").write_text(output)
    assert "partial world-size frame:" in output
    assert "large request shutdown completed" in output


@pytest.mark.parametrize("mode", ["exit", "poison"])
def test_a_life_lost_before_its_first_job_aborts(tmp_path: Path, mode: str) -> None:
    script = f"""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path
from engine_helpers import config, jobs, startup_loss_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError
os.environ['FLEETLAB_TEST_STARTUP_LOSS'] = {mode!r}
fleet.worker_main = startup_loss_worker
start = time.monotonic()
try:
    fleet.run(config(Path(sys.argv[1]), workers=1, watchdog_s=5), jobs(1))
except EngineError as error:
    assert 'lost before receiving any job' in str(error), str(error)
    assert 'cannot make progress' in str(error), str(error)
else:
    raise AssertionError('an unproductive life must abort')
assert time.monotonic() - start < 90
incidents = json.loads((Path(sys.argv[1]) / 'run.json').read_text())['extra']['incidents']
assert len(incidents) == {3 if mode == "exit" else 4}, incidents
assert incidents[-1]['jobs'] == []
assert incidents[-1]['ended_run'] and not incidents[-1]['recovered'], incidents
assert not mp.active_children(), mp.active_children()
print('startup loss bounded', flush=True)
"""
    assert "startup loss bounded" in run_worker_probe(tmp_path, script)


@pytest.mark.parametrize("mode", ["replace", "busy", "retire"])
def test_one_empty_life_loss_does_not_abort_other_workers(tmp_path: Path, mode: str) -> None:
    script = f"""
import multiprocessing as mp, os, sys, time
from pathlib import Path
from engine_helpers import config, jobs, isolated_startup_loss_worker
from fleetlab.engine import fleet
os.environ['FLEETLAB_TEST_EMPTY_LOSS'] = {mode!r}
out = Path(sys.argv[1])
fleet.worker_main = isolated_startup_loss_worker
ctx = mp.get_context('spawn')
original_process = ctx.Process
processes = {{}}
def observed_process(*args, **kwargs):
    process = original_process(*args, **kwargs)
    wid, generation = kwargs['args'][:2]
    processes[wid] = process
    class ObservedProcess:
        def __getattr__(self, name):
            return getattr(process, name)
        def start(self):
            process.start()
            if wid == 1 and generation == 1:
                (out / 'replacement-started').write_text('started')
    return ObservedProcess()
ctx.Process = observed_process
fleet.mp.get_context = lambda method: ctx
original_call = fleet.call_policy
def call(policy, requests):
    marker = out / 'all-assigned'
    release = {mode!r} == 'busy' or any(r.key == 'e000002' for r in requests)
    if {mode!r} != 'replace' and release and not marker.exists():
        marker.write_text('the other worker is executing')
        deadline = time.monotonic() + 30
        while processes[1].is_alive():
            assert time.monotonic() < deadline, 'the empty life did not exit'
            time.sleep(0.01)
    return original_call(policy, requests)
fleet.call_policy = call
manifest = jobs({8 if mode == "busy" else 2})
result = fleet.run(config(out, workers=2, life=16, stagger=False,
                          chain_sample=False), manifest)
incidents = result.record.extra['incidents']
assert len(incidents) == 1 and incidents[0]['recovered'], incidents
assert incidents[0]['worker'] == 1 and incidents[0]['jobs'] == [], incidents
assert incidents[0]['exitcode'] == 3 and not incidents[0]['received_job'], incidents
assert all(result.layout.is_done(j) for j in manifest)
assert (out / 'replacement-started').exists() is {mode != "retire"}, incidents
assert not mp.active_children(), mp.active_children()
print('isolated empty loss recovered', flush=True)
"""
    output = run_worker_probe(tmp_path, script)
    (tmp_path / "probe.log").write_text(output)
    assert "isolated empty loss recovered" in output


def test_a_normal_exit_with_a_queued_goodbye_is_not_lost(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp, sys, time
from pathlib import Path
from engine_helpers import config, jobs, normal_goodbye_during_loss_worker
from fleetlab.engine import fleet
out = Path(sys.argv[1])
fleet.worker_main = normal_goodbye_during_loss_worker
ctx = mp.get_context('spawn')
original_process = ctx.Process
processes = {}
def observed_process(*args, **kwargs):
    process = original_process(*args, **kwargs)
    wid, generation = kwargs['args'][:2]
    processes[wid] = process
    class ObservedProcess:
        def __getattr__(self, name):
            return getattr(process, name)
        def join(self, timeout=None):
            if wid == 0 and generation == 0 and not (out / 'loss-join').exists():
                (out / 'loss-join').write_text('joining the lost slot')
                deadline = time.monotonic() + 30
                while processes[1].is_alive():
                    assert time.monotonic() < deadline, 'the normal worker never exited'
                    time.sleep(0.01)
                assert processes[1].exitcode == 0
            return process.join(timeout)
    return ObservedProcess()
ctx.Process = observed_process
fleet.mp.get_context = lambda method: ctx
manifest = jobs(2)
result = fleet.run(config(out, workers=2, life=8, stagger=False,
                          chain_sample=False), manifest)
incidents = result.record.extra['incidents']
assert len(incidents) == 1 and incidents[0]['worker'] == 0, incidents
assert incidents[0]['recovered'], incidents
assert any(w['worker'] == 1 and w['jobs'] == 0 for w in result.record.extra['worker_lives'])
assert all(result.layout.is_done(j) for j in manifest)
assert not mp.active_children(), mp.active_children()
print('normal goodbye retained', flush=True)
"""
    output = run_worker_probe(tmp_path, script)
    (tmp_path / "probe.log").write_text(output)
    assert "normal goodbye retained" in output


def test_a_normal_exit_without_a_goodbye_is_bounded(tmp_path: Path) -> None:
    script = """
import json, multiprocessing as mp, sys, time
from pathlib import Path
from engine_helpers import config, jobs, zero_without_goodbye_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError
fleet.worker_main = zero_without_goodbye_worker
out = Path(sys.argv[1])
start = time.monotonic()
try:
    fleet.run(config(out, workers=1, watchdog_s=1), jobs(1))
except EngineError as error:
    assert 'exited normally but no goodbye was received within 1 s' in str(error), str(error)
else:
    raise AssertionError('a missing goodbye must end with an explicit error')
assert time.monotonic() - start < 30
assert json.loads((out / 'run.json').read_text())['extra']['incidents'] == []
assert not mp.active_children(), mp.active_children()
print('missing goodbye bounded', flush=True)
"""
    assert "missing goodbye bounded" in run_worker_probe(tmp_path, script)


@pytest.mark.parametrize("exit_window", ["grace", "final"])
def test_a_clean_exit_observed_during_loss_grace_is_not_lost(
    tmp_path: Path, exit_window: str
) -> None:
    script = (
        f"exit_window = {exit_window!r}\n"
        + """
import multiprocessing as mp, os, sys, threading, time
from pathlib import Path
from engine_helpers import config, jobs, normal_goodbye_during_loss_worker
from fleetlab.engine import fleet
os.environ['FLEETLAB_TEST_EXIT_DURING_GRACE'] = '1'
out = Path(sys.argv[1])
fleet.worker_main = normal_goodbye_during_loss_worker
ctx = mp.get_context('spawn')
original_queue = ctx.Queue
queues = []
def observed_queue(*args, **kwargs):
    q = original_queue(*args, **kwargs)
    if not queues:
        original_get = q.get
        def get(*args, **kwargs):
            (out / 'parent-waiting').write_text('waiting')
            return original_get(*args, **kwargs)
        q.get = get
    queues.append(q)
    return q
ctx.Queue = observed_queue
original_process = ctx.Process
processes = {}
def observed_process(*args, **kwargs):
    process = original_process(*args, **kwargs)
    wid, generation = kwargs['args'][:2]
    processes[wid] = process
    def finish_clean_exit():
        (out / 'exit-now').write_text('exit after the snapshot')
        deadline = time.monotonic() + 30
        while process.is_alive():
            assert time.monotonic() < deadline, 'the normal worker did not exit'
            time.sleep(0.01)
        assert process.exitcode == 0
    class ObservedProcess:
        snapshots = 0
        joined = False
        def __getattr__(self, name):
            return getattr(process, name)
        @property
        def exitcode(self):
            status = process.exitcode
            if wid == 1 and generation == 0 and (out / 'bye-queued').exists():
                self.snapshots += 1
                if self.snapshots == 3 and exit_window == 'grace':
                    # The process exits between the watchdog's status snapshot
                    # and lose's join. Return the status actually sampled first.
                    assert status is None
                    finish_clean_exit()
            return status
        def is_alive(self):
            if (wid == 1 and generation == 0 and self.joined
                    and exit_window == 'final' and threading.current_thread().name == 'MainThread'
                    and not (out / 'exit-now').exists()):
                # Finish after the grace join/status check but before the final
                # join. The actual process exits normally; is_alive reports that.
                assert process.is_alive()
                finish_clean_exit()
            return process.is_alive()
        def join(self, timeout=None):
            if wid == 0 and generation == 0 and not (out / 'loss-join').exists():
                (out / 'loss-join').write_text('joining the real lost slot')
                deadline = time.monotonic() + 30
                while not (out / 'bye-queued').exists():
                    assert time.monotonic() < deadline, 'the normal goodbye was never flushed'
                    time.sleep(0.01)
            result = process.join(timeout)
            if wid == 1 and generation == 0:
                self.joined = True
            return result
    return ObservedProcess()
ctx.Process = observed_process
fleet.mp.get_context = lambda method: ctx
result = fleet.run(config(out, workers=2, life=8, stagger=False, chain_sample=False,
                          watchdog_s=5, worker_exit_grace_s=6), jobs(2))
incidents = result.record.extra['incidents']
assert len(incidents) == 1 and incidents[0]['worker'] == 0, incidents
assert incidents[0]['recovered'], incidents
assert all(result.layout.is_done(j) for j in jobs(2))
assert not mp.active_children(), mp.active_children()
print('clean exit during grace retained', flush=True)
"""
    )
    output = run_worker_probe(tmp_path, script)
    (tmp_path / "probe.log").write_text(output)
    assert "clean exit during grace retained" in output


@pytest.mark.parametrize("hang", [False, True])
def test_a_reporting_worker_gets_time_to_exit(tmp_path: Path, hang: bool) -> None:
    script = f"""
import multiprocessing as mp, os, sys
from pathlib import Path
from engine_helpers import config, jobs, reported_error_worker
from fleetlab.engine import fleet
os.environ['FLEETLAB_TEST_REPORT_HANG'] = {"1" if hang else ""!r}
fleet.worker_main = reported_error_worker
result = fleet.run(config(Path(sys.argv[1]), workers=1), jobs(1))
incidents = result.record.extra['incidents']
assert len(incidents) == 1 and incidents[0]['recovered'], incidents
incident = incidents[0]
assert incident['worker_error'] == 'RuntimeError: reported worker failure', incident
assert incident['exitcode'] == {-signal.SIGKILL if hang else 1}, incident
assert incident['parent_killed'] is {hang}, incident
assert incident['attempts_kind'] == 'lost_assignments', incident
assert incident['assignment_status']['e000001'] == 'execution_unconfirmed', incident
assert result.record.chain_ok is True
assert not mp.active_children(), mp.active_children()
print('reporting worker recovered', flush=True)
"""
    assert "reporting worker recovered" in run_worker_probe(tmp_path, script)


@pytest.mark.parametrize("mode", ["exit", "error", "done"])
def test_a_lost_worker_keeps_its_exit_and_late_reports(tmp_path: Path, mode: str) -> None:
    script = f"""
import multiprocessing as mp, os, sys, time
from pathlib import Path
from engine_helpers import config, jobs, request_ending_worker
from fleetlab.engine import fleet
os.environ['FLEETLAB_TEST_REQUEST_END'] = {mode!r}
fleet.worker_main = request_ending_worker
original = fleet.call_policy
flag = Path(sys.argv[1]) / 'policy-entered'
def slow_call(policy, requests):
    target = 'e000002' if {mode!r} == 'done' else 'e000001'
    if not flag.exists() and any(r.key == target for r in requests):
        worker = mp.active_children()[0]
        flag.write_text('serving')
        until = time.monotonic() + 30
        while worker.is_alive():
            assert time.monotonic() < until, 'the worker did not exit'
            time.sleep(0.01)
    return original(policy, requests)
fleet.call_policy = slow_call
result = fleet.run(config(Path(sys.argv[1]), workers=1), jobs(2))
incidents = result.record.extra['incidents']
assert len(incidents) == 1 and incidents[0]['recovered'], incidents
incident = incidents[0]
assert 'BrokenPipeError' in incident['why'], incident
assert incident['exitcode'] == {1 if mode == "error" else 3}, incident
assert incident['parent_killed'] is False, incident
if {mode!r} == 'error':
    assert 'the disk is full' in incident['worker_error'], incident
elif {mode!r} == 'done':
    assert incident['assignment_status']['e000001'] == 'completed_report_late', incident
else:
    assert incident['worker_error'] is None, incident
assert result.record.chain_ok is True
assert all(result.layout.is_done(j) for j in jobs(2))
assert not mp.active_children(), mp.active_children()
print('worker ending retained', flush=True)
"""
    assert "worker ending retained" in run_worker_probe(tmp_path, script)


def test_a_missing_completion_report_cannot_return_a_completed_run(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp, sys
from pathlib import Path
from engine_helpers import config, jobs, missing_done_worker
from fleetlab.engine import fleet
from fleetlab.engine.episode import EngineError
fleet.worker_main = missing_done_worker
try:
    fleet.run(config(Path(sys.argv[1]), workers=1, chain_sample=False), jobs(1))
except EngineError as error:
    assert 'assigned jobs neither completed nor pending: e000001' in str(error), str(error)
else:
    raise AssertionError('missing handed-out work must fail the run')
assert not mp.active_children(), mp.active_children()
print('missing assignment refused', flush=True)
"""
    assert "missing assignment refused" in run_worker_probe(tmp_path, script)


def test_an_ask_of_a_retired_slot_cannot_take_work(tmp_path: Path) -> None:
    script = """
import multiprocessing as mp, sys
from pathlib import Path
from engine_helpers import config, jobs, retired_slot_worker
from fleetlab.engine import fleet
fleet.worker_main = retired_slot_worker
ctx = mp.get_context('spawn')
original = ctx.Process
def observed_process(*args, **kwargs):
    process = original(*args, **kwargs)
    wid = kwargs['args'][0]
    class ObservedProcess:
        def __getattr__(self, name):
            return getattr(process, name)
        def join(self, timeout=None):
            process.join(timeout)
            if wid == 1:
                (Path(sys.argv[1]) / 'retired').write_text('joined')
    return ObservedProcess()
ctx.Process = observed_process
fleet.mp.get_context = lambda method: ctx
result = fleet.run(config(Path(sys.argv[1]), workers=2, life=1, stagger=False,
                          chain_sample=False), jobs(2))
assert result.layout.is_done(jobs(3)[2])
assert len(result.summaries) == 3, result.summaries
assert not result.record.extra['incidents'], result.record.extra['incidents']
assert not mp.active_children(), mp.active_children()
print('retired ask ignored', flush=True)
"""
    assert "retired ask ignored" in run_worker_probe(tmp_path, script)


def test_incidents_are_kept_across_completed_and_aborted_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ONCE", str(tmp_path / "crashed"))
    cfg = config(
        tmp_path / "r",
        workers=1,
        episodes_per_robot_day=9,
        env_factory="engine_helpers:CrashingToyEnv",
    )
    first = run(cfg, jobs(4))
    earlier = cast(list[dict[str, Any]], first.record.extra["incidents"])
    assert len(earlier) == 1 and earlier[0]["session"] == 1
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_EPISODE", "e000006")
    monkeypatch.setenv("FLEETLAB_TEST_CRASH_ALWAYS", "1")
    with pytest.raises(EngineError, match="failed 3 times"):
        run(cfg, jobs(8))
    aborted = json.loads((tmp_path / "r" / "run.json").read_text())["extra"]
    history = aborted["incidents"]
    assert history[:1] == earlier
    assert all(i["session"] == 2 for i in history[1:])
    assert aborted["session_count"] == 2
    assert all(c == {"checked": False, "verdict": None}
               for i in history[1:] for c in i["replay_checks"].values())  # fmt: skip
    monkeypatch.delenv("FLEETLAB_TEST_CRASH_ALWAYS")
    monkeypatch.delenv("FLEETLAB_TEST_CRASH_ONCE")
    resumed = run(cfg, jobs(8))
    assert resumed.record.extra["incidents"] == history
    assert resumed.record.extra["session_count"] == 3


def test_a_legacy_incident_keeps_its_fields_and_gets_a_session_number(tmp_path: Path) -> None:
    cfg = config(tmp_path / "r", workers=1)
    run(cfg, jobs(1))
    path = tmp_path / "r" / "run.json"
    record = json.loads(path.read_text())
    del record["extra"]["session_count"]
    legacy = {"worker": 0, "why": "old loss", "jobs": ["e000001"],
              "recovered": True, "attempts": {"e000001": 1},
              "replay_checks": {"e000001": {"checked": False}}}  # fmt: skip
    record["extra"]["incidents"] = [legacy]
    path.write_text(json.dumps(record))
    result = run(cfg, jobs(1))
    history = cast(list[dict[str, Any]], result.record.extra["incidents"])
    assert len(history) == 1
    assert all(history[0][k] == legacy[k] for k in legacy if k != "replay_checks")
    assert history[0]["session"] == 1 and result.record.extra["session_count"] == 2
    assert history[0]["replay_checks"]["e000001"] == {"checked": False, "verdict": None}


@pytest.mark.parametrize("abort", [False, True])
def test_a_shutdown_record_error_preserves_the_original_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abort: bool
) -> None:
    from fleetlab.log import RunWriter

    def fail_write(self: Any, **fields: Any) -> Any:
        raise RuntimeError("incident write failed")

    monkeypatch.setattr(RunWriter, "update_record", fail_write)
    if abort:
        monkeypatch.setenv("FLEETLAB_TEST_CRASH_ALWAYS", "1")
    cfg = config(tmp_path / "r", workers=1, env_factory="engine_helpers:CrashingToyEnv")
    expected = EngineError if abort else RuntimeError
    message = "failed 3 times" if abort else "incident write failed"
    with pytest.raises(expected, match=message) as caught:
        run(cfg, jobs(4))
    if abort:
        assert any("incident write failed" in note for note in caught.value.__notes__)


def test_the_world_s_day_length_is_required(tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="episodes per robot and day"):
        run(config(tmp_path / "r", allow_manifest_day_length=False), jobs(2))
    done = run(
        config(tmp_path / "s", allow_manifest_day_length=False, episodes_per_robot_day=14), jobs(2)
    )
    assert done.record.extra["episodes_per_robot_day"] == 14


def test_images_per_call_are_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.engine.fleet import check_images

    monkeypatch.setenv("FLEETLAB_MIOPEN_DIR", str(tmp_path))
    base = miopen.root()
    (base / "db").mkdir(parents=True)
    (base / "db" / "gfx.udb.txt").write_text("conv=algo\n")
    record = {
        "entries": [{"batch_size": 4, "vision_format": "fp32"}],
        "files": miopen.file_hashes(base),
        "hash": miopen.database_hash(base),
    }
    (base / miopen.MANIFEST).write_text(json.dumps(record))
    assert miopen.images_per_call(4, 4) == miopen.images_per_call(4, 2) == 8
    cfg = config(tmp_path / "r", device="cuda", miopen="pinned")
    check_images(cfg, jobs(3))  # the toy policy's two views: 8 images, pinned
    one_view = toy.TOY_SPEC.model_copy(update={"image_views": 1})
    narrow = [Job(identity=ident(1, policy=one_view))]
    with pytest.raises(EngineError, match="4 images per call"):
        check_images(cfg, narrow)
