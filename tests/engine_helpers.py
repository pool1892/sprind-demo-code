"""Shared by the engine's tests: toy identities, jobs, contexts and run configurations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.engine import manifests, software, toy
from fleetlab.engine.episode import DirectSource, EpisodeContext, MemoryStore
from fleetlab.engine.fleet import RunConfig
from fleetlab.engine.supervisors import make as make_supervisor

SOFTWARE = software.current("none")
SIZE = 32


def ident(
    n: int = 1,
    *,
    robot: int = 1,
    task: int = 0,
    start: int | None = None,
    faults: Sequence[FaultSpec] = (),
    policy: PolicySpec = toy.TOY_SPEC,
    batch_size: int = 4,
    max_steps: int = 60,
) -> EpisodeIdentity:
    return manifests.identity(
        world_seed=3,
        episode_number=n,
        robot=robot,
        site=1,
        day=0,
        episode_index=n,
        suite="toy",
        task_index=task,
        start_state=n % 7 if start is None else start,
        policy=policy,
        batch_size=batch_size,
        software=SOFTWARE,
        image_size=SIZE,
        max_steps=max_steps,
        faults=list(faults),
    )


def fault(kind: str, number: int, robot: int = 1, **params: Any) -> FaultSpec:
    robot_id = neutral_id(IdKind.robot, robot)
    component = {
        "toy_camera_shift": Component.scene_camera,
        "toy_image_noise": Component.wrist_camera,
        "toy_detector": Component.controller,
    }.get(kind)
    if component is not None:
        target = Target(scope=robot_id, component=component)
        nominal = None
    else:
        key = {
            "toy_delay": ConfigKey.control_action_delay_steps,
            "toy_knob": ConfigKey.policy_sampling_steps,
        }[kind]
        target = Target(scope=robot_id, config_key=key)
        nominal = 0 if kind == "toy_delay" else 2
    return FaultSpec(
        fault_id=neutral_id(IdKind.fault, number),
        kind=kind,
        params=params,
        level=1.0,
        target=target,
        schedule=Schedule(onset_day=0),
        nominal=nominal,
    )


def jobs(n: int, *, video_every: int = 0, faulted: bool = True) -> list[Job]:
    """A small mixed manifest over two tasks and three robots, some faulted."""
    out = []
    cam = fault("toy_camera_shift", 1, robot=2, pixels=2)
    delay = fault("toy_delay", 2, robot=3, steps=3)
    for k in range(n):
        robot = 1 + k % 3
        faults = {1: [], 2: [cam], 3: [delay]}[robot] if faulted else []
        video = bool(video_every) and k % video_every == 0
        out.append(
            Job(identity=ident(k + 1, robot=robot, task=k % 2, faults=faults), record_video=video)
        )
    return out


def context(
    policy: Any | None = None,
    store: MemoryStore | None = None,
    steps: Sequence[int] = (),
    teleop: Any | None = None,
    video: bool = False,
) -> EpisodeContext:
    policy = policy if policy is not None else toy.ToyPolicy()
    return EpisodeContext(
        fault_types=toy.TOY_FAULTS.get,
        make_check=toy.ToyCheck,
        source=DirectSource(policy),
        noise_shape=policy.noise_shape,
        noise_dtype=policy.noise_dtype,
        teleoperator=teleop,
        supervisors=make_supervisor,
        store=store or MemoryStore(),
        snapshot_steps=steps,
        record_video=video,
    )


def config(out: Path, **changes: Any) -> RunConfig:
    fields: dict[str, Any] = {
        "out": str(out),
        "device": "cpu",
        "precision": "fp32",
        "miopen": None,
        "workers": 3,
        "batch_size": 4,
        "env_factory": "fleetlab.engine.toy:ToyEnv",
        "policy_factory": "fleetlab.engine.toy:ToyPolicy",
        "check_factory": "fleetlab.engine.toy:ToyCheck",
        "fault_registry": "fleetlab.engine.toy:TOY_FAULTS",
        "teleoperator": "fleetlab.engine.toy:ToyTeleoperator",
        "life": 4,
        "server_threads": 1,
        "allow_manifest_day_length": True,
        "rerender_light": False,
    }
    fields.update(changes)
    return RunConfig(**fields)


def exit_after_flushing(code: int) -> None:
    """A deliberate crash must not accidentally poison a Queue's shared writer lock.

    Run the worker's normal multiprocessing finalizers before the immediate
    exit: they close and join every feeder this process started. Only the
    explicit poisoned-queue helper below is meant to leave a lock held.
    """
    import os
    from multiprocessing.util import _exit_function  # type: ignore[attr-defined]

    _exit_function()
    os._exit(code)


class CrashingToyEnv(toy.ToyEnv):
    """Kills its worker at reset, with a file counting deliberate losses across processes.

    The target defaults to e000003 and the loss count to one; ALWAYS ignores the count.
    """

    def reset(self, identity: Any, cell_hooks: Any) -> Any:
        import os

        if identity.episode_id == os.environ.get("FLEETLAB_TEST_CRASH_EPISODE", "e000003"):
            if os.environ.get("FLEETLAB_TEST_CRASH_ALWAYS"):
                exit_after_flushing(3)
            marker = os.environ.get("FLEETLAB_TEST_CRASH_ONCE")
            if marker:
                path = Path(marker)
                count = int(path.read_text()) if path.exists() else 0
                if count < int(os.environ.get("FLEETLAB_TEST_CRASH_COUNT", "1")):
                    path.write_text(str(count + 1))
                    exit_after_flushing(3)
        return super().reset(identity, cell_hooks)


def poisoned_queue_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Die on each assignment; on the last try die holding the shared queue's writer lock."""
    import os

    from fleetlab.engine.fleet import MAX_JOB_FAILURES

    job_q.put(("ask", wid, None))
    conn.recv()
    job_q.close()
    job_q.join_thread()  # the first two losses must leave the shared lock free
    if life_no == MAX_JOB_FAILURES - 1:
        job_q._wlock.acquire()
    os._exit(3)


class CrashingContinuationToyEnv(toy.ToyEnv):
    """Die once while restoring a continuation; other job kinds never restore a snapshot."""

    def restore(self, snapshot: Any) -> Any:
        import os

        path = Path(os.environ["FLEETLAB_TEST_CRASH_ONCE"])
        if not path.exists():
            path.write_text("crashed")
            exit_after_flushing(3)
        return super().restore(snapshot)


def closed_ask_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """The review's stall probe: close the receiving end between ask and assignment."""
    if life_no == 0:
        import time

        job_q.put(("ask", wid, None, life_no))
        conn.recv()
        conn.close()
        job_q.put(("ask", wid, None, life_no))
        job_q.close()
        job_q.join_thread()  # flush the ask before the parent detects a process exit
        time.sleep(120)  # recovery must kill us because the assignment send fails
        raise AssertionError("the parent did not recover the closed pipe")
    from fleetlab.engine.fleet import worker_main

    worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def closed_request_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """The review's inference probe: the request arrives after its receiving pipe is gone."""
    from fleetlab.engine import fleet

    if life_no == 0:

        def chunk(client: Any, request: Any) -> Any:
            import time

            client.conn.close()
            client.req_q.put(("req", client.wid, request, client.life_no))
            client.req_q.close()
            client.req_q.join_thread()
            time.sleep(120)  # keep exit detection from substituting for failed-send recovery
            raise AssertionError("the parent did not recover the closed pipe")

        fleet.ServerClient.chunk = chunk  # type: ignore[method-assign, assignment]
    fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def dispatcher_error_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """An invalid sample kills an unguarded dispatcher before any job is assigned."""
    job_q.put(("sample", wid, "invalid sample JSON", life_no))
    conn.recv()


def idle_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Wait before asking, so a monitor error must be reported without any assigned job."""
    conn.recv()


def zero_without_goodbye_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """A broken custom worker protocol: a clean exit without the normal goodbye."""
    exit_after_flushing(0)


def startup_loss_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Exit before asking, or leave a poisoned ask queue and an idle replacement."""
    import os

    if os.environ["FLEETLAB_TEST_STARTUP_LOSS"] == "exit":
        exit_after_flushing(3)
    if life_no == 0:
        job_q.put(("ask", wid, None, life_no))
        conn.recv()
        job_q.close()
        job_q.join_thread()
        job_q._wlock.acquire()
        os._exit(3)
    conn.recv()  # never asks: the startup watchdog must abort this life


def isolated_startup_loss_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Lose one empty life while another slot works, or before any slot asks."""
    import os
    import time

    mode = os.environ["FLEETLAB_TEST_EMPTY_LOSS"]
    if wid == 1 and life_no == 0:
        if mode in ("retire", "busy"):
            until = time.monotonic() + 30
            while not (Path(config.out) / "all-assigned").exists():
                assert time.monotonic() < until, "the other slot never took the work"
                time.sleep(0.01)
        exit_after_flushing(3)
    if wid == 0 and mode == "replace":
        until = time.monotonic() + 30
        while not (Path(config.out) / "replacement-started").exists():
            assert time.monotonic() < until, "the empty lost life was not replaced"
            time.sleep(0.01)
    from fleetlab.engine import fleet

    if wid == 0 and mode == "busy":
        original = fleet.ServerClient.chunk
        first = True

        def chunk(client: Any, request: Any) -> Any:
            nonlocal first
            result = original(client, request)
            if first:
                first = False
                until = time.monotonic() + 30
                while not (Path(config.out) / "replacement-started").exists():
                    assert time.monotonic() < until, "the lost empty slot was not replaced"
                    time.sleep(0.01)
            return result

        fleet.ServerClient.chunk = chunk  # type: ignore[method-assign, assignment]
    fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def normal_goodbye_during_loss_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Queue a normal goodbye while the parent is waiting on another worker's loss."""
    import os
    import time

    during_grace = bool(os.environ.get("FLEETLAB_TEST_EXIT_DURING_GRACE"))

    if wid == 0 and life_no == 0:
        job_q.put(("ask", wid, None, life_no))
        conn.recv()
        if during_grace:
            until = time.monotonic() + 30
            while not (Path(config.out) / "parent-waiting").exists():
                assert time.monotonic() < until, "the parent never waited for a request"
                time.sleep(0.01)
        conn.close()
        job_q.put(("ask", wid, None, life_no))
        job_q.close()
        job_q.join_thread()
        time.sleep(120)
        raise AssertionError("the closed pipe was not recovered")
    if wid == 1 and life_no == 0:
        until = time.monotonic() + 30
        while not (Path(config.out) / "loss-join").exists():
            assert time.monotonic() < until, "the other worker's loss was never handled"
            time.sleep(0.01)
        job_q.put(("ask", wid, None, life_no))
        assert conn.recv() is None, "the other slot should hold all the work"
        req_q.put(("bye", wid, {"worker": wid, "life": life_no, "jobs": 0}, life_no))
        if during_grace:
            req_q.close()
            req_q.join_thread()
            (Path(config.out) / "bye-queued").write_text("flushed")
            until = time.monotonic() + 30
            while not (Path(config.out) / "exit-now").exists():
                assert time.monotonic() < until, "the parent never sampled our ending"
                time.sleep(0.01)
        return  # multiprocessing flushes our goodbye before a status-0 exit
    from fleetlab.engine import fleet

    fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def large_shutdown_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Fill the request pipe during a policy call or the terminal error's grace wait.

    This helper intentionally leaves a world-size frame in flight when the
    parent shuts down. It is testing that specific condition, unlike the
    ordinary crash helpers which flush before exiting.
    """
    import os
    import time

    from fleetlab.engine import fleet
    from fleetlab.engine.episode import ChunkRequest

    mode = os.environ["FLEETLAB_TEST_LARGE_SHUTDOWN"]
    if wid == 0 and mode != "third":
        fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)
        return
    job_q.put(("ask", wid, None, life_no))
    payload, _ = conn.recv()
    job = Job.model_validate_json(payload)
    if wid == 0:
        req_q.put(("error", wid, "RuntimeError: deliberate reported loss", life_no))
        req_q.close()
        req_q.join_thread()
        if life_no == 2:
            time.sleep(120)  # keep the terminal grace wait open for the other feeder
        raise RuntimeError("deliberate reported loss")
    until = time.monotonic() + 30
    while not (Path(config.out) / "fill-request-pipe").exists():
        assert time.monotonic() < until, "the parent never entered shutdown's precursor"
        time.sleep(0.01)
    env = toy.ToyEnv(job.identity.suite, job.identity.task_index, job.identity.image_size)
    observation = env.reset(job.identity, [])
    assert sum(image.nbytes for image in observation["pixels"].values()) == 777_600
    request = ChunkRequest(
        key=job.key,
        observation=observation,
        instruction=env.task_description,
        noise=None,
        spec=toy.TOY_SPEC,
        row=None,
        batch_size=config.batch_size,
    )
    req_q.put(("req", wid, request, life_no))
    if mode == "dispatcher":
        job_q.put(("sample", wid, "invalid mid-run sample JSON", life_no))
        job_q.close()
        job_q.join_thread()
    conn.recv()


def reported_error_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Report under a held feeder lock, then exit normally or refuse to exit."""
    import os
    import time

    if life_no == 0:
        job_q.put(("ask", wid, None, life_no))
        conn.recv()
        job_q.close()
        job_q.join_thread()
        original = req_q._send_bytes

        def delayed_send(data: Any) -> None:
            original(data)
            time.sleep(0.25)  # the parent sees the report while the feeder owns the lock

        req_q._send_bytes = delayed_send
        req_q.put(("error", wid, "RuntimeError: reported worker failure", life_no))
        req_q.close()
        req_q.join_thread()
        if os.environ.get("FLEETLAB_TEST_REPORT_HANG"):
            conn.recv()
        raise RuntimeError("reported worker failure")
    assert req_q._wlock.acquire(False), "the parent killed the reporting feeder"
    req_q._wlock.release()
    from fleetlab.engine.fleet import worker_main

    worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def request_ending_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """End while the parent serves a request; its own report will arrive after detection."""
    import os
    import time

    from fleetlab.engine import fleet

    if life_no == 0:
        original = fleet.ServerClient.chunk
        mode = os.environ["FLEETLAB_TEST_REQUEST_END"]
        seen: set[str] = set()

        def chunk(client: Any, request: Any) -> Any:
            seen.add(request.key)
            if mode == "done" and len(seen) < 2:
                return original(client, request)
            job_q.close()
            job_q.join_thread()
            client.req_q.put(("req", wid, request, life_no))
            until = time.monotonic() + 30
            while not (Path(config.out) / "policy-entered").exists():
                assert time.monotonic() < until, "the parent never served the request"
                time.sleep(0.01)
            if mode == "error":
                raise RuntimeError("the disk is full while writing a job")
            if mode == "done":
                client.run_deferred()
            exit_after_flushing(3)

        fleet.ServerClient.chunk = chunk  # type: ignore[method-assign, assignment]
    fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def missing_done_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Write a job's files but omit its done report, then send a normal goodbye."""
    original = req_q.put

    def omit_done(message: Any) -> None:
        if message[0] != "done":
            original(message)

    req_q.put = omit_done
    from fleetlab.engine.fleet import worker_main

    worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


def retired_slot_worker(
    wid: int, life_no: int, config: Any, noise: Any, sample: Any, req_q: Any, job_q: Any, conn: Any
) -> None:
    """Announce work and forge an ask for another slot after the parent retired it."""
    import time

    from fleetlab.engine import fleet

    if wid == 0 and life_no == 0:
        original = fleet.ServerClient.chunk
        calls = 0

        def chunk(client: Any, request: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                until = time.monotonic() + 30
                while not (Path(config.out) / "retired").exists():
                    assert time.monotonic() < until, "the other slot was never retired"
                    time.sleep(0.01)
                extra = jobs(3)[2]
                job_q.put(("spawn", wid, extra.model_dump_json(), life_no))
                job_q.put(("ask", 1, None, 0))
            return original(client, request)

        fleet.ServerClient.chunk = chunk  # type: ignore[method-assign, assignment]
    fleet.worker_main(wid, life_no, config, noise, sample, req_q, job_q, conn)


class GlitchToyEnv(toy.ToyEnv):
    """Renders one frame differently once: the toy's render transient.

    FLEETLAB_TEST_GLITCH is "DIR;EPISODE:STEP:CAMERA[;...]": on the second
    reset of EPISODE across all processes (counted by files in DIR), the image
    of CAMERA after STEP gets one pixel changed. The first rendering (the run)
    and every later one (the re-rendering) are the toy's own.
    """

    def reset(self, identity: Any, cell_hooks: Any) -> Any:
        import os

        self._glitch: tuple[int, str] | None = None
        spec = os.environ.get("FLEETLAB_TEST_GLITCH", "")
        if spec:
            folder, *targets = spec.split(";")
            for target in targets:
                episode, step, camera = target.split(":")
                if episode != identity.episode_id:
                    continue
                count = len(list(Path(folder).glob(f"{episode}.*")))
                (Path(folder) / f"{episode}.{count + 1}").write_text("reset")
                if count + 1 == 2:
                    self._glitch = (int(step), camera)
        return super().reset(identity, cell_hooks)

    def step(self, action: Any) -> Any:
        from fleetlab.contracts.env import PIXEL_KEYS, Camera

        result = super().step(action)
        glitch = getattr(self, "_glitch", None)
        if glitch is not None and self._step == glitch[0]:
            key = PIXEL_KEYS[Camera(glitch[1])]
            pixels = dict(result.observation["pixels"])
            image = pixels[key].copy()
            image[0, 0, 0] ^= 1
            pixels[key] = image
            result.observation["pixels"] = pixels
        return result
