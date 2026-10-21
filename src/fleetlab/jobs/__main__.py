"""The operator's command line for jobs (docs/design/jobs.md).

    python -m fleetlab.jobs run <job> --library LIB --starts release|dev --n N --out DIR
        [--video] [--frame-every K] [--first F] [--no-naive] [--policy xvla|hold]
    python -m fleetlab.jobs check <check> --library LIB --run RUN_DIR [--param k=v ...]
    python -m fleetlab.jobs list --library LIB [--kind job|check|recipe]
    python -m fleetlab.jobs replay --run RUN_DIR --episode E --out DIR [--size S]
        [--camera C] [--frame-every K]
    python -m fleetlab.jobs save-job <name> --library LIB --run RUN_DIR --job-id J
        [--acceptance N,K] [--replace]

`run` plays a saved job and, beside it on the same starts, the host's naive chain of its
requested goals (`runner.naive_program`): job j0001 (episodes e000101 on) and j0002
(e000201 on) in DIR. The starts are the saved job's base task's, derived under the job's
allocation profile (`jobs.starts`), as the request it was saved for derived them. On
release starts (the reserved initial states 30 on) it first writes the attempt with its
full start identities to the library's release exposure log, then runs, then writes
DIR/receipt.json, a `ReleaseReceipt`: the program's and the naive chain's deliveries,
rescued and broken, the order rules. `passed` needs a run of exactly the approved N (at
least K delivered); `released` needs passed, every order rule kept, and starts that no
earlier attempt examined (`starts_fresh`: the full identities, for any request, finished
or not, whatever program it ran). The first receipt of a run of the approved size is stored
with the job in the library. On development starts `run` writes DIR/summary.json. `run`
and `replay` simulate: run them through `tools/heavy-run`, `run` with X-VLA as a heavy
job.

`check` runs a saved check read-only over a run's tables and prints its rows as JSON.
`save-job` saves the program of a recorded job with its request's goals and rules, refusing
a program whose recorded runs broke an order rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.contracts.jobs import (
    DEFAULT_LEAD_FRACTION,
    Acceptance,
    JobBudget,
    JobEpisode,
    JobProgram,
    LibraryKind,
    ReleasePair,
    ReleaseReceipt,
    RequestState,
    SavedJob,
    acceptance_problems,
)
from fleetlab.jobs import library, tables

PROGRAM_JOB = "j0001"
NAIVE_JOB = "j0002"


def _state(job: SavedJob) -> RequestState:
    ceilings = JobBudget(
        starts_per_request=20, requests_per_session=4, sessions=6, starts_per_investigation=400
    )
    return RequestState(
        request_text=f"operator run of the saved job {job.name}",
        requested_goals=job.requested_goals,
        base_task=job.base_task,
        order_rules=job.order_rules,
        budget=ceilings,
        acceptance=job.acceptance,
        message_log=[],
        allocation_profile=job.allocation_profile,
    )


def release_receipt(
    job: SavedJob,
    naive: JobProgram,
    program_runs: Sequence[JobEpisode],
    naive_runs: Sequence[JobEpisode],
    run_dir: Path,
    *,
    fresh: bool,
) -> ReleaseReceipt:
    """The receipt of a release run: both programs on the same release starts, paired.

    `fresh` says whether none of these starts had been examined by an earlier attempt (the
    library's exposure log, read before this attempt ran). Every start must be the one the
    job's allocation profile gives for its base task and initial state: a run from another
    identity with the same index is refused, not counted.
    """
    from fleetlab.jobs.starts import start_for

    pairs = []
    for p, q in zip(program_runs, naive_runs, strict=True):
        if p.start != q.start:
            raise ValueError(f"{p.episode_id} and {q.episode_id} ran from different starts")
        if p.start != start_for(job.base_task, p.start.init_state, profile=job.allocation_profile):
            raise ValueError(
                f"{p.episode_id} ran from a start the job's allocation does not give "
                f"(task {p.start.base_task}, initial state {p.start.init_state})"
            )
        if p.program_digest != job.program.digest:
            raise ValueError(f"{p.episode_id} ran another program than {job.name}'s")
        if q.program_digest != naive.digest:
            raise ValueError(f"{q.episode_id} ran another program than the naive chain")
        pairs.append(
            ReleasePair(
                start=p.start,
                episode_id=p.episode_id,
                delivered=p.delivered,
                order_kept=p.order_kept,
                naive_episode_id=q.episode_id,
                naive_delivered=q.delivered,
            )
        )
    n = len(pairs)
    delivered = sum(p.delivered for p in pairs)
    acceptance = job.acceptance
    full = acceptance is not None and n == acceptance.N
    passed = (delivered >= acceptance.K) if (acceptance is not None and full) else None
    violations = sum(not p.order_kept for p in pairs)
    return ReleaseReceipt(
        job_name=job.name,
        program_digest=job.program.digest,
        naive_digest=naive.digest,
        release_starts=[p.start for p in pairs],
        N=n,
        K=acceptance.K if acceptance is not None else None,
        delivered=delivered,
        naive_delivered=sum(p.naive_delivered for p in pairs),
        rescued=sum(p.delivered and not p.naive_delivered for p in pairs),
        broken=sum(p.naive_delivered and not p.delivered for p in pairs),
        order_violations=violations,
        passed=passed,
        released=bool(passed) and violations == 0 and fresh,
        pairs=pairs,
        run_dir=str(run_dir),
        starts_fresh=fresh,
        lead_fraction=job.lead_fraction,
    )


def cmd_run(args: argparse.Namespace) -> int:
    from fleetlab.jobs.runner import JobRunner, make_source, naive_program
    from fleetlab.jobs.starts import development_starts_for, release_starts_for

    lib, out = Path(args.library), Path(args.out)
    job = library.load_job(lib, args.job)
    release = args.starts == "release"
    if release and args.no_naive:
        raise SystemExit("a release run plays the naive chain beside the job")
    n = args.n or (job.acceptance.N if job.acceptance else None)
    if n is None:
        raise SystemExit("--n is needed: the job has no approved acceptance condition")
    profile = job.allocation_profile
    starts = (
        release_starts_for(job.base_task, n, profile=profile)
        if release
        else development_starts_for(job.base_task, n, args.first, profile=profile)
    )
    if tables.read_episodes(out):
        raise SystemExit(f"{out} holds a run already")
    out.mkdir(parents=True, exist_ok=True)
    attempt = None
    fresh = False
    if release:
        examined = library.exposed_identities(lib)
        seen = sorted(s.init_state for s in starts if s in examined)
        fresh = not seen
        if seen:
            print(
                f"{job.name}: release starts {seen} were examined by an earlier attempt; "
                "this attempt is reported and cannot be released",
                flush=True,
            )
        attempt = library.record_release_exposure(lib, job, starts, out)  # before it runs
    source, soft = make_source(args.policy)
    runner = JobRunner(
        source, soft, frame_every=args.frame_every, log=lambda line: print(line, flush=True)
    )
    state = _state(job)
    try:
        naive = naive_program(job.requested_goals, runner.suite.instructions())
        program_runs = runner.run_starts(
            out,
            PROGRAM_JOB,
            job.program,
            state,
            starts,
            record_video=args.video,
            first_episode=101,
            allow_release=release,
        )
        naive_runs: list[JobEpisode] = []
        if not args.no_naive:
            naive_runs = runner.run_starts(
                out,
                NAIVE_JOB,
                naive,
                state,
                starts,
                record_video=args.video,
                first_episode=201,
                allow_release=release,
            )
    finally:
        runner.close()
    if release:
        assert attempt is not None
        receipt = release_receipt(job, naive, program_runs, naive_runs, out, fresh=fresh)
        (out / "receipt.json").write_text(receipt.model_dump_json(indent=1) + "\n")
        library.finish_release_exposure(lib, attempt, receipt)
        stored = receipt.passed is not None and job.receipt is None
        if stored:
            library.save_job(lib, job.model_copy(update={"receipt": receipt}), replace=True)
        print(
            f"{job.name}: delivered {receipt.delivered} of {receipt.N} release starts "
            f"(K {receipt.K}); naive chain {receipt.naive_delivered}; rescued {receipt.rescued}, "
            f"broken {receipt.broken}; order violations {receipt.order_violations}; "
            f"passed {receipt.passed}, released {receipt.released}"
            + ("; receipt stored in the library" if stored else ""),
            flush=True,
        )
        return 0
    summary = {
        "job_name": job.name,
        "program_digest": job.program.digest,
        "starts": [s.model_dump(mode="json") for s in starts],
        "delivered": sum(e.delivered for e in program_runs),
        "naive_delivered": None if args.no_naive else sum(e.delivered for e in naive_runs),
        "episodes": [e.episode_id for e in program_runs],
        "naive_episodes": [e.episode_id for e in naive_runs],
        "order_violations": sum(not e.order_kept for e in program_runs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    spec = library.load_check(Path(args.library), args.check)
    params = library.parse_params(spec, args.param)
    rows = library.run_check(spec, Path(args.run), params, max_rows=args.max_rows)
    print(
        json.dumps(
            {
                "check": spec.name,
                "purpose": spec.purpose,
                "expects": spec.expects,
                "parameters": params,
                "rows": rows,
            },
            indent=1,
            default=str,
        )
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    kind = LibraryKind(args.kind) if args.kind else None
    for entry in library.list_entries(Path(args.library), kind):
        print(f"{entry.kind.value:<7} {entry.name:<40} {entry.digest[:12]}  {entry.purpose}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from fleetlab.jobs.replay import rerender

    report = rerender(
        Path(args.run),
        args.episode,
        Path(args.out),
        image_size=args.size,
        camera=args.camera,
        frame_every=args.frame_every,
    )
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_verdict")}, indent=1))
    return 0 if report["accepted"] else 1


def saved_job_from_run(
    run_dir: Path,
    job_id: str,
    name: str,
    acceptance: Acceptance | None = None,
    fraction: tuple[int, int] = DEFAULT_LEAD_FRACTION,
) -> SavedJob:
    """A recorded job's program with its request's goals, rules and allocation profile (the
    run's own recorded request), as the library keeps it.

    Refused when any recorded run of the program broke an order rule.
    """
    record: dict[str, Any] = json.loads((run_dir / tables.RUN_RECORD).read_text())
    state = RequestState.model_validate(record["state"])
    approved = state.acceptance
    if acceptance is not None and approved is not None and acceptance != approved:
        raise SystemExit(
            f"the request approved N={approved.N}, K={approved.K}; a saved job keeps the "
            "approved condition"
        )
    chosen = approved if approved is not None else acceptance
    if chosen is not None:
        problems = acceptance_problems(chosen, fraction)
        if problems:
            raise SystemExit("not an acceptance the lead approves: " + "; ".join(problems))
    entry = next((j for j in record.get("jobs", []) if j.get("job_id") == job_id), None)
    if entry is None:
        raise SystemExit(f"{run_dir} has no job {job_id}")
    program = JobProgram.model_validate(entry["program"])
    runs = [e for e in tables.read_episodes(run_dir) if e.job_id == job_id]
    if not runs:
        raise SystemExit(f"{job_id} has no recorded runs in {run_dir}")
    broken = [e.episode_id for e in runs if not e.order_kept]
    if broken:
        raise SystemExit(f"{job_id}'s runs {', '.join(broken)} broke an order rule: not saved")
    return SavedJob(
        name=name,
        program=program,
        requested_goals=state.requested_goals,
        base_task=state.base_task,
        order_rules=state.order_rules,
        acceptance=chosen,
        saved_from=[job_id],
        receipt=None,
        lead_fraction=list(fraction),
        allocation_profile=state.allocation_profile,
    )


def cmd_save_job(args: argparse.Namespace) -> int:
    acceptance = None
    if args.acceptance:
        n, k = (int(x) for x in args.acceptance.split(","))
        acceptance = Acceptance(N=n, K=k)
    a, b = (int(x) for x in args.lead_fraction.split("/"))
    job = saved_job_from_run(Path(args.run), args.job_id, args.name, acceptance, (a, b))
    path = library.save_job(Path(args.library), job, replace=args.replace)
    print(f"saved {job.name} ({job.program.digest[:12]}) to {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.jobs", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a saved job, and the naive chain beside it")
    run.add_argument("job")
    run.add_argument("--library", required=True)
    run.add_argument("--starts", choices=("release", "dev"), required=True)
    run.add_argument("--n", type=int)
    run.add_argument("--first", type=int, default=0, help="first development start")
    run.add_argument("--out", required=True)
    run.add_argument("--video", action="store_true")
    run.add_argument("--frame-every", type=int, default=4)
    run.add_argument("--no-naive", action="store_true")
    run.add_argument("--policy", default="xvla", choices=("xvla", "hold"))
    run.set_defaults(func=cmd_run)
    check = sub.add_parser("check", help="run a saved check over a run's tables")
    check.add_argument("check")
    check.add_argument("--library", required=True)
    check.add_argument("--run", required=True)
    check.add_argument("--param", action="append", default=[])
    check.add_argument("--max-rows", type=int, default=500)
    check.set_defaults(func=cmd_check)
    listing = sub.add_parser("list", help="the library's jobs, checks and recipes")
    listing.add_argument("--library", required=True)
    listing.add_argument("--kind", choices=[k.value for k in LibraryKind])
    listing.set_defaults(func=cmd_list)
    replay = sub.add_parser("replay", help="re-render a recorded episode, verified")
    replay.add_argument("--run", required=True)
    replay.add_argument("--episode", required=True)
    replay.add_argument("--out", required=True)
    replay.add_argument("--size", type=int)
    replay.add_argument("--camera", default="scene")
    replay.add_argument("--frame-every", type=int, default=1)
    replay.set_defaults(func=cmd_replay)
    save = sub.add_parser("save-job", help="save a recorded job's program to the library")
    save.add_argument("name")
    save.add_argument("--library", required=True)
    save.add_argument("--run", required=True)
    save.add_argument("--job-id", required=True)
    save.add_argument(
        "--acceptance", help="N,K when the run's request has none; never a different one"
    )
    save.add_argument(
        "--lead-fraction", default="4/5", help="the lead's bar the acceptance was approved under"
    )
    save.add_argument("--replace", action="store_true")
    save.set_defaults(func=cmd_save_job)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
