"""Planning the oracle's replays: which fault lists each scoring episode is replayed under,
as a resumable manifest, with its counts before anything runs.

Per scoring episode (contracts section 9): the episode's faults split into the
**toggled** ones (its effective faults) and the **excluded** ones (decoys and inert
confounder entries, `WorldSpec.patterns.excluded_from_replay`).

- **Contract replays**: every non-empty subset of the toggled faults removed, the
  excluded ones in force (`contracts.oracle.power_set_replays`), 2^n - 1 for n toggled
  faults. An episode with more than `POWER_SET_CAP` toggled faults gets none and is
  listed as above the cap; its lookups are fresh replays (`fleetlab.oracle.fresh`).
- **Confirmation replays** (the miniature fleet's addition, kept), in one of three modes:
  `full` (the default): every one of those removal sets, and the empty one, once more
  with each non-empty subset of the episode's excluded faults removed as well, so the
  decoys and the inert entries are confirmed by replay, every action on them alone or
  with others is a lookup, and the episode's all-faults-removed row exists (claim 5);
  with one excluded fault that is 2^n more replays, the miniature fleet's 65 and 65.
  `single`: each excluded fault removed alone, and every fault removed, m + 1 replays
  for m excluded faults: enough to confirm them and for claim 5, while an action on a
  decoy together with others becomes a fresh replay. `none`: the contract's alone.
- **Proof replays**: when the replays run in a directory of their own, the engine's
  chain sample has no "run" job to replay there, so the plan adds the contract's sample
  itself: unmodified replays of every tenth scoring episode, at least 20
  (`contracts.rollout.chain_sample`), whose chains must equal the as-run chains in the
  world's runs (`fleetlab.oracle.proof`).

The manifest is `oracle/jobs.jsonl` (replays, then proof replays) with `oracle/plan.json`
beside it (counts, world hash, the manifest's SHA-256). Planning again over the same world
writes nothing new; a plan that differs from the stored one is refused unless replaced on
purpose, because the replays already run belong to the stored plan.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import EpisodeIdentity, seeds_for
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.oracle import POWER_SET_CAP, fault_list_hash, power_set_replays
from fleetlab.contracts.rollout import Job, JobMode, chain_sample
from fleetlab.contracts.world import WorldSpec
from fleetlab.oracle.world import OracleLayout, OracleWorld

# Replays per hour of the queue, for the time estimate: the oracle's own w10 run with four
# workers (1,868 per hour end to end, the memory guard's choice that morning) and with
# ten (the miniature fleet's later run, 3,199 per hour). The range is what the estimate
# prints.
PER_HOUR_SLOW = 2_000
PER_HOUR_FAST = 3_000

Confirm = Literal["full", "single", "none"]
CONFIRM_MODES: tuple[Confirm, ...] = ("full", "single", "none")


def confirm_mode(confirm: bool | str) -> Confirm:
    """The confirmation mode: True is "full", False is "none"."""
    if confirm is True:
        return "full"
    if confirm is False:
        return "none"
    if confirm not in CONFIRM_MODES:
        raise ValueError(f"confirmation is one of {CONFIRM_MODES}, not {confirm!r}")
    return confirm


@dataclass(frozen=True)
class EpisodeReplays:
    """One scoring episode's replays."""

    episode_id: str
    toggled: tuple[str, ...]
    excluded: tuple[str, ...]
    contract: tuple[frozenset[str], ...]  # removal sets of the contract's power set
    confirmation: tuple[frozenset[str], ...]  # removal sets that touch an excluded fault
    over_cap: bool


@dataclass(frozen=True)
class OraclePlan:
    """The replay jobs of a world's scoring window and their counts."""

    jobs: list[Job]  # contract replays, then confirmation replays, per episode
    contract: int
    confirmation: int
    over_cap: list[str]  # scoring episodes with more toggled faults than the cap
    episodes: list[EpisodeReplays] = field(default_factory=list)
    proof: list[Job] = field(default_factory=list)  # unmodified replays (a run of its own)
    cap: int = POWER_SET_CAP
    confirm: Confirm = "full"

    @property
    def replays(self) -> int:
        return self.contract + self.confirmation

    @property
    def total(self) -> int:
        """Every job of the manifest: replays and proof replays."""
        return len(self.jobs) + len(self.proof)

    def manifest(self) -> list[Job]:
        return [*self.jobs, *self.proof]

    def toggled_histogram(self) -> dict[int, int]:
        """Scoring episodes by their number of toggled faults."""
        counts = Counter(len(e.toggled) for e in self.episodes)
        return dict(sorted(counts.items()))

    def counts(self) -> dict[str, Any]:
        n = len(self.episodes)
        return {
            "scoring_episodes": n,
            "contract": self.contract,
            "confirmation": self.confirmation,
            "replays": self.replays,
            "proof": len(self.proof),
            "jobs": self.total,
            "per_scoring_episode": round(self.replays / n, 2) if n else None,
            "over_cap": list(self.over_cap),
            "toggled_faults": {str(k): v for k, v in self.toggled_histogram().items()},
            "cap": self.cap,
            "confirm_excluded": self.confirm,
            "hours": hours(self.total),
        }


def hours(jobs: int) -> dict[str, float]:
    """The queue time of `jobs` replays at the two rates of the estimate, in hours."""
    return {
        f"at_{PER_HOUR_SLOW}_per_hour": round(jobs / PER_HOUR_SLOW, 2),
        f"at_{PER_HOUR_FAST}_per_hour": round(jobs / PER_HOUR_FAST, 2),
    }


def _subsets(items: Sequence[str], *, empty: bool) -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    for size in range(0 if empty else 1, len(items) + 1):
        out += [frozenset(c) for c in itertools.combinations(items, size)]
    return out


def episode_replays(
    identity: EpisodeIdentity,
    excluded: Iterable[str],
    *,
    confirm: bool | str = True,
    cap: int = POWER_SET_CAP,
) -> EpisodeReplays:
    """The removal sets one episode is replayed under (module docstring)."""
    leave = frozenset(excluded)
    every = identity.fault_ids
    toggled = tuple(f.fault_id for f in identity.faults if f.fault_id not in leave)
    held = tuple(f.fault_id for f in identity.faults if f.fault_id in leave)
    lists = power_set_replays(identity, leave, cap)
    if lists is None:
        return EpisodeReplays(identity.episode_id, toggled, held, (), (), True)
    contract = [every - {f.fault_id for f in kept} for kept in lists]
    confirmation: list[frozenset[str]] = []
    mode = confirm_mode(confirm)
    seen = set(contract)
    wanted: list[frozenset[str]] = []
    if mode == "full" and held:
        bases = [*contract, frozenset()]  # every removal of the toggled ones, then none
        wanted = [frozenset(b | extra) for extra in _subsets(held, empty=False) for b in bases]
    elif mode == "single" and held:
        wanted = [*(frozenset({c}) for c in held), frozenset(every)]
    for remove in wanted:
        if remove not in seen:
            seen.add(remove)
            confirmation.append(remove)
    return EpisodeReplays(
        identity.episode_id, toggled, held, tuple(contract), tuple(confirmation), False
    )


def plan_replays(
    identities: Sequence[EpisodeIdentity],
    excluded: Iterable[str],
    *,
    confirm: bool | str = True,
    cap: int = POWER_SET_CAP,
    proof: bool = False,
) -> OraclePlan:
    """The replay jobs of these scoring episodes; with `proof`, the proof replays as well."""
    leave = frozenset(excluded)
    jobs: list[Job] = []
    episodes: list[EpisodeReplays] = []
    contract = confirmation = 0
    for identity in identities:
        plan = episode_replays(identity, leave, confirm=confirm, cap=cap)
        episodes.append(plan)
        for remove in [*plan.contract, *plan.confirmation]:
            jobs.append(Job(identity=identity, mode=JobMode.replay, remove=sorted(remove)))
        contract += len(plan.contract)
        confirmation += len(plan.confirmation)
    over = [e.episode_id for e in episodes if e.over_cap]
    proofs = proof_jobs(identities) if proof else []
    return OraclePlan(
        jobs, contract, confirmation, over, episodes, proofs, cap, confirm_mode(confirm)
    )


def proof_jobs(identities: Sequence[EpisodeIdentity]) -> list[Job]:
    """Unmodified replays of every tenth episode, at least 20 (`contracts.rollout.chain_sample`)."""
    by_id = {i.episode_id: i for i in identities}
    picked = chain_sample([i.episode_id for i in identities])
    return [Job(identity=by_id[e], mode=JobMode.replay) for e in picked]


def plan_world(
    world: OracleWorld,
    *,
    confirm: bool | str = True,
    cap: int = POWER_SET_CAP,
    proof: bool = True,
) -> OraclePlan:
    """The world's oracle plan over its scoring window."""
    return plan_replays(world.scoring(), world.excluded, confirm=confirm, cap=cap, proof=proof)


# ---- checks without the simulator ----------------------------------------------------------


def check_jobs(world: OracleWorld, jobs: Sequence[Job]) -> list[str]:
    """What is wrong with a replay manifest for this world; empty when it is sound.

    Every job validates as a `Job` (the identity's own checks: faults active on its
    day and above level zero, a configuration fault with its nominal value), is a
    replay from the reset, carries exactly the as-run identity of a scoring episode
    of the world, removes only that episode's faults, adds nothing, has a unique key
    and a unique effective fault list per episode, and the identity's seeds are the
    ones `seeds_for` derives from the world seed.
    """
    problems: list[str] = []
    scoring = {i.episode_id: i for i in world.scoring()}
    keys: set[str] = set()
    lists: set[tuple[str, str]] = set()
    seeds: dict[str, bool] = {}
    for n, job in enumerate(jobs, start=1):
        where = f"job {n} ({job.identity.episode_id})"
        try:
            again = Job.model_validate_json(job.model_dump_json())
        except ValueError as exc:
            problems.append(f"{where}: does not validate: {exc}")
            continue
        if again != job:
            problems.append(f"{where}: changes in a JSON round trip")
        if job.mode is not JobMode.replay or job.from_snapshot is not None:
            problems.append(f"{where}: not a replay from the reset ({job.mode.value})")
        if job.add:
            problems.append(f"{where}: adds changes; the planned replays only remove faults")
        identity = scoring.get(job.identity.episode_id)
        if identity is None:
            problems.append(f"{where}: not a scoring episode of {world.world_id}")
            continue
        if job.identity != identity:
            problems.append(f"{where}: its identity differs from the world's manifest")
        if job.key in keys:
            problems.append(f"{where}: key {job.key} twice")
        keys.add(job.key)
        listed = (job.identity.episode_id, fault_list_hash(job.effective_faults))
        if listed in lists and job.remove:
            problems.append(f"{where}: the same effective fault list twice")
        lists.add(listed)
        episode = job.identity.episode_id
        if episode not in seeds:
            want = seeds_for(
                world.spec.seed, identity.robot_id, identity.day, identity.episode_index
            )
            seeds[episode] = (identity.scene_seed, identity.noise_seed) == want
            if not seeds[episode]:
                problems.append(f"{where}: seeds are not seeds_for(world seed, robot, day, index)")
    return problems


# ---- the manifest on disk --------------------------------------------------------------------


class PlanChanged(RuntimeError):
    """A stored plan differs from the one the world gives now."""


def manifest_text(jobs: Sequence[Job]) -> str:
    return "".join(j.model_dump_json() + "\n" for j in jobs)


def plan_record(world: OracleWorld, plan: OraclePlan, text: str) -> dict[str, Any]:
    return {
        "world_id": world.world_id,
        "world_hash": world.world_hash,
        "jobs_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "excluded": sorted(world.excluded),
        "decoys": sorted(world.decoys),
        "inert": sorted(world.inert),
        **plan.counts(),
    }


def write_plan(
    layout: OracleLayout, world: OracleWorld, plan: OraclePlan, *, replace: bool = False
) -> bool:
    """Write the manifest and its record; False if the same plan was already there.

    Raises `PlanChanged` if another plan is stored, unless `replace`; and always once the
    stored plan's replays have run (`layout.run` exists).
    """
    text = manifest_text(plan.manifest())
    record = plan_record(world, plan, text)
    if layout.jobs.is_file():
        if layout.jobs.read_text() == text:
            if not layout.plan.is_file():
                _write(layout.plan, json.dumps(record, indent=1) + "\n")
            return False
        if (layout.run / "run.json").is_file():
            raise PlanChanged(
                f"{layout.jobs} holds another plan whose replays have run in {layout.run}; "
                "plan into another oracle directory (the old replays would mix in by job key)"
            )
        if not replace:
            raise PlanChanged(
                f"{layout.jobs} holds another plan (the world or the planning changed); "
                "replace it on purpose, or plan into another oracle directory"
            )
    _write(layout.jobs, text)
    _write(layout.plan, json.dumps(record, indent=1) + "\n")
    return True


def read_manifest(path: Path) -> list[Job]:
    return [Job.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


# ---- an estimate from the spec alone -----------------------------------------------------------


def estimate_from_spec(
    spec: WorldSpec, *, confirm: bool | str = True, cap: int = POWER_SET_CAP
) -> dict[str, Any]:
    """The replay count from a spec before the world is generated: an upper bound.

    Robots are numbered site by site (`r01..` at `s1`, as the generators do), every
    robot runs `episodes_per_robot_day` episodes a day, and a fault counts on a robot's
    day when its target covers the robot and its schedule is active that day. Ignored:
    configuration shadowing (a robot's own setting hides a wider fault, which only
    lowers the count) and drift levels (a drifting fault counts on every active day).
    """
    excluded = spec.patterns.excluded_from_replay()
    mode = confirm_mode(confirm)
    replays = contract = over = 0
    episodes = 0
    histogram: Counter[int] = Counter()
    for s in range(spec.sites):
        site = neutral_id(IdKind.site, s + 1)
        for r in range(spec.robots_per_site):
            robot = neutral_id(IdKind.robot, s * spec.robots_per_site + r + 1)
            for day in range(spec.windows.scoring.first, spec.windows.scoring.last + 1):
                active: list[FaultSpec] = [
                    f
                    for f in spec.fault_schedule
                    if f.target.covers(robot, site) and f.schedule.active_on(day)
                ]
                n = sum(f.fault_id not in excluded for f in active)
                m = sum(f.fault_id in excluded for f in active)
                count = spec.episodes_per_robot_day
                episodes += count
                histogram[n] += count
                if n > cap:
                    over += count
                    continue
                contract += count * (2**n - 1)
                replays += count * (2**n - 1 + confirmations(n, m, mode))
    proof = len(chain_sample([str(i) for i in range(episodes)]))
    return {
        "world_id": spec.world_id,
        "scoring_episodes": episodes,
        "contract": contract,
        "confirmation": replays - contract,
        "replays": replays,
        "proof": proof,
        "jobs": replays + proof,
        "over_cap_episodes": over,
        "toggled_faults": {str(k): v for k, v in sorted(histogram.items())},
        "hours": hours(replays + proof),
        "confirm_excluded": mode,
        "bound": "upper: configuration shadowing ignored",
    }


def confirmations(n: int, m: int, mode: Confirm) -> int:
    """Confirmation replays of an episode with n toggled and m excluded faults."""
    if m == 0 or mode == "none":
        return 0
    if mode == "full":
        return int(2 ** (n + m) - 2**n)
    return m + (0 if n == 0 and m == 1 else 1)
