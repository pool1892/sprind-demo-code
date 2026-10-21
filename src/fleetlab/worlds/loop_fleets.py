"""Loop fleets: designed development fleets for the two-round improvement loop (lane fleet).

    uv run --locked python -m fleetlab.worlds.loop_fleets check LEAD        # CPU: design, forecast
    uv run --locked python -m fleetlab.worlds.loop_fleets generate LEAD     # CPU: the world files
    tools/heavy-run uv run --locked python -m fleetlab.worlds.loop_fleets run LEAD
        # round 1's operation (the observed window): engine run, records, events, export
    uv run --locked python -m fleetlab.worlds.loop_fleets report LEAD       # success, truth table

docs/design/loop.md and the plan's lead fleet: twelve robots at three sites, a lasting
condition at one site from day 0, causes in the change log, physical causes in no log, a
harmless change that looks suspicious, one place where two changes coincide and the records
cannot say which matters, and more worth doing than three changes can fix.

Why a builder of its own. The calibrated composer (`fleetlab.worlds.compose`) searches fault
placements until a forecast lands in a band, and its measured estimator refuses every joint
context nobody measured. A loop fleet is designed instead: each fault has its part in the
story. So the schedule is stated here, and everything else is the version-1 generator's,
unchanged: the change log's rule (a configuration or software fault writes its change; benign
software patches, camera calibration records and maintenance come on top,
`fleetlab.worlds.world.changes`/`maintenance`), the health model (`worlds.health`), the
slots with each site's task emphasis (`worlds.base.slots`), the episode identities and seeds
(`worlds.world.jobs`), and the world files with their hash. `generator.json` names the
version-1 generator, so `fleetlab.loop.world.LoopWorld` derives health under a ledger and the
reserved panel's tasks with the same functions.

Nothing about an effect is planted. `forecast` is design guidance from measured calibration
cells (the exact joint where one was measured, otherwise the members' suite-level ratios
multiplied); the run's physics decides, and `report` shows what it decided.

The loop's layout is the stated one: round 1's operation is the observed window, its
experiments the experiment window, round 2's operation the scoring window; round 2's
experiment days and the reserved evaluation follow past the stored horizon. It equals
`fleetlab.loop.layout.layout_for_world` of the spec, which the driver uses, and is written
beside the world as `loop_layout.json`.

Truth side only: the recipes below and everything this module writes stay out of every
export; the world root is never mounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import fleetlab  # noqa: F401  (torch before anything that imports mujoco)
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.contracts.model import Fleet, describe_for_agents
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import LATER_CONFIG_KEYS
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec, seal_world
from fleetlab.worlds import base
from fleetlab.worlds.generate import World
from fleetlab.worlds.truth import FAULT_TABLE_FILE, FaultTable, FaultTruth, Role

GENERATOR = "fleetlab.worlds.loop_fleets v1"
WINDOW_NAMES = ("observed", "experiment", "scoring")
TASKS: tuple[Task, ...] = tuple(Task("libero_spatial", i) for i in range(10)) + tuple(
    Task("libero_object", i) for i in range(10)
)


# ---- the recipes -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Planted:
    """One fault of a loop fleet, at a calibration point.

    `handle` is the private draw handle (`f01`, ...) the fault is built under; the world
    gets an opaque identity for it (`compose.opaque_fault_ids`). `label` names the
    calibration point (`compose.make_fault`); a release names what it carries as
    `kind|label` (several joined by ` + `). `part` is its role in the story, `note` says
    what it is in words, `shows_in` where an operator could see it. All truth side.
    """

    handle: str
    kind: str
    label: str
    scope: str
    onset: int
    part: str
    note: str
    shows_in: str
    version: str | None = None
    nominal_version: str | None = None


@dataclass(frozen=True)
class LoopFleetRecipe:
    """A designed loop fleet: identity, shape, windows and the stated schedule."""

    name: str
    world_id: str
    public_fleet_id: str
    seed: int
    planted: tuple[Planted, ...]
    confounders: tuple[tuple[str, str], ...] = ()  # (effective, inert) handles
    decoys: tuple[str, ...] = ()
    reveals: tuple[tuple[str, str], ...] = ()  # (first repair, the layer it reveals) handles
    lasting: tuple[str, ...] = ()  # handles no action removes
    site_suites: tuple[tuple[str, str], ...] = ()
    sites: int = 3
    robots_per_site: int = 4
    episodes_per_robot_day: int = 12
    windows: Windows = field(
        default_factory=lambda: Windows(
            observed=DayRange(first=0, last=5),
            experiment=DayRange(first=6, last=7),
            scoring=DayRange(first=8, last=11),
        )
    )
    extended_conditions: bool = True
    note: str = ""

    @property
    def days(self) -> int:
        """The stored horizon: through round 2's operation (the scoring window)."""
        return self.windows.scoring.last + 1

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "world_id": self.world_id,
            "public_fleet_id": self.public_fleet_id,
            "seed": self.seed,
            "sites": self.sites,
            "robots_per_site": self.robots_per_site,
            "days": self.days,
            "episodes_per_robot_day": self.episodes_per_robot_day,
            "windows": self.windows.model_dump(mode="json"),
            "site_suites": [list(p) for p in self.site_suites],
            "extended_conditions": self.extended_conditions,
            "planted": [p.__dict__ for p in self.planted],
            "confounders": [list(p) for p in self.confounders],
            "decoys": list(self.decoys),
            "reveals": [list(p) for p in self.reveals],
            "lasting": list(self.lasting),
            "note": self.note,
        }


def _lead() -> LoopFleetRecipe:
    """The lead fleet (hidden w70, public w01): the remote link at s1."""
    remote_note = (
        "lasting: every command reaches the robot 17 control steps late over the site's "
        "remote link (no action removes it; adapt with latency compensation)"
    )
    planted = [
        *(
            Planted(
                handle=f"f{n:02d}",
                kind="remote_link_latency",
                label="17 steps",
                scope=robot,
                onset=0,
                part="lasting",
                note=remote_note,
                shows_in="site description (remote link, no size); command-tracking lag in "
                "the motion records",
            )
            for n, robot in enumerate(("r01", "r02", "r03", "r04"), start=1)
        ),
        Planted(
            handle="f05",
            kind="software_release",
            label="camera_degradation|4 px blur",
            scope="s2",
            onset=1,
            version="4.3.0",
            nominal_version="4.2.1",
            part="logged cause (coincidence)",
            note="release 4.3.0 at s2 blurs the scene camera image by 4 px",
            shows_in="change log (software.version at s2, day 1)",
        ),
        Planted(
            handle="f06",
            kind="calibration_update",
            label="scene calibration rewritten",
            scope="s2",
            onset=1,
            part="inert (coincidence)",
            note="the scene cameras' calibration record at s2 rewritten the same day: no mechanism",
            shows_in="change log (camera.scene.calibration at s2, day 1)",
        ),
        Planted(
            handle="f07",
            kind="software_release",
            label="camera_degradation|2 px blur",
            scope="s3",
            onset=2,
            version="4.3.1",
            nominal_version="4.2.1",
            part="decoy",
            note="release 4.3.1 at s3 on the day s3 drops; its 2 px image smoothing is "
            "measured flat on both suites",
            shows_in="change log (software.version at s3, day 2)",
        ),
        Planted(
            handle="f08",
            kind="furniture_shift",
            label="2 cm sideways",
            scope="s3",
            onset=2,
            part="unlogged physical cause",
            note="the tables at s3 moved 2 cm sideways (worse on the object tasks); fixed by "
            "putting the placement back (change_condition placement at s3)",
            shows_in="no log; outcomes and images only",
        ),
        Planted(
            handle="f09",
            kind="gripper_wear",
            label="1 wear",
            scope="r03",
            onset=2,
            part="unlogged physical cause, masked",
            note="r03's gripper pads worn through (hurts the spatial tasks); hidden behind "
            "the remote link until s1 is adapted; fixed by replacing or servicing the gripper",
            shows_in="health (grip force) only",
        ),
        Planted(
            handle="f10",
            kind="action_latency",
            label="14 steps",
            scope="r06",
            onset=3,
            part="logged cause, masked",
            note="r06's command buffer set to 14 steps; under the s2 release its own share is "
            "hard to see; fixed by reverting control.action_delay_steps at r06",
            shows_in="change log (control.action_delay_steps at r06, day 3); command lag",
        ),
        Planted(
            handle="f11",
            kind="camera_pose_offset",
            label="3 deg scene pan",
            scope="r10",
            onset=3,
            part="unlogged physical cause, masked",
            note="r10's scene camera knocked 3 degrees; partly hidden behind the s3 table "
            "shift; fixed by recalibrating or replacing the scene camera",
            shows_in="health (scene camera calibration residual) only",
        ),
    ]
    return LoopFleetRecipe(
        name="LEAD",
        world_id="w70",
        public_fleet_id="w01",
        seed=9092241658733730548,  # recipe.new_world_seed() on 2026-10-13; development world
        planted=tuple(planted),
        confounders=(("f05", "f06"),),
        decoys=("f07",),
        reveals=(("f01", "f09"), ("f05", "f10"), ("f08", "f11")),
        lasting=("f01", "f02", "f03", "f04"),
        site_suites=(("s1", "libero_spatial"),),
        note=(
            "the lead fleet: s1's robots are driven over a remote link (17 steps, lasting, "
            "spatial tasks); s2 got release 4.3.0 with a scene camera calibration the same day; "
            "s3's tables moved the day s3 got release 4.3.1; under each site's first cause "
            "a second one waits (r03's pads, r06's command buffer, r10's camera)"
        ),
    )


def _later() -> LoopFleetRecipe:
    """The later fleet for the reuse comparison (hidden w71, public w02).

    The same kind of lasting condition at another site and of another size, on the other
    task suite (s3, 14 steps, object tasks), so the remembered site and setting are not the
    answer, among other faults: a logged release with a different mechanism, a heavier
    object batch, an arm, a different harmless change. Two lookalikes make a skill that is
    applied without its competing explanations pay (lane learn's proposal,
    docs/design/learned.md): s1 carries the same public note over a short link that costs
    nothing measurable on its spatial tasks, and later drops for a logged release whose
    image lag is a sensing delay, not a command delay; s2 is worse from day 0 with no lag
    at all, from heavier objects, beside a logged change that looks like it.
    """
    lasting_note = (
        "lasting: every command reaches the robot 14 control steps late over the site's "
        "remote link (no action removes it; adapt with latency compensation)"
    )
    short_note = (
        "the same remote link and site note, 8 steps: measured flat on the spatial tasks; "
        "no loss to adapt away"
    )
    planted = [
        *(
            Planted(
                handle=f"f{n:02d}",
                kind="remote_link_latency",
                label="8 steps",
                scope=robot,
                onset=0,
                part="lookalike (lasting, harmless)",
                note=short_note,
                shows_in="site description (remote link, no size); a small command lag",
            )
            for n, robot in enumerate(("r01", "r02", "r03", "r04"), start=1)
        ),
        *(
            Planted(
                handle=f"f{n:02d}",
                kind="remote_link_latency",
                label="14 steps",
                scope=robot,
                onset=0,
                part="lasting",
                note=lasting_note,
                shows_in="site description (remote link, no size); command-tracking lag in "
                "the motion records",
            )
            for n, robot in enumerate(("r09", "r10", "r11", "r12"), start=5)
        ),
        Planted(
            handle="f09",
            kind="software_release",
            label="camera_latency|24 steps scene lag",
            scope="s1",
            onset=2,
            version="4.4.0",
            nominal_version="4.2.1",
            part="logged cause",
            note="release 4.4.0 at s1 buffers the scene camera 24 steps (a sensing delay; "
            "compensating commands does not help); fixed by reverting software.version at s1",
            shows_in="change log (software.version at s1, day 2)",
        ),
        Planted(
            handle="f10",
            kind="object_batch_variant",
            label="6 mass doublings",
            scope="s2",
            onset=0,
            part="unlogged physical cause (lookalike)",
            note="s2's objects come from a heavier batch from day 0: worse from the first day "
            "across robots, with no command lag; fixed by change_condition object_batch at s2",
            shows_in="no log; outcomes only",
        ),
        Planted(
            handle="f11",
            kind="policy_knob",
            label="5 sampling steps",
            scope="s2",
            onset=0,
            part="decoy",
            note="s2's policy sampling steps 10 -> 5 on day 0, the day s2 starts worse; "
            "measured flat on the spatial tasks",
            shows_in="change log (policy.sampling_steps at s2, day 0)",
        ),
        Planted(
            handle="f12",
            kind="arm_torque_loss",
            label="0.3 torque lost",
            scope="r06",
            onset=2,
            part="unlogged physical cause",
            note="r06's arm drives lost 30% of their torque; no health reading moves; fixed by "
            "servicing or replacing the arm",
            shows_in="no log, no health reading; outcomes and motion only",
        ),
    ]
    return LoopFleetRecipe(
        name="LATER",
        world_id="w71",
        public_fleet_id="w02",
        seed=6862509924276675127,  # recipe.new_world_seed() on 2026-10-13; development world
        planted=tuple(planted),
        # The short link is flat on s1's spatial tasks: a decoy in the patterns' sense, and
        # still lasting (no action removes it).
        decoys=("f01", "f02", "f03", "f04", "f11"),
        lasting=("f05", "f06", "f07", "f08"),
        site_suites=(
            ("s1", "libero_spatial"),
            ("s2", "libero_spatial"),
            ("s3", "libero_object"),
        ),
        note=(
            "the later fleet: s3's robots are driven over a remote link (14 steps, lasting, "
            "object tasks); s1 has the same note over a harmless 8-step link and drops on day "
            "2 for release 4.4.0 (scene camera lag); s2 is worse from day 0 for heavier "
            "objects beside a harmless sampling-steps change; r06's arm is weak from day 2"
        ),
    )


RECIPES: dict[str, LoopFleetRecipe] = {"LEAD": _lead(), "LATER": _later()}


# ---- M3: the demonstration's recipes (docs/design/m3-recipe.md, section 2) -------------------
#
# The three demonstration fleets are the development recipes above, used as templates:
# every part keeps its kind, calibration point, kind of scope, onset, release versions and
# its role in the patterns; only where it lands moves, drawn from the new fleet's seed.
# Nothing here draws a seed or composes a world: the builders take the seed, and the M3
# recipes are not in `RECIPES`, so the command line cannot compose one.

SITES: tuple[str, ...] = tuple(base.site_ids(3))
ROBOTS_PER_SITE = 4
LATER_CONDITION_RULE = "m3-later-condition-site:"


@dataclass(frozen=True)
class Placement:
    """Where a template's parts land: each template site's site, and each template robot's
    robot (a robot stays at the site its template site moved to). Both are bijections."""

    sites: dict[str, str]
    robots: dict[str, str]

    def scope(self, scope: str) -> str:
        if scope == "fleet":
            return scope
        if scope in self.sites:
            return self.sites[scope]
        return self.robots[scope]


def _robots_of(site: str) -> list[str]:
    return base.robot_ids(ROBOTS_PER_SITE, SITES.index(site))


def _placement(seed: int, sites: dict[str, str]) -> Placement:
    """The robots' part of a placement: within each template site, its four robots in an
    order drawn from the seed, onto the robots of the site it moved to."""
    robots: dict[str, str] = {}
    for template, site in sites.items():
        order = base.rng(seed, "m3-placement", "robots", template).permutation(ROBOTS_PER_SITE)
        targets = _robots_of(site)
        for slot, robot in enumerate(_robots_of(template)):
            robots[robot] = targets[int(order[slot])]
    return Placement(sites=dict(sites), robots=robots)


def drawn_placement(seed: int) -> Placement:
    """The lead's and the variant's rule: the three sites in an order drawn from the seed,
    and within each site its four robots likewise (`base.rng`, so the same seed gives the
    same placement in any process)."""
    order = base.rng(seed, "m3-placement", "sites").permutation(len(SITES))
    return _placement(seed, {t: SITES[int(order[i])] for i, t in enumerate(SITES)})


def later_condition_site(lead_condition_site: str, later_seed: int) -> str:
    """The later fleet's condition site: one of the two sites other than the lead's, in id
    order, by the first byte of SHA-256 over the rule's tag and the later seed, mod 2. It is
    never the lead's site (the reuse comparison's "another site")."""
    if lead_condition_site not in SITES:
        raise ValueError(f"no site {lead_condition_site!r}")
    others = [s for s in SITES if s != lead_condition_site]
    digest = hashlib.sha256(f"{LATER_CONDITION_RULE}{later_seed}".encode()).digest()
    return others[digest[0] % 2]


def later_placement(
    later_seed: int, lead_condition_site: str, template: LoopFleetRecipe | None = None
) -> Placement:
    """The later fleet's rule: its condition's template site goes to
    `later_condition_site`; its other two template sites take the remaining two sites in an
    order drawn from the seed; robots within each site as for the lead. `template` is the
    later fleet's template (default w71's, `RECIPES["LATER"]`)."""
    template = template if template is not None else RECIPES["LATER"]
    condition = condition_site(template)
    site = later_condition_site(lead_condition_site, later_seed)
    others = [s for s in SITES if s != condition]
    remaining = [s for s in SITES if s != site]
    order = base.rng(later_seed, "m3-placement", "later-sites").permutation(len(remaining))
    sites = {condition: site, **{t: remaining[int(order[i])] for i, t in enumerate(others)}}
    return _placement(later_seed, sites)


def condition_site(recipe: LoopFleetRecipe) -> str:
    """The site of a recipe's lasting condition (its `lasting` faults, all at one site)."""
    by_handle = {p.handle: p for p in recipe.planted}
    found = {
        s
        for h in recipe.lasting
        for s in SITES
        if by_handle[h].scope == s or by_handle[h].scope in _robots_of(s)
    }
    if len(found) != 1:
        raise ValueError(f"{recipe.name}: the lasting condition is at {sorted(found)}")
    return found.pop()


def _relabel(text: str, placement: Placement) -> str:
    """Site and robot ids in a part's prose, moved with the part (all at once)."""
    import re

    return re.sub(
        r"\b(s[0-9]|r[0-9]{2})\b",
        lambda m: (
            placement.scope(m.group(1))
            if m.group(1) in placement.sites or m.group(1) in placement.robots
            else m.group(1)
        ),
        text,
    )


def masked_layers(recipe: LoopFleetRecipe) -> tuple[str, ...]:
    """The handles of a recipe's masked second layers: what each first repair reveals."""
    return tuple(second for _, second in recipe.reveals)


def place(
    template: LoopFleetRecipe,
    placement: Placement,
    *,
    name: str,
    world_id: str,
    public_fleet_id: str,
    seed: int,
    drop: Sequence[str] = (),
    summary: str | None = None,
) -> LoopFleetRecipe:
    """`template` with every part moved by `placement` and the handles in `drop` left out
    (from the parts and from every pattern); nothing else changes. `summary`, in the
    template's site and robot ids, replaces the template's note where parts are dropped,
    so that the summary names only the parts that remain."""
    if (template.sites, template.robots_per_site) != (len(SITES), ROBOTS_PER_SITE):
        raise ValueError(f"{template.name}: M3 places three sites of four robots")
    dropped = set(drop)
    unknown = dropped - {p.handle for p in template.planted}
    if unknown:
        raise KeyError(f"{template.name} has no parts {sorted(unknown)}")
    planted = tuple(
        replace(
            p,
            scope=placement.scope(p.scope),
            note=_relabel(p.note, placement),
            shows_in=_relabel(p.shows_in, placement),
        )
        for p in template.planted
        if p.handle not in dropped
    )
    return replace(
        template,
        name=name,
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
        planted=planted,
        confounders=tuple(c for c in template.confounders if not dropped & set(c)),
        decoys=tuple(h for h in template.decoys if h not in dropped),
        reveals=tuple(r for r in template.reveals if not dropped & set(r)),
        lasting=tuple(h for h in template.lasting if h not in dropped),
        site_suites=tuple((placement.sites[s], suite) for s, suite in template.site_suites),
        note=f"{name}: {template.name}'s parts"
        f"{' except ' + ', '.join(sorted(dropped)) if dropped else ''}, placed by the seed "
        f"(sites {placement.sites}); "
        f"{_relabel(template.note if summary is None else summary, placement)}",
    )


def m3_lead(seed: int, world_id: str, public_fleet_id: str) -> LoopFleetRecipe:
    """M3-LEAD: w70's parts, unchanged, placed by `drawn_placement(seed)`."""
    return place(
        RECIPES["LEAD"],
        drawn_placement(seed),
        name="M3-LEAD",
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
    )


# The variant's own summary, in the lead template's ids: its parts, and nothing it drops.
VARIANT_SUMMARY = (
    "the modest variant: s1's robots are driven over a remote link (17 steps, lasting, "
    "spatial tasks); s2 got release 4.3.0 with a scene camera calibration the same day; "
    "s3's tables moved the day s3 got release 4.3.1; no masked second layer"
)


def m3_variant(seed: int, world_id: str, public_fleet_id: str) -> LoopFleetRecipe:
    """M3-VARIANT: the lead's parts without its masked second layers, placed likewise,
    with a summary of its own."""
    lead = RECIPES["LEAD"]
    return place(
        lead,
        drawn_placement(seed),
        name="M3-VARIANT",
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
        drop=masked_layers(lead),
        summary=VARIANT_SUMMARY,
    )


@dataclass(frozen=True)
class PendingOption:
    """A choice the accepted recipe leaves open until M2's and w71's lessons: listed, never
    applied. The recipe holds w71's part until the option is decided and implemented."""

    name: str
    handle: str  # the template part it would change
    default: str  # what the recipe holds meanwhile: w71's part
    candidate: str  # what it would become, at an existing calibration point
    waits_on: str
    status: str = "pending"


PENDING_LATER_OPTIONS: dict[str, PendingOption] = {
    option.name: option
    for option in (
        PendingOption(
            name="release-blur",
            handle="f09",
            default="release 4.4.0 carrying camera_latency|24 steps scene lag (w71)",
            candidate="the same release carrying camera_degradation|4 px blur",
            waits_on="M2's and w71's lessons (docs/design/m3-recipe.md, sections 2 and 7)",
        ),
        PendingOption(
            name="arm-0.4",
            handle="f12",
            default="arm_torque_loss|0.3 torque lost (w71)",
            candidate="arm_torque_loss|0.4 torque lost",
            waits_on="M2's and w71's lessons (docs/design/m3-recipe.md, sections 2 and 7)",
        ),
    )
}


def m3_later(
    seed: int,
    world_id: str,
    public_fleet_id: str,
    lead_condition_site: str,
    options: Sequence[str] = (),
) -> LoopFleetRecipe:
    """M3-LATER: w71's parts placed by `later_placement` (its condition never at the
    lead's site). The two options in `PENDING_LATER_OPTIONS` are pending: asking for one
    raises, and the recipe holds w71's parts."""
    for name in options:
        if name not in PENDING_LATER_OPTIONS:
            raise KeyError(f"no option {name!r}; the options are {sorted(PENDING_LATER_OPTIONS)}")
        raise NotImplementedError(
            f"option {name!r} is pending ({PENDING_LATER_OPTIONS[name].waits_on}); "
            "the recipe holds w71's part"
        )
    return place(
        RECIPES["LATER"],
        later_placement(seed, lead_condition_site),
        name="M3-LATER",
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
    )


# ---- story B: the instrument's lookalike fleets (lane P) ----------------------------------------
#
# Lane P's spike (spikes/instrument/README.md) measured each candidate's step-probe signature
# and its ordinary-task symptom. The remote link is a fixed dead time under the probe. The
# overloaded controller shows the same symptom in the records (a like success drop and a
# command-tracking lag about the link's) but a dead time that grows within the probe; a
# latency compensation set from its lag does nothing, a smaller one helps part of the way,
# and servicing the controller removes it. The image release and the calibration record
# leave the probe as on a healthy robot. Like LEAD and LATER these are development templates:
# the demonstration fleets are placed from them by a seed (`storyb_lead`, `storyb_later`).
# Every site runs the spatial tasks, where every part is measured.

_LINK_NOTE = (
    "lasting: every command reaches the robot 17 control steps late over the site's remote "
    "link (no action removes it; adapt with latency compensation)"
)
_STORY_SUITES = (
    ("s1", "libero_spatial"),
    ("s2", "libero_spatial"),
    ("s3", "libero_spatial"),
)


def _link(robots: Sequence[str], first: int) -> list[Planted]:
    return [
        Planted(
            handle=f"f{n:02d}",
            kind="remote_link_latency",
            label="17 steps",
            scope=robot,
            onset=0,
            part="lasting",
            note=_LINK_NOTE,
            shows_in="site description (remote link, no size); command-tracking lag in the "
            "motion records; a fixed dead time under a step probe",
        )
        for n, robot in enumerate(robots, start=first)
    ]


def _storyb() -> LoopFleetRecipe:
    """Story B's fleet (hidden w80, public w08): the remote link at s1; r06's controller,
    overloaded, at s2; at s3 an image release logged on the day of a camera calibration
    record."""
    planted = [
        *_link(("r01", "r02", "r03", "r04"), 1),
        Planted(
            handle="f05",
            kind="controller_overload",
            label="0.2 cycles missed",
            scope="r06",
            onset=0,
            part="lookalike (fixable)",
            note="r06's controller misses a fifth of its cycles and its command buffer never "
            "catches up: in the records the same kind of success drop and command lag as the "
            "link, but a delay that grows within each attempt; fixed by servicing or "
            "replacing the controller (a partial latency compensation helps part of the way)",
            shows_in="no log, no health reading; command-tracking lag in the motion records; "
            "a dead time that grows under a step probe",
        ),
        Planted(
            handle="f06",
            kind="image_pipeline_release",
            label="20% cropped",
            scope="s3",
            onset=2,
            version="4.3.0",
            nominal_version="4.2.1",
            part="logged cause (coincidence)",
            note="release 4.3.0 at s3 crops a fifth of each camera image before the policy "
            "sees it; fixed by reverting software.version at s3",
            shows_in="change log (software.version at s3, day 2); motion as on a healthy "
            "robot under a step probe",
        ),
        Planted(
            handle="f07",
            kind="calibration_update",
            label="scene calibration rewritten",
            scope="s3",
            onset=2,
            part="inert (coincidence)",
            note="the scene cameras' calibration record at s3 rewritten the same day: no mechanism",
            shows_in="change log (camera.scene.calibration at s3, day 2)",
        ),
    ]
    return LoopFleetRecipe(
        name="STORYB",
        world_id="w80",
        public_fleet_id="w08",
        seed=4740889858747787853,  # recipe.new_world_seed() on 2026-10-19; development world
        planted=tuple(planted),
        confounders=(("f06", "f07"),),
        lasting=("f01", "f02", "f03", "f04"),
        site_suites=_STORY_SUITES,
        note=(
            "story B's fleet: s1's robots are driven over a remote link (17 steps, lasting); "
            "r06 at s2 has an overloaded controller with the same symptom in the records; s3 "
            "got release 4.3.0 (images cropped) with a scene camera calibration the same day"
        ),
    )


def _storyb_later() -> LoopFleetRecipe:
    """Story B's later fleet for the second use (hidden w81, public w09): the remote link at
    another site (s3), and r06 at s2 whose scene camera's images arrive late from day 2:
    motion prompt under the probe while its tasks fail."""
    planted = [
        *_link(("r09", "r10", "r11", "r12"), 1),
        Planted(
            handle="f05",
            kind="camera_latency",
            label="24 steps scene lag",
            scope="r06",
            onset=2,
            part="lookalike (prompt motion)",
            note="r06's scene camera hands over frames 24 control steps old from day 2: the "
            "robot acts on a stale picture and fails, while its commands arrive on time; "
            "fixed by servicing or replacing the scene camera, not by compensating commands",
            shows_in="no log, no health reading; outcomes and images only; motion as on a "
            "healthy robot under a step probe",
        ),
    ]
    return LoopFleetRecipe(
        name="STORYB-LATER",
        world_id="w81",
        public_fleet_id="w09",
        seed=5170722821263939881,  # recipe.new_world_seed() on 2026-10-19; development world
        planted=tuple(planted),
        lasting=("f01", "f02", "f03", "f04"),
        site_suites=_STORY_SUITES,
        note=(
            "story B's later fleet: s3's robots are driven over a remote link (17 steps, "
            "lasting); r06 at s2 acts on scene camera images 24 steps late from day 2"
        ),
    )


# Story B's recipes, apart from `RECIPES` (whose tests pin the M3 fleets' design); the
# command line takes either.
STORY_RECIPES: dict[str, LoopFleetRecipe] = {"STORYB": _storyb(), "STORYB-LATER": _storyb_later()}


def storyb_lead(seed: int, world_id: str, public_fleet_id: str) -> LoopFleetRecipe:
    """A demonstration fleet from STORYB: its parts, unchanged, placed by
    `drawn_placement(seed)` as M3-LEAD's are."""
    return place(
        STORY_RECIPES["STORYB"],
        drawn_placement(seed),
        name="STORYB-DRAWN",
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
    )


def storyb_later(
    seed: int, world_id: str, public_fleet_id: str, lead_condition_site: str
) -> LoopFleetRecipe:
    """The later demonstration fleet from STORYB-LATER, placed by `later_placement`: its
    remote link never at the site the lead fleet's link went to."""
    template = STORY_RECIPES["STORYB-LATER"]
    return place(
        template,
        later_placement(seed, lead_condition_site, template),
        name="STORYB-LATER-DRAWN",
        world_id=world_id,
        public_fleet_id=public_fleet_id,
        seed=seed,
    )


# ---- composition -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Composed:
    """A loop fleet's spec under its draw handles, with each fault's part."""

    recipe: LoopFleetRecipe
    spec: WorldSpec
    parts: dict[str, str]  # handle -> part


def compose(recipe: LoopFleetRecipe) -> Composed:
    """The recipe's `WorldSpec`, faults under their draw handles (pure)."""
    from fleetlab.worlds.compose import make_fault
    from fleetlab.worlds.generate import policy_spec_nominal

    faults: list[FaultSpec] = []
    for p in recipe.planted:
        faults.append(
            make_fault(
                p.handle,
                p.kind,
                p.label,
                p.scope,
                p.onset,
                nominal_version=p.nominal_version,
                version=p.version,
            )
        )
    patterns = Patterns(
        confounders=[tuple(pair) for pair in recipe.confounders],  # type: ignore[misc]
        decoys=list(recipe.decoys),
    )
    spec = WorldSpec(
        world_id=recipe.world_id,
        seed=recipe.seed,
        size="small",
        sites=recipe.sites,
        robots_per_site=recipe.robots_per_site,
        days=recipe.days,
        windows=recipe.windows,
        fault_schedule=faults,
        patterns=patterns,
        policy=policy_spec_nominal(),
        tasks=list(TASKS),
        episodes_per_robot_day=recipe.episodes_per_robot_day,
        # Every loop fleet has the settings added later (the compensation, lane adapt):
        # the fleet default is recorded, described and offered (docs/design/loop.md, 7).
        later_settings=sorted(LATER_CONFIG_KEYS),
    )
    return Composed(recipe, spec, {p.handle: p.part for p in recipe.planted})


# ---- the world ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteSuites:
    """What `world.world_truth` reads of a recipe: the site suite restrictions."""

    site_suites: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LoopFleetWorld(World):
    """A built loop fleet: the version-0 `World` the runner, engine and records read, with
    the recipe, the handle map and the design forecast (truth side)."""

    loop_recipe: LoopFleetRecipe
    handles: dict[str, str]  # handle -> opaque fault id
    emphasis: tuple[base.Emphasis, ...]
    design: dict[str, Any]

    @property
    def recipe(self) -> SiteSuites:
        return SiteSuites(self.loop_recipe.site_suites)

    def fault_of(self, handle: str) -> FaultSpec:
        fid = self.handles[handle]
        return next(f for f in self.spec.fault_schedule if f.fault_id == fid)


def build(
    recipe: LoopFleetRecipe, software: SoftwareVersion, texts: dict[Task, str] | None = None
) -> LoopFleetWorld:
    """The world of a loop fleet recipe for jobs made for `software` (pure, given texts)."""
    import fleetlab.faults as faults
    from fleetlab.worlds import health as health_model
    from fleetlab.worlds import world as v1
    from fleetlab.worlds.compose import opaque_fault_ids, rename_fault_ids
    from fleetlab.worlds.generate import task_texts

    composed = compose(recipe)
    spec = composed.spec
    for fault in spec.fault_schedule:
        faults.REGISTRY.check(fault)
    robots = base.all_robots(spec.sites, spec.robots_per_site)
    log = v1.changes(spec)
    upkeep = v1.maintenance(spec)
    readings = health_model.rows(
        spec.seed, robots, spec.days, spec.episodes_per_robot_day, spec.fault_schedule, log, upkeep
    )

    def fleet_on(day: int, traits: bool) -> Fleet:
        return v1._fleet(
            spec,
            log,
            upkeep,
            readings,
            day,
            with_traits=traits,
            extended_conditions=recipe.extended_conditions,
            public_fleet_id=recipe.public_fleet_id,
        )

    fleet = fleet_on(spec.days - 1, True)
    observed = fleet_on(spec.windows.observed.last, False)
    daily = [fleet_on(d, False) for d in range(spec.days)]
    slot_plan = base.slots(
        spec.seed,
        spec.sites,
        spec.robots_per_site,
        spec.days,
        spec.episodes_per_robot_day,
        spec.tasks,
        recipe.site_suites,
    )
    all_jobs = v1.jobs(spec, daily, software, slot_plan)
    mapping = opaque_fault_ids(spec.seed, [f.fault_id for f in spec.fault_schedule])
    final_spec = rename_fault_ids(spec, mapping)
    final_jobs = rename_fault_ids(all_jobs, mapping)
    v1.assert_visible_ids_absent(
        _visible_probe(fleet, readings, texts or {}, spec.tasks),  # type: ignore[arg-type]
        {*mapping, *mapping.values()},
    )
    texts = texts if texts is not None else task_texts(spec.tasks)
    design = forecast_design(recipe, final_spec, final_jobs, mapping)
    table = _fault_table(recipe, final_spec, final_jobs, mapping, design)
    return LoopFleetWorld(
        spec=final_spec,
        software=software,
        fleet=fleet,
        observed_fleet=observed,
        jobs=final_jobs,
        health=readings,
        task_texts=texts,
        fault_table=table,
        loop_recipe=recipe,
        handles=mapping,
        emphasis=tuple(base.emphasis(spec.seed, spec.sites, spec.tasks, recipe.site_suites)),
        design=design,
    )


@dataclass(frozen=True)
class _Probe:
    """What `world.assert_visible_ids_absent` reads: the fleet, health and task classes."""

    fleet: Fleet
    health: list[dict[str, Any]]
    task_classes: list[str]


def _visible_probe(
    fleet: Fleet, readings: list[dict[str, Any]], texts: dict[Task, str], tasks: Sequence[Task]
) -> _Probe:
    classes = [texts.get(t, f"{t.suite} {t.task_index}") for t in tasks]
    return _Probe(fleet, readings, classes)


# ---- the design forecast -------------------------------------------------------------------------


def _members(faults: Sequence[FaultSpec]) -> tuple[tuple[str, str], ...] | None:
    """The measured-cell members of an episode's faults: releases expanded into what they
    carry, the inert calibration record dropped, a type read off another type's curve
    (`curve_kind`) renamed. None when a member is not at a calibrated point."""
    from fleetlab.faults import get
    from fleetlab.faults.software_release import ReleaseParams, carried_spec
    from fleetlab.worlds.curves import calibrated_label

    expanded: list[FaultSpec] = []
    for f in faults:
        if f.kind == "software_release":
            expanded.extend(
                carried_spec(c, f.target.scope, f.schedule, f.fault_id)
                for c in ReleaseParams.model_validate(f.params).carries
            )
        elif f.kind != "calibration_update":
            expanded.append(f)
    out = []
    for f in expanded:
        # The point's label is the type's own; a type with another's mechanism bit for bit
        # (`curve_kind`) labels its points the same way and is read off that type's cells.
        label = calibrated_label(f)
        if label is None:
            return None
        out.append((str(getattr(get(f.kind), "curve_kind", f.kind)), label))
    return tuple(sorted(out))


@dataclass
class _Cells:
    """Measured success by (suite, task, members) and by (suite, members) over tasks."""

    by_task: dict[tuple[str, int, tuple[tuple[str, str], ...]], tuple[int, int]]
    by_suite: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[int, int]]

    @classmethod
    def load(cls) -> _Cells:
        from fleetlab.worlds import curves

        by_task: dict[tuple[str, int, tuple[tuple[str, str], ...]], tuple[int, int]] = {}
        by_suite: dict[tuple[str, tuple[tuple[str, str], ...]], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        for (suite, task, members), cell in curves.measured_cells().items():
            by_task[(suite, task, members)] = (cell.successes, cell.n)
            agg = by_suite[(suite, members)]
            agg[0] += cell.successes
            agg[1] += cell.n
        return cls(by_task, {k: (v[0], v[1]) for k, v in by_suite.items()})

    def rate(self, suite: str, members: tuple[tuple[str, str], ...]) -> float | None:
        found = self.by_suite.get((suite, members))
        return None if found is None or found[1] < 20 else found[0] / found[1]

    def probability(
        self, suite: str, task: int, members: tuple[tuple[str, str], ...]
    ) -> tuple[float, str]:
        """Success of one attempt: the exact measured joint (suite level, at least 20
        attempts), else the healthy task rate times each member's suite-level ratio."""
        healthy_task = self.by_task.get((suite, task, ()))
        healthy_suite = self.rate(suite, ())
        assert healthy_suite is not None
        base_rate = (
            healthy_task[0] / healthy_task[1]
            if healthy_task and healthy_task[1] >= 20
            else healthy_suite
        )
        if not members:
            return base_rate, "healthy"
        joint = self.rate(suite, members)
        if joint is not None:
            return joint * base_rate / healthy_suite, "joint"
        p = base_rate
        for member in members:
            single = self.rate(suite, (member,))
            if single is None:
                return float("nan"), "unmeasured"
            p *= min(1.0, single / healthy_suite)
        return p, "product"


def forecast_design(
    recipe: LoopFleetRecipe,
    spec: WorldSpec,
    jobs: Sequence[Job],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Design guidance: expected success by window, site, robot and day, and each fault's
    loss when it alone is removed (points of fleet success over each window)."""
    cells = _Cells.load()
    windows = {name: getattr(spec.windows, name) for name in WINDOW_NAMES}
    sources: dict[str, int] = defaultdict(int)

    def p_of(identity: Any, without: str | None = None) -> float:
        faults = [f for f in identity.faults if f.fault_id != without]
        members = _members(faults)
        if members is None:
            sources["uncalibrated"] += 1
            return float("nan")
        p, how = cells.probability(identity.suite, identity.task_index, members)
        if without is None:
            sources[how] += 1
        return p

    rows = []
    for job in jobs:
        i = job.identity
        rows.append((i, p_of(i)))
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, p in rows:
        window = next((w for w, r in windows.items() if i.day in r), None)
        if window is not None:
            by["window"][window].append(p)
        by["site_window"][f"{i.site_id} {window}"].append(p)
        by["robot_window"][f"{i.robot_id} {window}"].append(p)
        by["site_day"][f"{i.site_id} d{i.day:02d}"].append(p)
        by["robot_day"][f"{i.robot_id} d{i.day:02d}"].append(p)
        by["suite_site"][f"{i.site_id} {i.suite}"].append(p)

    def mean(values: list[float]) -> float:
        return round(100.0 * sum(values) / len(values), 1) if values else float("nan")

    summary = {group: {k: mean(v) for k, v in sorted(d.items())} for group, d in by.items()}
    counts = {k: len(v) for k, v in sorted(by["suite_site"].items())}
    inverse = {v: k for k, v in mapping.items()}
    losses: dict[str, dict[str, float]] = {}
    for fault in spec.fault_schedule:
        loss = {}
        for window, days in windows.items():
            here = [(i, p) for i, p in rows if i.day in days]
            if not here:
                continue
            removed = sum(
                p_of(i, fault.fault_id) - p
                if any(f.fault_id == fault.fault_id for f in i.faults)
                else 0.0
                for i, p in here
            )
            loss[window] = round(100.0 * removed / len(here), 2)
        losses[inverse[fault.fault_id]] = loss
    return {
        "rule": "measured cells: the exact joint where measured on the suite (at least 20 "
        "attempts), else the healthy task rate times each member's suite-level ratio; a "
        "release counts as what it carries, the calibration record as nothing; design "
        "guidance only, the run decides",
        "sources": dict(sources),
        "success_percent": summary,
        "attempts_by_site_suite": counts,
        "loss_points_if_removed_alone": losses,
    }


# ---- the truth table -------------------------------------------------------------------------


def _fault_table(
    recipe: LoopFleetRecipe,
    spec: WorldSpec,
    jobs: Sequence[Job],
    mapping: dict[str, str],
    design: dict[str, Any],
) -> FaultTable:
    """The hidden fault table in the scorer's schema, from the design forecast."""
    import fleetlab.faults as faults
    from fleetlab.contracts.faults import Direction

    truth = faults.REGISTRY.truth_table()
    windows = {name: getattr(spec.windows, name) for name in WINDOW_NAMES}
    per_window = {w: sum(1 for j in jobs if j.identity.day in r) for w, r in windows.items()}
    inverse = {v: k for k, v in mapping.items()}
    decoys = {mapping[h] for h in recipe.decoys}
    inert = {mapping[i] for _, i in recipe.confounders}
    rows = []
    for fault in spec.fault_schedule:
        fid = fault.fault_id
        handle = inverse[fid]
        role: Role = "decoy" if fid in decoys else "inert" if fid in inert else "effective"
        loss = design["loss_points_if_removed_alone"][handle]
        episodes = {
            w: sum(
                1
                for j in jobs
                if j.identity.day in r and any(f.fault_id == fid for f in j.identity.faults)
            )
            for w, r in windows.items()
        }
        qualified = [w for w in ("observed", "scoring") if loss.get(w, 0.0) >= 2.0]
        mapping_ = truth[fault.kind]
        directions = set(mapping_.directions) - {Direction.drifting}
        rows.append(
            FaultTruth(
                fault_id=fid,
                kind=fault.kind,
                fault_class=faults.get(fault.kind).fault_class,
                target=fault.target,
                role=role,
                onset_day=fault.schedule.onset_day,
                level=fault.level,
                directions=sorted(directions, key=lambda d: d.value),
                fixing_verbs=sorted(mapping_.fixing_verbs, key=lambda v: v.value),
                episodes=episodes,  # type: ignore[arg-type]
                expected_loss_points={w: float(loss.get(w, 0.0)) for w in WINDOW_NAMES},  # type: ignore[misc]
                identifiable=True,
                qualified_by=qualified,  # type: ignore[arg-type]
                detectable=role == "effective" and bool(qualified),
                small=role == "effective" and not qualified,
                calibration="loop fleet design forecast (measured cells)",
            )
        )
    return FaultTable(
        world_id=spec.world_id,
        windows=spec.windows,
        episodes_per_window=per_window,  # type: ignore[arg-type]
        rule="loop fleet: each fault removed alone from the design forecast",
        faults=rows,
    )


# ---- the files ---------------------------------------------------------------------------------


def _dump(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=False) + "\n"


def loop_layout(world: LoopFleetWorld) -> Any:
    """The loop's layout of this fleet (lane state's `layout_for_world` of its spec)."""
    from fleetlab.loop.layout import layout_for_world

    return layout_for_world(world.spec)


def generator_record(world: LoopFleetWorld) -> dict[str, Any]:
    """`generator.json`: the version-1 generator's name (so the loop derives health and the
    panel's slots the same way) and the recipe with its site suites.

    Only keys the generator's own records already carry: the leak audit reads every key of
    the world's files as a term to search the export for, and a new key such as a forecast
    table's would become one. The design, the forecast and the parts lie beside the world
    files (`write_beside`), which the audit does not read as world files."""
    recipe = world.loop_recipe
    roles = {f.fault_id: f.role for f in world.fault_table.faults}
    return {
        "generator": base.GENERATOR,
        "recipe": {
            "name": recipe.name,
            "world_id": recipe.world_id,
            "seed": recipe.seed,
            "public_fleet_id": recipe.public_fleet_id,
            "kind": "development",
            "size": "small",
            "sites": recipe.sites,
            "robots_per_site": recipe.robots_per_site,
            "days": recipe.days,
            "episodes_per_robot_day": recipe.episodes_per_robot_day,
            "windows": recipe.windows.model_dump(mode="json"),
            "tasks": [list(t) for t in world.spec.tasks],
            "site_suites": [list(p) for p in recipe.site_suites],
            "sealed_hash": world_hash(world),
            "note": f"{GENERATOR}: {recipe.note}",
        },
        "roles": roles,
        "emphasis": [e.as_json() for e in world.emphasis],
        "software": world.software.model_dump(mode="json"),
        "traits": {r.id: dict(r.traits) for r in world.fleet.robots()},
    }


def serialized_world(world: LoopFleetWorld) -> dict[str, bytes]:
    description = describe_for_agents(world.observed_fleet, True, world.task_classes)
    return {
        name: text.encode()
        for name, text in {
            "spec.json": world.spec.model_dump_json(indent=1) + "\n",
            "model.json": world.fleet.model_dump_json(indent=1) + "\n",
            "description.json": _dump(description),
            "manifest.jsonl": "".join(j.model_dump_json() + "\n" for j in world.jobs),
        }.items()
    }


def world_hash(world: LoopFleetWorld) -> str:
    from fleetlab.contracts.world import WORLD_FILES

    files = serialized_world(world)
    digest = hashlib.sha256()
    for name in WORLD_FILES:
        data = files[name]
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def write_world(world: LoopFleetWorld, directory: Path | str, base_runs: Sequence[str] = ()) -> str:
    """The world files and the truth-side files beside them; returns the world hash.
    The runner's `step_files` calls this on a scratch directory and compares hashes."""
    from fleetlab.engine.experiments import WorldTruth
    from fleetlab.worlds import world as v1

    root = Path(directory)
    if (root / "hash").is_file() and (root / "hash").read_text().strip() != world_hash(world):
        raise FileExistsError("refusing to replace a different sealed world")
    root.mkdir(parents=True, exist_ok=True)
    for name, data in serialized_world(world).items():
        (root / name).write_bytes(data)
    value = seal_world(root)
    texts = [
        {"suite": t.suite, "task_index": t.task_index, "text": world.task_texts[t]}
        for t in world.spec.tasks
    ]
    truth = WorldTruth.model_validate(v1.world_truth(world, base_runs))  # type: ignore[arg-type]
    files = {
        FAULT_TABLE_FILE: world.fault_table.model_dump_json(indent=1) + "\n",
        "health.jsonl": "".join(json.dumps(r) + "\n" for r in world.health),
        "task_texts.json": _dump(texts),
        "world_truth.json": truth.model_dump_json(indent=1) + "\n",
        "generator.json": _dump(generator_record(world)),
    }
    for name, text in files.items():
        target = root / name
        if not target.is_file() or target.read_text() != text:
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(text)
            tmp.replace(target)
    return value


def write_beside(world: LoopFleetWorld, root: Path) -> None:
    """Beside the world files, outside what the leak audit reads as world files: the loop
    layout and the as-composed truth table (development material)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "loop_layout.json").write_text(loop_layout(world).model_dump_json(indent=1) + "\n")
    (root / "design.json").write_text(_dump(truth_report(world)))


# ---- the truth table for people ------------------------------------------------------------------


def truth_report(world: LoopFleetWorld) -> dict[str, Any]:
    """The as-composed truth table: every fault with its part, where it shows, the fix, the
    measured single-fault reference per suite and its forecast loss."""
    import fleetlab.faults as faults

    cells = _Cells.load()
    table = world.fault_table.by_id()
    rows = []
    for p in world.loop_recipe.planted:
        fault = world.fault_of(p.handle)
        members = _members([fault])
        ref = {}
        if members:
            for suite in ("libero_spatial", "libero_object"):
                r = cells.rate(suite, members)
                h = cells.rate(suite, ())
                if r is not None and h is not None:
                    ref[suite] = f"{100 * r:.0f}% (healthy {100 * h:.0f}%)"
        truth = faults.REGISTRY.truth_table()[fault.kind]
        if not truth.fixing_verbs:
            fix = "none: no action removes it" + (" (adapt)" if p.part == "lasting" else "")
        else:
            verbs = "/".join(sorted(v.value for v in truth.fixing_verbs))
            thing = fault.target.thing.value if fault.target.thing else ""
            fix = f"{verbs} at {fault.target.scope} ({thing})"
        rows.append(
            {
                "handle": p.handle,
                "fault_id": fault.fault_id,
                "kind": p.kind,
                "point": p.label,
                "scope": p.scope,
                "onset_day": p.onset,
                "part": p.part,
                "role": table[fault.fault_id].role,
                "what": p.note,
                "shows_in": p.shows_in,
                "fix": fix,
                "measured_alone": ref,
                "forecast_loss_points": world.design["loss_points_if_removed_alone"][p.handle],
                **({"release": f"{p.nominal_version} -> {p.version}"} if p.version else {}),
            }
        )
    return {
        "fleet": world.loop_recipe.name,
        "world_id": world.spec.world_id,
        "public_fleet_id": world.loop_recipe.public_fleet_id,
        "note": world.loop_recipe.note,
        "layout": json.loads(loop_layout(world).model_dump_json()),
        "attempts": {
            "per_robot_day": world.spec.episodes_per_robot_day,
            "observed (round 1 operation)": len(world.jobs_in("observed")),
            "experiment window (as-run, not part of the loop)": len(world.jobs_in("experiment")),
            "scoring (round 2 operation)": len(world.jobs_in("scoring")),
        },
        "site_suites": {e.site_id: f"{e.favoured} {e.share:.2f}" for e in world.emphasis},
        "faults": rows,
        "confounders": [list(c) for c in world.loop_recipe.confounders],
        "decoys": list(world.loop_recipe.decoys),
        "reveals": [list(r) for r in world.loop_recipe.reveals],
        "forecast": world.design["success_percent"],
        "forecast_rule": world.design["rule"],
        "forecast_sources": world.design["sources"],
    }


# ---- what the run measured -------------------------------------------------------------------


def measured(run_dir: Path) -> dict[str, Any]:
    """Success of a finished run by site, robot and day (and the cross tables), from its
    episode log."""
    import pyarrow.parquet as pq

    rows = pq.read_table(run_dir / "log" / "episodes.parquet").to_pylist()
    tally: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        ok = int(r["outcome"] == "success")
        for group, key in (
            ("fleet", "all"),
            ("site", r["site_id"]),
            ("robot", r["robot_id"]),
            ("day", f"d{r['day']:02d}"),
            ("site_day", f"{r['site_id']} d{r['day']:02d}"),
            ("robot_day", f"{r['robot_id']} d{r['day']:02d}"),
        ):
            cell = tally[group][key]
            cell[0] += ok
            cell[1] += 1
    return {
        group: {
            k: {"successes": s, "attempts": n, "percent": round(100 * s / n, 1)}
            for k, (s, n) in sorted(d.items())
        }
        for group, d in tally.items()
    }


def suites_measured(run_dir: Path, world: LoopFleetWorld) -> dict[str, Any]:
    """Success by site and suite, the suite read from the manifest's identities."""
    import pyarrow.parquet as pq

    suite_of = {j.identity.episode_id: j.identity.suite for j in world.jobs}
    rows = pq.read_table(run_dir / "log" / "episodes.parquet").to_pylist()
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        key = f"{r['site_id']} {suite_of[r['episode_id']]}"
        tally[key][0] += int(r["outcome"] == "success")
        tally[key][1] += 1
    return {
        k: {"successes": s, "attempts": n, "percent": round(100 * s / n, 1)}
        for k, (s, n) in sorted(tally.items())
    }


def run_status(world: LoopFleetWorld, paths: Any) -> dict[str, Any]:
    """Where round 1's operation stands. The runner marks a world ready only with its later
    run, which a loop fleet does not run, and then writes `leaks_clean: false` into its
    record whatever the export check found; the export's own receipt is what counts, read
    here with the current checker (`matching_export_check`)."""
    from fleetlab.log.leaks import matching_export_check
    from fleetlab.worlds.runner import _chain_ok

    clean = (paths.export / "data_dictionary.md").is_file() and matching_export_check(
        paths.export, [paths.observed], fleet=world.fleet, receipt_dir=paths.root / "export-checks"
    )
    return {
        "world_id": world.spec.world_id,
        "observed_run": str(paths.observed),
        "observed_chain_ok": _chain_ok(paths.observed),
        "export": str(paths.export),
        "export_check_clean": clean,
    }


def export_round_one(world: LoopFleetWorld, paths: Any, out: Path) -> dict[str, Any]:
    """Round 1's export as the loop builds it (`fleetlab.loop.export.export_round` under the
    empty ledger, from the world's observed run), with its pre-export checks; raises
    `fleetlab.log.ExportRefused` on any problem. `out` must not exist yet."""
    from fleetlab.loop.export import export_round
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld

    loop_world = LoopWorld.load(paths.root)
    layout = loop_layout(world)
    scratch = LoopPaths(out.parent / f".{out.name}-loop")
    export_round(loop_world, layout, 1, Ledger.empty(loop_world.world_id), scratch, out)
    episodes = sum(1 for _ in (out / "log").glob("episodes*"))
    return {"export": str(out), "checks": "passed", "episode_tables": episodes}


# ---- command line ------------------------------------------------------------------------------


def _software(fake: bool) -> SoftwareVersion:
    if fake:
        return SoftwareVersion(
            lock_hash="0" * 16,
            libero_patch_set="1" * 16,
            mujoco_version="3.3.7",
            miopen_db_hash="2" * 16,
        )
    from fleetlab.worlds import runner

    return runner.engine_software()


def _paths(recipe: LoopFleetRecipe, root: Path | None) -> Any:
    from fleetlab.worlds.layout import WorldPaths

    return (
        WorldPaths.default(recipe.world_id) if root is None else WorldPaths(root, recipe.world_id)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.worlds.loop_fleets")
    sub = parser.add_subparsers(dest="command", required=True)
    named = {**RECIPES, **STORY_RECIPES}
    for name in ("check", "generate", "run", "export-r1", "report"):
        p = sub.add_parser(name)
        p.add_argument("name", choices=sorted(named))
        p.add_argument("--root", type=Path)
        if name == "check":
            p.add_argument("--fake-software", action="store_true")
        if name == "run":
            p.add_argument("--workers", type=int, default=10)
            p.add_argument(
                "--only",
                choices=["files", "observed", "observed-records", "observed-events", "export"],
            )
        if name == "export-r1":
            p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = named[args.name]
    if args.command == "check":
        world = build(recipe, _software(args.fake_software))
        print(json.dumps(truth_report(world), indent=1))
        return 0
    from fleetlab.worlds import runner

    paths = _paths(recipe, args.root)
    world = build(recipe, runner.engine_software())
    if args.command == "generate":
        with runner.runner_lock(paths):
            value = runner.step_files(world, paths, write_world)
        write_beside(world, paths.root)
        print(json.dumps({"hash": value, "root": str(paths.root)}, indent=1))
        return 0
    if args.command == "run":
        # Round 1's operation is the observed window as run; the world's later run (its
        # experiment and scoring windows as-run) is not part of the loop and is not run.
        engine = runner.EngineSettings(workers=args.workers)
        if args.only:
            runner.run_world(world, paths, engine, write=write_world, only=args.only)
        else:
            runner.run_world(world, paths, engine, write=write_world, through="observed-events")
            runner.run_world(world, paths, engine, write=write_world, only="export")
        write_beside(world, paths.root)
        print(json.dumps(run_status(world, paths), indent=1))
        return 0
    if args.command == "export-r1":
        print(json.dumps(export_round_one(world, paths, args.out), indent=1))
        return 0
    # report
    out = {
        "truth": truth_report(world),
        "measured": measured(paths.observed),
        "measured_by_site_suite": suites_measured(paths.observed, world),
        "runner": {
            k: {kk: vv for kk, vv in v.items() if kk != "attempts"}
            for k, v in paths.read_record().get("steps", {}).items()
        },
    }
    (paths.root / "report.json").write_text(_dump(out))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
