"""World recipes, private creation, and immutable generation/calibration pins.

`compose(recipe)` (`fleetlab.worlds.compose`) turns a recipe into the contracts'
`WorldSpec`, fault schedule and patterns included, and `generate` (`fleetlab.worlds.world`)
turns that into the world; both are pure functions, so the recipe and its seed are all
a world is. `DEVELOPMENT` holds the requested recipes; `INTEGRATION` pins the archived w44/w45
records. D2/D3 use independently confirmed mild interaction points and pinned task mixes.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fleetlab.contracts.identity import WORLD_SEED_LIMIT, Task
from fleetlab.contracts.world import DayRange, Windows

TASKS: tuple[Task, ...] = tuple(Task("libero_spatial", i) for i in range(10)) + tuple(
    Task("libero_object", i) for i in range(10)
)

SMALL_WINDOWS = Windows(
    observed=DayRange(first=0, last=5),
    experiment=DayRange(first=6, last=7),
    scoring=DayRange(first=8, last=9),
)

WorldKind = Literal["development", "test", "reserve", "nightly"]


def new_world_seed() -> int:
    """A signed-int64-compatible hidden seed with 62 independently random bits."""
    return (1 << 62) | secrets.randbits(62)


def private_path(path: Path) -> Path:
    """Refuse a truth-side path inside any checkout, including symlink destinations."""
    resolved = path.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if resolved.is_relative_to(checkout) or any(
        (parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
        for parent in (resolved, *resolved.parents)
    ):
        raise ValueError("private world recipes and outputs must be outside the repository")
    return resolved


def private_root(path: Path) -> Path:
    """Check before drawing a seed or writing; existing file symlinks count as outputs."""
    root = private_path(path)
    files = private_path(root / "files")
    if files.is_dir():
        for entry in files.iterdir():
            private_path(entry)
    return root


def external_recipe(path: Path, *, stored: Path | None = None) -> WorldRecipe:
    """Read a private test/reserve template outside the checkout; draw only on creation.

    The input must omit the seed and state every other field (`stated_fields`,
    `private_recipe`): a field left out would take the model's default, which for the
    settings added since the third generation is the old rule, not the template's. It
    names its calibration jobs, so that a later import cannot move its cells.
    Resumption uses the recipe saved in generator.json on the truth side and refuses
    any change to the external template.
    """
    private_path(path)
    if stored is not None:
        private_path(stored)
    raw = json.loads(path.read_text())
    if "seed" in raw:
        raise ValueError("an external recipe must omit seed; creation draws it privately")
    if raw.get("kind") not in ("test", "reserve"):
        raise ValueError("external recipes are for test or reserve worlds")
    missing = sorted(set(WorldRecipe.model_fields) - {"seed"} - set(raw))
    if missing:
        raise ValueError(f"an external recipe must state every field; missing: {missing}")
    if raw["estimator"] == "measured-v3" and raw["calibration_jobs"] is None:
        raise ValueError("an external recipe must name the calibration jobs it stands on")
    # Stated as `null` is as good as left out: no floor, no pinned digest. Where the code's
    # recipe for this identity states a value, the file states one too.
    try:
        code = private_recipe(
            raw["world_id"], raw["public_fleet_id"], name=raw["name"], kind=raw["kind"]
        )
    except TypeError as error:
        raise ValueError("an external recipe needs a world id, a fleet id and a name") from error
    empty = sorted(key for key, value in code.items() if value is not None and raw[key] is None)
    if empty:
        raise ValueError(f"an external recipe must not leave empty what the code states: {empty}")
    if stored is not None and stored.is_file():
        previous = WorldRecipe.model_validate(json.loads(stored.read_text())["recipe"])
        pins = {
            key: getattr(previous, key)
            for key in ("sealed_hash", "calibration_sources")
            if raw.get(key) is None
        }
        current = WorldRecipe.model_validate({**raw, **pins, "seed": previous.seed})
        if current != previous:
            raise ValueError("external recipe differs from the sealed world's recipe")
        return previous
    return WorldRecipe.model_validate(raw)


class WorldRecipe(BaseModel):
    """The parameters a world is generated from.

    `band` is the gate-3 pass mark for the faulted fleet's success over the observed
    window (on the outcome the fleet logs); `aim` is the narrower interval the composer
    accepts, so that the noise of an as-run world of about 1,000 observed episodes
    (a standard error of about 1.5 points) keeps it inside the band. `min_detectable` is
    the historical gate's sizing rule (detectable causes of at least two points whose
    calibrated point is an effect beyond noise); `margin` asks for that many more.
    In measured-v3, `min_detectable` instead requires that many scoring-window
    single-removal causes in at least 95% of the shared calibration resamples; the
    observed predictive interval, with the retained error the recipe names
    (`retained_error`), must fit `band`. `robot_success_floor` is the least expected
    success of every robot in every window (plus the estimator's reserve of 0.03):
    0.40 in the third generation, 0.30 for the fourth generation's development worlds.
    `decoy_margin` and `decoy_margin_of` state the decoy rule's margin.
    `scoring_floor` keeps the scoring window's expected success from falling below it as
    faults with late onsets accumulate (a fleet half broken by the time fixes are valued
    is no longer the fleet the observed window showed). `config_share` is the share of
    configuration and software faults among the effective faults and inert entries (D3:
    two thirds). `total_failures` caps the faults that fail every episode of a robot
    (relative control, one image view, also when a release carries them).
    `min_clean_robots` asks for robots that no effective fault reaches at any scope in the
    observed window (small causes and neutral entries aside), a reference. `repeats` is
    how many plain faults may share a mechanism at one kind of scope (a release counts as
    what it carries); the calibrated configuration and software mechanisms with an effect
    are few (the delay and relative control), so D3's share needs three; D1 repeats
    twice. `worn_robots` bounds how many robots of the interaction's site have worn
    grippers (D3 one, so that its share of configuration faults can hold).
    `strong`
    starts every plain fault at its type's strongest calibrated level and an early onset
    (D1). `provisional` lets causes count that rest on points not measured for the suite
    they act on (`libero_spatial` points applied to `libero_object`) or on an assumed
    joint point: for composing on the CPU before the supplement exists, never for a run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    world_id: str = Field(pattern=r"^w[0-9]{2,}$")
    seed: int = Field(default_factory=new_world_seed, ge=0, lt=WORLD_SEED_LIMIT)
    public_fleet_id: str = Field(default="w01", pattern=r"^w[0-9]{2,}$")
    legacy_identity: bool = False  # explicit historical reproduction only
    kind: WorldKind = "development"
    size: Literal["small", "medium"] = "small"
    sites: int = Field(default=3, ge=1)
    robots_per_site: int = Field(default=4, ge=1)
    days: int = Field(default=10, ge=1)
    episodes_per_robot_day: int = Field(default=14, ge=1)
    windows: Windows = SMALL_WINDOWS
    tasks: tuple[Task, ...] = TASKS
    site_suites: tuple[tuple[str, str], ...] = Field(default=(), exclude_if=lambda v: not v)
    interaction_single_loss_points: float | None = Field(default=None, ge=0, le=100)
    band: tuple[float, float] = (0.60, 0.75)
    aim: tuple[float, float] = (0.645, 0.705)
    scoring_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    robot_success_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    # The decoy rule: a decoy's same-attempt interval must include zero and exclude a loss
    # of `decoy_margin`. "carriers" reads the margin as success on the decoy's own
    # attempts (round 6 as built); "fleet" reads it as fleet success over the scoring
    # window, as the design note's entry 15 separates decoy from cause, which on the
    # carriers is the margin divided by their share of that window's attempts.
    decoy_margin: float = Field(default=0.02, gt=0.0, lt=1.0, exclude_if=lambda v: v == 0.02)
    decoy_margin_of: Literal["carriers", "fleet"] = Field(
        default="carriers", exclude_if=lambda v: v == "carriers"
    )
    # Which retained forecast error the acceptance interval carries
    # (`estimate.RETAINED_ERRORS`): generation-3's historical reserve,
    # generation-4's test reserve, or none for a world measuring the error anew.
    retained_error: Literal["generation-3", "generation-4", "none"] = Field(
        default="generation-3", exclude_if=lambda v: v == "generation-3"
    )
    # The calibration jobs whose cells the recipe stands on (`curves.refinement`); None
    # is the record as it is. A sealed recipe names them, so that a later job cannot
    # move it.
    calibration_jobs: tuple[str, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    # Preserve the proposal order of existing seals. Admission still checks current
    # contracts; round 7 changes proposals and settling, not a sealed world's contents.
    # "r8" is "r7" with the onset days of the interaction's members and of the plain
    # faults drawn and moved as round 6 did (decision of 2026-10-11 on finding F2 of the
    # round-7 review): "r7" ranked days by the forecast and so started faults early.
    search_revision: Literal["r6", "r7", "r8"] = Field(default="r6", exclude_if=lambda v: v == "r6")
    # What admits a draft besides the recipe's other rules. "target": the observed
    # forecast inside `aim` and its interval inside `band`, as every world sealed so far
    # was composed. "interval": the interval inside `band` alone, which is all that entry
    # 20 of the design note names; `aim` then guides the search and refuses nothing.
    admission: Literal["target", "interval"] = Field(
        default="target", exclude_if=lambda v: v == "target"
    )
    warnable_drift: bool = False
    estimator: Literal["measured-v3", "task-v2", "suite-v1"] = "measured-v3"
    calibration_sources: tuple[str, ...] | None = None
    sealed_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    total_failures: int = Field(default=1, ge=0)
    min_clean_robots: int = Field(default=2, ge=0)
    repeats: int = Field(default=1, ge=1)
    worn_robots: tuple[int, int] = (1, 2)  # the interaction: worn grippers at its site
    # Options added for D3 after the supplement (off by default, so that D1 and D2 compose
    # as they ran): settle by levels before onsets, draw interaction pairs by their joint
    # loss, and keep the interaction's site free of other robot-scoped faults.
    settle_levels_first: bool = False
    pairs_by_loss: bool = False
    clear_interaction_site: bool = False
    # Historical recipe compatibility; entry 8 now always judges each member alone.
    members_alone_confirm: bool = False
    provisional: bool = False
    plain: int = Field(default=6, ge=0)
    small: int = Field(default=1, ge=0)
    confounders: int = Field(default=1, ge=0, le=1)
    confounder_companion: Literal["sampling_steps", "calibration_record"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    decoys: int = Field(default=1, ge=0, le=2)
    interactions: int = Field(default=1, ge=0, le=1)
    interaction_scope: Literal["robot", "site"] = "robot"
    additive_pairs: int = Field(default=0, ge=0, le=1)
    fleet_fault: bool = False
    fleet_image_blur: Literal[3, 4] = 4
    growing_drift: bool = False  # historical field; all new drifts must be linear
    extension_types: bool = False
    drifts: int = Field(default=1, ge=0, le=1)
    config_share: float | None = Field(default=None, ge=0.0, le=1.0)
    strong: bool = False
    min_detectable: int = Field(default=6, ge=0)
    margin: int = Field(default=1, ge=0)
    attempts: int = Field(default=400, ge=1)
    note: str = ""

    @property
    def require_warnable_drift(self) -> bool:
        """New worlds always require a precursor; the flag preserves old recipes."""
        return self.estimator != "suite-v1" or self.warnable_drift

    @model_validator(mode="after")
    def _sound(self) -> WorldRecipe:
        if self.legacy_identity:
            if (self.name, self.world_id, self.seed, self.public_fleet_id) != (
                "D1",
                "w41",
                41,
                "w41",
            ):
                raise ValueError("legacy identity is restricted to historical D1/w41")
        else:
            if self.seed < 1 << 62:
                raise ValueError("new worlds require a hidden seed with 62 random bits")
            if self.public_fleet_id == self.world_id:
                raise ValueError("the public fleet id must differ from the truth world id")
            if int(self.public_fleet_id[1:]) == self.seed:
                raise ValueError("the public fleet id must not encode the hidden seed")
        if self.estimator == "suite-v1" and (
            self.kind != "development"
            or (self.world_id, self.seed)
            not in {
                ("w41", 41),
                ("w42", 42),
                ("w43", 43),
                ("w44", 5540438229502422191),
                ("w45", 6689360473052309405),
                ("w46", 6025588286618577878),
                ("w47", 6132661456928807008),
            }
        ):
            raise ValueError("suite-v1 is restricted to the seven historical development worlds")
        if self.kind in ("test", "reserve") and self.estimator != "measured-v3":
            raise ValueError("private worlds require measured context forecasts")
        if self.estimator == "measured-v3" and self.provisional:
            raise ValueError("measured context forecasts cannot be provisional")
        sites = [site for site, _ in self.site_suites]
        if len(sites) != len(set(sites)) or any(
            site not in {f"s{i + 1}" for i in range(self.sites)}
            or suite not in {t.suite for t in self.tasks}
            for site, suite in self.site_suites
        ):
            raise ValueError(
                "site suite restrictions must name unique existing sites and task suites"
            )
        lo, hi = self.band
        alo, ahi = self.aim
        if not (0.0 < lo <= alo < ahi <= hi < 1.0):
            raise ValueError("the aim lies inside the band, both inside (0, 1)")
        if self.windows.scoring.last >= self.days:
            raise ValueError("the scoring window ends after the last day")
        if (
            self.fleet_fault
            and self.confounder_companion == "calibration_record"
            and self.estimator != "measured-v3"
        ):
            raise ValueError("a fleet image fault leaves no camera for an inert calibration record")
        if self.estimator != "measured-v3" and (
            self.decoy_margin != 0.02
            or self.decoy_margin_of != "carriers"
            or self.retained_error != "generation-3"
            or self.calibration_jobs is not None
            or self.admission != "target"
        ):
            raise ValueError("the decoy margin, retained error and jobs are measured-v3 settings")
        return self


DEVELOPMENT: dict[str, WorldRecipe] = {
    "D1": WorldRecipe(
        name="D1",
        estimator="suite-v1",
        world_id="w41",
        seed=41,
        public_fleet_id="w41",
        legacy_identity=True,
        plain=6,
        small=0,
        confounders=0,
        decoys=1,
        interactions=0,
        drifts=0,
        strong=True,
        margin=0,
        repeats=2,
        note="few, clear, strong faults for debugging the pipeline: six plain faults at their "
        "strongest calibrated levels, one decoy, no confounder (experiment.md section 1)",
    ),
    "D2": WorldRecipe(
        name="D2",
        estimator="suite-v1",
        world_id="w44",
        seed=5540438229502422191,  # private pinned 62-bit draw; never exported
        additive_pairs=1,
        extension_types=True,
        margin=0,
        plain=5,
        small=1,
        confounders=1,
        decoys=2,
        interactions=1,
        drifts=1,
        note="the planned difficulty with all four patterns: one confounder pair, two decoys, "
        "one paired interaction, one additive pair, one drift, plain faults and one small cause",
    ),
    "D3": WorldRecipe(
        name="D3",
        estimator="suite-v1",
        world_id="w45",
        seed=6689360473052309405,  # private pinned 62-bit draw; never exported
        additive_pairs=1,
        fleet_fault=True,
        growing_drift=True,
        extension_types=True,
        margin=0,
        plain=5,
        small=0,
        confounders=1,
        decoys=1,
        interactions=1,
        drifts=1,
        config_share=2.0 / 3.0,
        total_failures=2,
        repeats=3,
        worn_robots=(1, 1),
        min_clean_robots=0,
        settle_levels_first=True,
        pairs_by_loss=True,
        clear_interaction_site=True,
        members_alone_confirm=True,
        note="the mix shifted toward configuration and software faults: at least two thirds "
        "of the effective faults and inert entries from those classes, all four patterns, "
        "the additive pair, a fleet configuration fault and a growing scoring drift",
    ),
}

# Historical integration recipes are records. New calibration or code must never
# silently reuse their identifiers; reproduce them with their frozen source instead.
INTEGRATION = {
    name: DEVELOPMENT[name].model_copy(
        update={
            "sealed_hash": digest,
            "calibration_sources": (
                "fleetlab/faults/curves.json sha256 61465e1e731c8d50",
                "fleetlab/worlds/curves_supplement.json sha256 a676af0b51eb985c",
            ),
        }
    )
    for name, digest in {
        "D2": "2bc5a53ff1230988bc7f44e322a141fb86c4611caca1c9df5796feadfd24ab95",
        "D3": "39e7af8254fdca5edc7d4a244104549d836ad84b408c0c637f24ecf259c8135d",
    }.items()
}
# Replacement development worlds: spatial-only sites carry the mild interaction and
# confirmed-flat decoys; s2 retains both suites. Old integration worlds stay frozen.
for _name, _world_id, _seed in (
    ("D2", "w46", 6025588286618577878),
    ("D3", "w47", 6132661456928807008),
):
    DEVELOPMENT[_name] = INTEGRATION[_name].model_copy(
        update={
            "world_id": _world_id,
            "seed": _seed,
            "sealed_hash": None,
            "robot_success_floor": 0.40,
            "warnable_drift": True,
            "site_suites": (("s1", "libero_spatial"), ("s3", "libero_spatial")),
            "interaction_single_loss_points": 10,
            "min_clean_robots": 0,
            "calibration_sources": (
                "fleetlab/faults/curves.json sha256 6c0402890bbe635c",
                "fleetlab/worlds/curves_supplement.json sha256 0fec9a3e17435b69",
            ),
        }
    )

DEVELOPMENT["D2"] = DEVELOPMENT["D2"].model_copy(
    update={
        "plain": 4,
        "fleet_fault": True,
        "confounder_companion": "sampling_steps",
        "interaction_scope": "robot",
        "aim": (0.645, 0.725),
        "sealed_hash": "beaac37883079862454e86fde1e3420b113f420636e81473a81b61fffb62790b",
        "note": "planned difficulty, seven observed causes, a measured fleet image release, "
        "a spatial-only mild interaction, the additive pair and a linear gripper drift",
    }
)
DEVELOPMENT["D3"] = DEVELOPMENT["D3"].model_copy(
    update={
        "plain": 5,
        "fleet_fault": False,
        "confounder_companion": "calibration_record",
        "interaction_scope": "site",
        "sealed_hash": "f03a31c6868d3efcef0813cc2cc3b7d56a7163c2518e0fa9e1c136aa984d6849",
        "note": "two thirds configuration/software faults and inert entries, six observed "
        "causes, a spatial-only site interaction, the additive pair, an inert robot "
        "calibration companion and a linear camera drift",
    }
)


# The fourth generation's settings, decided by the orchestrating session on 2026-10-10
# after round 6c showed that no recipe composes under the rules as built (README, "Round
# 6c"). Each is a field of the recipe, so that a recipe states which rule it was composed
# under; the fields' defaults are the rules as they stood.
#   - The decoy rule's margin is two points of fleet success (design note, entry 15), six
#     points on the carriers of a site-wide decoy; it was two points on the carriers.
#   - No retained error: these worlds measure the forecast's error anew and are accepted
#     when the forecast with its own spread lies inside the band. The third generation's
#     error (-0.7 to +6.0 points, observed) left the forecast a range of 1.3 points.
#   - A floor of 0.30 per robot and window (0.40 before), with the reserve of 0.03. The
#     band is frozen; the floor was added after the freeze (entry 17) and gives way.
# Entry 20 subsequently kept this floor and fleet decoy margin for the test template,
# adding the generation-4 retained error below. Development recipes keep no error.
FOURTH_GENERATION: dict[str, object] = {
    "decoy_margin": 0.02,
    "decoy_margin_of": "fleet",
    "retained_error": "none",
    "robot_success_floor": 0.30,
    # The search's target once no error is retained: the forecast's own spread is about
    # 3.5 points down and 2.5 up, which this leaves inside 60 to 75%. The interval
    # remains the admission criterion.
    "aim": (0.640, 0.715),
}


# How many plain faults of one mechanism the recipes with a fleet-wide release may place
# at one kind of scope (`WorldRecipe.repeats`): three, as D3 has; it was one.
FLEET_RECIPE_REPEATS = 3


def planned_recipe(
    name: Literal["D2", "D3", "planned"], *, world_id: str, seed: int | None = None
) -> WorldRecipe:
    """Unsealed fourth-generation candidate; no test seed or private recipe is stored.

    The benchmark template uses D2's planned difficulty; unseen types remain the
    private composer's responsibility. The third-generation DEVELOPMENT entries "D2" and
    "D3" are immutable records; a fourth-generation world that qualifies is sealed under
    its own world id as key.

    D3 here has its interaction on one robot and four plain faults (the third
    generation had it on a site, with five): with the interaction on a whole site no
    seed composes on measured contexts. It keeps what the design note's section 1
    asks of D3: one confounder pair, one decoy, one interaction, one drift, and two
    thirds of the faults from configuration and software.

    D2 and the template may place up to three plain faults of one mechanism at one
    kind of scope (`repeats`; the third generation's D2 had one, D3 always three).
    Their fleet-wide release holds the software version and the scene camera on every
    robot, which takes the camera faults and every further release off the menu; one
    fault per mechanism and scope then leaves five candidates for the confounder and
    four plain faults, and no seed composes even when every context is measured.
    Section 1 fixes the patterns, the fleet-wide release, one scope per plain fault and
    about ten active faults; it does not limit a mechanism to one fault per scope
    (orchestrating session, 2026-10-10).
    """
    parent = DEVELOPMENT["D3" if name == "D3" else "D2"]
    values = {
        **parent.model_dump(),
        "name": name,
        "world_id": world_id,
        "estimator": "measured-v3",
        "sealed_hash": None,
        "calibration_sources": None,
        "seed": new_world_seed() if seed is None else seed,
        "confounder_companion": "calibration_record",
        "fleet_image_blur": 3,
        "margin": 0,
        "decoys": 1,
        "small": 0,
        "search_revision": "r8",
        "admission": "interval",
        **FOURTH_GENERATION,
        "note": "unsealed candidate; measured contexts, robot-window floor and "
        "scoring-removal resampling; no guarantee of feasibility",
    }
    if name == "D3":
        values.update(interaction_scope="robot", plain=4)
    else:
        values.update(repeats=FLEET_RECIPE_REPEATS)
    if name == "planned":
        values.update(retained_error="generation-4", aim=(0.660, 0.695))
    return WorldRecipe.model_validate(values)


# The fourth generation's development worlds, each sealed under its world id as key (the
# keys "D2" and "D3" stay the third generation's records). A sealed recipe names the
# calibration jobs whose cells it was composed on: a job imported later leaves it as it
# is, and it refuses to compose on anything else.
FOURTH_GENERATION_JOBS = ("20261010-task-cells-r6", "20261010-task-cells-r6c")
DEVELOPMENT["w49"] = planned_recipe(
    "D3",
    world_id="w49",
    seed=7945803106414901658,  # private pinned 62-bit draw; never exported
).model_copy(
    update={
        "calibration_jobs": FOURTH_GENERATION_JOBS,
        "search_revision": "r6",
        "admission": "target",
        "calibration_sources": (
            "fleetlab/faults/curves.json sha256 6c0402890bbe635c",
            "fleetlab/worlds/curves_supplement.json sha256 0fec9a3e17435b69",
            "task_refinement sha256 ed44717b1a2cf3a6",
        ),
        "sealed_hash": "d9aa350ba380f0e080b03859822907f0bf2e57cfdc4932e5034c46f78e2cf202",
        "note": "fourth generation: two thirds configuration/software faults and inert "
        "entries, the interaction on one robot, the additive pair, an inert robot "
        "calibration companion and a linear gripper drift, on measured contexts; no "
        "retained error, floor 0.30, decoy margin two points of fleet success",
    }
)
DEVELOPMENT["w48"] = planned_recipe(
    "D2",
    world_id="w48",
    seed=4801408483580963576,  # private pinned 62-bit draw; never exported
).model_copy(
    update={
        "calibration_jobs": (*FOURTH_GENERATION_JOBS, "20261010-task-cells-r6d"),
        "search_revision": "r6",
        "admission": "target",
        "calibration_sources": (
            "fleetlab/faults/curves.json sha256 6c0402890bbe635c",
            "fleetlab/worlds/curves_supplement.json sha256 0fec9a3e17435b69",
            "task_refinement sha256 3c66af84118dcafd",
        ),
        "sealed_hash": "9c2ca2cfa8201313ad2a4096a70e9cc797ecfc1af4dc9223ff11d604e93bff96",
        "note": "fourth generation: the planned difficulty with a fleet-wide release "
        "carrying a 3-pixel blur, the interaction on one robot, the additive pair, an "
        "inert robot calibration companion and a linear gripper drift, on measured "
        "contexts; up to three plain faults per mechanism and kind of scope; no "
        "retained error, floor 0.30, decoy margin two points of fleet success",
    }
)


# The test worlds' recipe (design note, entries 16 and 20). The six calibration jobs are
# the table the template was assessed on at the end of round 7 (1,081 cells of the
# refinement record); the digest is `curves.measured_sources` of the record as of those
# jobs. Naming both in the recipe is what makes "one recipe for T1 to T5" mean the same
# cells: a job imported later leaves a named-job recipe as it is, and any other change of
# the record under it is refused at composition ("sealed recipe calibration changed").
TEST_WORLD_JOBS = (
    *FOURTH_GENERATION_JOBS,
    "20261010-task-cells-r6d",
    "20261011-task-cells-r7a",
    "20261011-task-cells-r7b",
    "20261011-task-cells-r7c",
)
TEST_WORLD_SOURCES = (
    "fleetlab/faults/curves.json sha256 6c0402890bbe635c",
    "fleetlab/worlds/curves_supplement.json sha256 0fec9a3e17435b69",
    "task_refinement sha256 4d970aba2f5a8550",
)


def stated_fields(recipe: WorldRecipe) -> dict[str, Any]:
    """Every field of a recipe but its seed, as JSON values: what a private file states.

    `model_dump` leaves out the settings that equal the model's defaults, so that the
    records of older worlds keep their bytes; a private recipe must not rely on them.
    """
    from pydantic_core import to_jsonable_python

    fields = {name: getattr(recipe, name) for name in WorldRecipe.model_fields if name != "seed"}
    stated: dict[str, Any] = to_jsonable_python(fields)
    return stated


def private_recipe(
    world_id: str,
    public_fleet_id: str,
    *,
    name: str,
    kind: Literal["test", "reserve"] = "test",
) -> dict[str, Any]:
    """The test worlds' recipe as its private file carries it; nothing is written or drawn.

    The template (`planned_recipe("planned")`) with its six calibration jobs named and
    their digest pinned, every field stated, and no seed: `external_recipe` draws the
    seed when the world is created. The three identity arguments are all that differs
    between the places T1 to T5; `tests/test_worlds_round8.py` pins every other field.
    The unseen fault types are not in it (the recipe has no field for them).
    """
    # The seed is a public placeholder that only satisfies the model; it is not returned.
    template = planned_recipe("planned", world_id=world_id, seed=1 << 62)
    recipe = WorldRecipe.model_validate(
        {
            **template.model_dump(),
            "name": name,
            "kind": kind,
            "public_fleet_id": public_fleet_id,
            "calibration_jobs": TEST_WORLD_JOBS,
            "calibration_sources": TEST_WORLD_SOURCES,
            "note": "test worlds' template: the planned difficulty on measured contexts, "
            "entry 20's settings, six named calibration jobs; no unseen fault types",
        }
    )
    return stated_fields(recipe)


# The fields in which the recipes of the places T1 to T5 differ. Everything else is "the
# recipe" of entry 16, which is fixed before the first draw and whose hash is recorded.
IDENTITY_FIELDS = ("world_id", "public_fleet_id", "name")


def stated_hash(stated: dict[str, Any]) -> str:
    """SHA-256 of a stated recipe's canonical JSON without the three identity fields.

    Canonical: keys sorted, no whitespace, ASCII. The same for every place, so it is one
    number to record on the tracker before the first draw and to find in every ledger
    line afterwards.
    """
    import hashlib

    rules = {key: value for key, value in stated.items() if key not in IDENTITY_FIELDS}
    text = json.dumps(rules, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def recipe_hash(kind: Literal["test", "reserve"] = "test") -> str:
    """The hash of the recipe the running code draws test (or reserve) worlds from."""
    return stated_hash(private_recipe("w00", "w01", name="any", kind=kind))


def code_recipe(path: Path) -> dict[str, Any]:
    """The stated fields of a private recipe file, if it is the running code's recipe.

    A test or reserve world is created only from a file that equals what
    `private_recipe` returns for the file's own identity: any other value, however
    valid, would be another recipe under the same name. Reads the file and nothing
    else; draws no seed.
    """
    private_path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("kind") not in ("test", "reserve"):
        raise ValueError("external recipes are for test or reserve worlds")
    try:
        code = private_recipe(
            raw["world_id"], raw["public_fleet_id"], name=raw["name"], kind=raw["kind"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("an external recipe needs a world id, a fleet id and a name") from error
    if raw != code:
        raise ValueError("the recipe file differs from the code's recipe for its identity")
    return code
