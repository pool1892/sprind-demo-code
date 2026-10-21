"""The scorer's inputs and its report (`ScoreReport`), as data.

The report is a pydantic model, so that it is written as JSON beside the
Markdown that `report_markdown` makes of it. Every number that could not be
computed is None with a reason beside it (`Metric.missing`, `missing` on the
larger records). Missing submissions choose no fixes and find no causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from fleetlab.contracts.actions import COSTS, Action, Budget, CostTable, FixSet
from fleetlab.contracts.faults import Direction, FaultSpec, Target, TruthTable
from fleetlab.contracts.oracle import CurrentValues, Gain, OracleTable
from fleetlab.contracts.tools import Diagnosis, RankedFixSet
from fleetlab.oracle.causes import MeasuredCauses
from fleetlab.oracle.world import Cause as OracleCause
from fleetlab.worlds.truth import FaultTable, Role

# The budget the headline is read at.
HEADLINE_K = 3
# The sizes of the ranked list whose share of the achievable gain claim 2 reports.
TOP_K: tuple[int, ...] = (3, 6)


@dataclass(frozen=True)
class ScoringInputs:
    """Everything the scorer reads for one world and one analysis.

    `oracle` holds the scoring window's episodes with their active faults and
    the replayed rows; `truth` maps each fault kind to its truth mapping
    (`fleetlab.faults.truth_table()`); `current` gives each episode's current
    configuration and conditions, which `resolve` needs for a `set` or
    `change_condition` of a key no fault touches. `diagnosis` and `fix_set`
    are the agent's answer; either may be missing, then no causes are found
    or fixes chosen. Prediction metrics remain undefined. `tool_log` is the
    runtime's call log of the whole analysis, every session. `catalogue` holds complete actions for
    the random baseline; `no_data_ranked` is the no-data baseline's ranked
    list (None: not supplied). `confounders` are the world's confounder pairs
    (effective, inert) from its spec, when the caller has them: the fault
    table does not carry them.
    """

    fault_table: FaultTable
    oracle: OracleTable
    truth: TruthTable
    current: CurrentValues | None = None
    diagnosis: Diagnosis | None = None
    fix_set: RankedFixSet | None = None
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    catalogue: list[Action] = field(default_factory=list)
    no_data_ranked: list[Action] | None = None
    seed: int = 0
    draws: int = 200
    confounders: list[tuple[str, str]] | None = None
    costs: CostTable = field(default_factory=lambda: dict(COSTS))
    as_run: OracleTable | None = None
    drift_faults: list[FaultSpec] = field(default_factory=list)
    issued_day: int | None = None
    baseline_rankings: dict[str, list[Action]] = field(default_factory=dict)
    baseline_fix_sets: dict[str, dict[str, FixSet]] = field(default_factory=dict)
    baseline_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    decision_actions: list[Action] | None = None
    cause_units: tuple[OracleCause, ...] = ()
    additive_pairs: list[tuple[str, str]] = field(default_factory=list)
    oracle_chance: list[dict[str, Any]] = field(default_factory=list)
    chance_command: str = "uv run --locked python -m fleetlab.oracle report WORLD_ROOT"
    drift_crossing_days: dict[str, int | None] = field(default_factory=dict)
    condition: str = "logs alone"
    measured_causes: MeasuredCauses | None = None
    measured_missing: str = "oracle measured cause list absent"
    replay_missing: str | None = None
    analysis_binding_missing: str | None = None


class ReportModel(BaseModel):
    """Base of the report's records: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")


class Metric(ReportModel):
    """A number, or why it could not be computed."""

    value: float | None = None
    missing: str | None = None
    provisional: bool = False

    @classmethod
    def of(cls, value: float | None, missing: str | None = None) -> Metric:
        if value is None and missing is None:
            raise ValueError("a missing metric says why")
        return cls(value=value, missing=None if value is not None else missing)

    @classmethod
    def absent(cls, why: str) -> Metric:
        return cls(value=None, missing=why)


# Claim 1.

Verdict = Literal["hit", "wrong direction", "small", "decoy", "inert", "no fault", "vague"]


class CauseVerdict(ReportModel):
    """One cause of the diagnosis against the fault table."""

    index: int  # position in the agent's list, from 0
    target: Target
    direction: Direction
    confidence: float
    named: bool
    verdict: Verdict
    faults: list[str]  # the faults whose target it matches (same scope, same thing)


class FaultFinding(ReportModel):
    """One fault of the table, and which causes named it."""

    fault_id: str
    kind: str
    role: Role
    target: Target
    detectable: bool
    small: bool
    directions: list[Direction]
    found: bool  # matched by a named cause with an accepted direction
    named_by: list[int]  # named causes matching it, any direction
    wrong_direction_by: list[int]  # named causes matching it with a direction it does not accept


class ConfounderResolution(ReportModel):
    """One confounder pair: the effective fault found, the inert change not named."""

    inert: str
    inert_named: bool
    effective: str | None  # None: the pair is not known (the fault table does not carry it)
    effective_found: bool | None
    resolved: bool | None  # effective found and inert not named; None without the pair


class DiagnosisScore(ReportModel):
    """Claim 1: the diagnosis against the hidden fault table."""

    missing: str | None = None
    detectable: int = 0  # n
    named_cut: int = 0  # n + 2: causes within this rank by confidence are named
    named: int = 0
    recall: Metric = Metric.absent("not computed")
    precision: Metric = Metric.absent("not computed")
    precision_with_vague: Metric = Metric.absent("not computed")  # alias of primary precision
    precision_without_vague: Metric = Metric.absent("not computed")
    targetless: int = 0
    truth_disagreements: list[str] = []
    decoy_rejection: Metric = Metric.absent("not computed")
    hits: int = 0
    false_positives: int = 0
    wrong_direction: int = 0
    small: int = 0
    vague: int = 0
    causes: list[CauseVerdict] = []
    faults: list[FaultFinding] = []
    confounders: list[ConfounderResolution] = []
    confounder_note: str = ""
    units: list[dict[str, Any]] = []


# Claims 2 and 3.


class ActionValue(ReportModel):
    """One ranked action: what the agent expected and what the oracle says."""

    rank: int  # from 1
    action: Action
    expected_gain: float  # points
    low: float
    high: float
    true: Gain | None = None
    true_points: float | None = None
    covered: bool | None = None  # low <= true_points <= high
    missing: str | None = None


class RankingScore(ReportModel):
    """Claim 2: expected against true gains over the ranked list."""

    missing: str | None = None
    actions: list[ActionValue] = []
    valued: int = 0
    spearman: Metric = Metric.absent("not computed")
    mae_points: Metric = Metric.absent("not computed")
    slope: Metric = Metric.absent("not computed")
    coverage: Metric = Metric.absent("not computed")
    top_k_share: dict[str, Metric] = {}  # by k ("3", "6"): the share of the achievable gain
    complete: bool = False


class FixSetValue(ReportModel):
    """A method's fix set at one budget, valued by the oracle."""

    actions: list[Action] = []
    hours: float = 0.0
    gain: Gain | None = None
    points: float | None = None
    share: Metric = Metric.absent("not computed")
    missing: str | None = None


class Achievable(ReportModel):
    """The oracle's best fix set at a budget among the resolved fixes of the active faults."""

    gain: Gain | None = None
    points: float | None = None
    faults: list[str] | None = None  # the faults removed (cost budgets only)
    fixes: list[Action] | None = None  # the cheapest fixing action of each (cost budgets only)
    hours: float | None = None
    missing: str | None = None


class RandomBaseline(ReportModel):
    """Random fixes at one budget: the chance level over the valued draws."""

    draws: int = 0
    valued: int = 0
    skipped_fresh_replay: int = 0  # draws that would need a fresh replay: not valued
    skipped_error: int = 0  # draws whose actions cannot be resolved (for instance no `current`)
    mean_points: Metric = Metric.absent("not computed")
    p5_points: Metric = Metric.absent("not computed")
    p95_points: Metric = Metric.absent("not computed")
    above_zero: Metric = Metric.absent("not computed")  # share of valued draws with net > 0
    mean_share: Metric = Metric.absent("not computed")
    p5_share: Metric = Metric.absent("not computed")
    p95_share: Metric = Metric.absent("not computed")
    missing: str | None = None


class BudgetScore(ReportModel):
    """Claim 3 at one budget: the agent and the baselines against the achievable gain."""

    budget: Budget
    world_id: str | None = None
    condition: str = "logs alone"
    valuation_missing: str | None = None
    analysis_binding_missing: str | None = None
    label: str
    achievable: Achievable
    agent: FixSetValue
    no_data: FixSetValue
    random: RandomBaseline
    baselines: dict[str, FixSetValue] = {}
    baseline_sources: dict[str, dict[str, Any]] = {}
    exact_chance: dict[str, Any] | None = None
    chance_missing: str | None = None
    chance_bounds: dict[str, Any] = {}
    sampling_percentile_range: dict[str, Any] = {}
    chance_command: str = "uv run --locked python -m fleetlab.oracle report WORLD_ROOT"


class DecisionScore(ReportModel):
    """Claim 3: the fix sets at every scoring budget."""

    budgets: list[BudgetScore] = []


class HeadlineRow(ReportModel):
    method: str
    share: Metric
    points: Metric


class Headline(ReportModel):
    """The share of the achievable gain recovered at k = 3."""

    k: int = HEADLINE_K
    rows: list[HeadlineRow] = []
    best_baseline: str | None = None
    chance_p95_share: Metric = Metric.absent("not computed")
    ratio: float | None = None
    verdict: str = ""
    opponents: dict[str, Any] = {}


# Traceability.

TraceStatus = Literal["traced", "echo", "untraced", "small"]


class TracedNumber(ReportModel):
    """A number of the answer and where a tool result shows it."""

    where: str  # e.g. "fix_set.actions[2].expected_gain", "diagnosis.mode_summaries[0]"
    text: str  # as the answer wrote it
    value: float
    status: TraceStatus
    source: str | None = (
        None  # the call that shows it, and the scale ("as written", "x100", "/100")
    )


class JudgedNumber(ReportModel):
    """A number that is judgement, not data: reported, not checked."""

    where: str
    value: float


class TraceScore(ReportModel):
    """Every number of the answer against the tool results of the call log (honesty rule 8)."""

    missing: str | None = None
    calls: int = 0  # calls in the log
    calls_searched: int = 0  # calls whose results count as sources
    traced: int = 0
    echo: int = 0  # found only in results of calls whose own arguments carry the number
    untraced: int = 0
    small: int = 0  # whole numbers below 10 written without decimals: not checked
    numbers: list[TracedNumber] = []
    not_checked: list[JudgedNumber] = []


# The whole report.


class ReplayNeedRecord(ReportModel):
    """An episode to simulate under an effective fault list, and what needs it."""

    episode_id: str
    fault_ids: list[str]
    faults: list[FaultSpec]
    needed_for: list[str]


class ScoreReport(ReportModel):
    """The score of one analysis of one world."""

    world_id: str
    scoring_episodes: int
    seed: int
    draws: int
    claim1: DiagnosisScore
    claim1_curve: DiagnosisScore | None = None
    measured_causes: dict[str, Any] = {}
    claim2: RankingScore
    claim3: DecisionScore
    headline: Headline
    traceability: TraceScore
    needs_fresh_replay: list[ReplayNeedRecord]
    notes: list[str]
    claim5: dict[str, Any] = {}
    claim6: dict[str, Any] = {}
    pass_marks: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    remarks: list[dict[str, Any]] = []
