"""Oracle-based evaluation of schema-checked fleet analyses.

`score(ScoringInputs)` evaluates diagnosis, sensitivity, decisions and foresight,
with frozen pass marks and generated error feedback. `python -m fleetlab.scoring`
reads a development-world root and runtime analysis, adds ledger cost accounting,
and writes JSON and Markdown; `--reports` pools world metrics without mixing repeats.
See README.md for input formats, evidence limits and choices where the design is silent.
"""

from fleetlab.scoring.baselines import trivial_no_data_ranker
from fleetlab.scoring.markdown import report_markdown
from fleetlab.scoring.models import (
    Achievable,
    ActionValue,
    BudgetScore,
    CauseVerdict,
    ConfounderResolution,
    DecisionScore,
    DiagnosisScore,
    FaultFinding,
    FixSetValue,
    Headline,
    HeadlineRow,
    JudgedNumber,
    Metric,
    RandomBaseline,
    RankingScore,
    ReplayNeedRecord,
    ScoreReport,
    ScoringInputs,
    TracedNumber,
    TraceScore,
)
from fleetlab.scoring.score import replays_needed_for, score

__all__ = [
    "Achievable",
    "ActionValue",
    "BudgetScore",
    "CauseVerdict",
    "ConfounderResolution",
    "DecisionScore",
    "DiagnosisScore",
    "FaultFinding",
    "FixSetValue",
    "Headline",
    "HeadlineRow",
    "JudgedNumber",
    "Metric",
    "RandomBaseline",
    "RankingScore",
    "ReplayNeedRecord",
    "ScoreReport",
    "ScoringInputs",
    "TraceScore",
    "TracedNumber",
    "replays_needed_for",
    "report_markdown",
    "score",
    "trivial_no_data_ranker",
]
