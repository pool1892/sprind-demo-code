"""Which skills each role gets: the analysis agent, the live supervisor, the job agent.

The skills live in `skills/` at the repository root, one directory each
(docs/design/contracts.md, section 13; skills/README.md). The runtime mounts
a role's skills into the agent's sandbox; this module names them, so that the
list lives in one place and the runtime does not repeat it. It holds names
only: no procedure, and nothing of a world.

    from fleetlab.skill_library import skill_dirs
    skill_dirs("analysis")      # the eight ex-post skills, in pipeline order
    skill_dirs("jobs")          # the job agent's eight skills (docs/design/jobs.md)
"""

from __future__ import annotations

from pathlib import Path

import fleetlab

__all__ = [
    "ANALYSIS_SKILLS",
    "JOB_SKILLS",
    "ROLES",
    "SUPERVISOR_SKILLS",
    "library_dir",
    "skill_dirs",
]

# The ex-post pipeline, in the order an analysis usually takes them.
ANALYSIS_SKILLS: tuple[str, ...] = (
    "triage",
    "episode-cards",
    "failure-modes",
    "scope",
    "sensitivity-from-logs",
    "sensitivity-by-experiment",
    "rank-actions",
    "write-diagnosis",
)
# The live supervisor's skills; they use no tool of the tool server.
SUPERVISOR_SKILLS: tuple[str, ...] = ("assess-risk", "choose-intervention")
# The job agent's skills (lane K's skills/jobs-*), in the order its task text names them
# (`fleetlab.jobs.task_text.JOB_SKILLS`, which a test keeps equal to this list).
JOB_SKILLS: tuple[str, ...] = (
    "jobs-traces",
    "jobs-installation",
    "jobs-release",
    "jobs-decompose",
    "jobs-practice",
    "jobs-preserve",
    "jobs-transitions",
    "jobs-counts",
    "jobs-check",
)
ROLES: dict[str, tuple[str, ...]] = {
    "analysis": ANALYSIS_SKILLS,
    "supervisor": SUPERVISOR_SKILLS,
    "jobs": JOB_SKILLS,
}


def library_dir() -> Path:
    """The `skills/` directory of the checkout this package is installed from (editable)."""
    return Path(fleetlab.__file__).resolve().parents[2] / "skills"


def skill_dirs(role: str, root: Path | str | None = None) -> list[Path]:
    """The directories of a role's skills under `root` (default: the checkout's library)."""
    if role not in ROLES:
        raise KeyError(f"no role {role!r}; the roles are {sorted(ROLES)}")
    base = Path(root) if root is not None else library_dir()
    return [base / name for name in ROLES[role]]


KNOWLEDGE_BEGIN = "<!-- fault-class-knowledge:begin -->"
KNOWLEDGE_END = "<!-- fault-class-knowledge:end -->"


def strip_fault_class_knowledge(text: str) -> str:
    """Remove explicitly marked class knowledge; malformed markers fail closed."""
    kept = []
    inside = False
    for line in text.splitlines(keepends=True):
        if KNOWLEDGE_BEGIN in line:
            if inside or line.strip() != KNOWLEDGE_BEGIN:
                raise ValueError("invalid fault-class knowledge opening marker")
            inside = True
        elif KNOWLEDGE_END in line:
            if not inside or line.strip() != KNOWLEDGE_END:
                raise ValueError("invalid fault-class knowledge closing marker")
            inside = False
        elif not inside:
            kept.append(line)
    if inside:
        raise ValueError("unclosed fault-class knowledge passage")
    return "".join(kept)
