"""The episode log, version 1: writing a run, exporting what the agents see, reading it back.

docs/design/contracts.md, section 11; the schema is `fleetlab.contracts.log`.
See README.md in this directory.

| module | what |
|---|---|
| `writer` | `RunWriter`, `EpisodeWriter`: a run directory in the contracts' layout |
| `export` | `export_runs`: the tree the agents' tool server mounts |
| `dictionary` | the data dictionary the agents receive |
| `leaks` | the word search and the canary check of an export |
| `reader` | `LogReader`: DuckDB over an export; frames by episode and step |
| `runfiles` | reading a whole run, hidden tables included (oracle, dashboard, tests) |

`fleetlab.faults` is imported only inside `leaks.truth_terms`, never when this
package is imported.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Only the reader's side is imported with the package. Writer, export and leak
# search load on first use (`__getattr__`), so that a process that only reads, the
# agents' tool server in its sandbox above all, loads and needs nothing else of this
# package: its sandbox holds exactly the modules it imports (fleetlab.toolserver;
# review of the runtime, M6).
from fleetlab.log.dictionary import data_dictionary
from fleetlab.log.reader import Frame, LogReader

if TYPE_CHECKING:
    from fleetlab.log._rows import RowError, Rows
    from fleetlab.log.export import ExportRefused, ExportReport, export_problems, export_runs
    from fleetlab.log.leaks import LeakReport, audit_export, leak_search
    from fleetlab.log.writer import (
        EpisodeFiles,
        EpisodeWriter,
        ResumeMismatch,
        RunLocked,
        RunStatus,
        RunWriter,
        SchemaError,
        VideoSettings,
        run_status,
    )

_LAZY = {
    "RowError": "_rows",
    "Rows": "_rows",
    "ExportRefused": "export",
    "ExportReport": "export",
    "export_problems": "export",
    "export_runs": "export",
    "LeakReport": "leaks",
    "audit_export": "leaks",
    "leak_search": "leaks",
    "EpisodeFiles": "writer",
    "EpisodeWriter": "writer",
    "ResumeMismatch": "writer",
    "RunLocked": "writer",
    "RunStatus": "writer",
    "RunWriter": "writer",
    "SchemaError": "writer",
    "VideoSettings": "writer",
    "run_status": "writer",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


__all__ = [
    "EpisodeFiles",
    "EpisodeWriter",
    "ExportRefused",
    "ExportReport",
    "Frame",
    "LeakReport",
    "LogReader",
    "ResumeMismatch",
    "RowError",
    "Rows",
    "RunLocked",
    "RunStatus",
    "RunWriter",
    "SchemaError",
    "VideoSettings",
    "audit_export",
    "data_dictionary",
    "export_problems",
    "export_runs",
    "leak_search",
    "run_status",
]
