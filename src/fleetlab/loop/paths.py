"""Where one run of the loop keeps its files (truth side; never mounted).

```
<loop>/
  layout.json                       the LoopLayout
  ledgers/ledger-r<N>.json          every version of the ledger, never rewritten
  rounds/r<K>/ops/                  round K's operation run (round 1 under the original
                                    state is the world's own observed run, not copied)
  rounds/r<K>/experiments/<analysis>/allocations.json, session-NN/, session-NN.plan.json
  rounds/r<K>/exports/<name>/       the exports round K's agent read (log/, dictionary)
  evaluation/panel.json             the reserved panel's identities (original state)
  evaluation/<state>/               one engine run per state, on the panel's identities
  evaluation/summary.json           success per state, rescued and broken between states
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["LoopPaths"]


@dataclass(frozen=True)
class LoopPaths:
    root: Path

    @property
    def layout(self) -> Path:
        return self.root / "layout.json"

    @property
    def ledgers(self) -> Path:
        return self.root / "ledgers"

    def ledger(self, version: int) -> Path:
        return self.ledgers / f"ledger-r{version}.json"

    def round(self, number: int) -> Path:
        return self.root / "rounds" / f"r{number}"

    def ops(self, number: int) -> Path:
        return self.round(number) / "ops"

    def experiments(self, number: int, analysis_id: str) -> Path:
        return self.round(number) / "experiments" / analysis_id

    def exports(self, number: int) -> Path:
        return self.round(number) / "exports"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"
