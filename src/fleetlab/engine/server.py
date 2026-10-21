"""The inference server's scheduling rule (contracts, section 10; xvla-speed section 5).

The batch size is fixed for a run (replay rule 3). For a row-independent
policy the row is free: any waiting request takes any free row, the requests
that have waited longest first, and rows nobody asked for are padded with a
copy of row 0. Requests whose effective spec differs in what the sampling
loop reads (`episode.call_key`) cannot share a call; the oldest waiting
request decides which group is served next. A request with a fixed row (a
policy whose chunks depend on the row) goes into that row, and two such
requests for the same row wait for different calls.

No waiting period: with about two and a half workers per row the next
requests are already waiting when a call ends (xvla-speed section 5).
`Batcher` is the rule without processes, so that it can be tested alone.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from fleetlab.engine.episode import ChunkRequest, call_key


@dataclass
class Batcher[W]:  # W: who waits, a worker id
    batch_size: int
    waiting: OrderedDict[W, ChunkRequest] = field(default_factory=OrderedDict)
    calls: list[int] = field(default_factory=list)  # rows in use per call

    def add(self, who: W, request: ChunkRequest) -> None:
        if request.batch_size != self.batch_size:
            raise ValueError(
                f"request for batch size {request.batch_size} in a run of {self.batch_size}"
            )
        if who in self.waiting:
            raise ValueError(f"{who!r} is already waiting")
        self.waiting[who] = request

    def next_call(self) -> list[tuple[W, ChunkRequest]]:
        """The requests of the next call, in order of arrival; empty when nothing waits."""
        if not self.waiting:
            return []
        first = next(iter(self.waiting.values()))
        key = call_key(first.spec)
        chosen: list[tuple[W, ChunkRequest]] = []
        rows_taken: set[int] = set()
        free_rows = self.batch_size
        for who, request in self.waiting.items():
            if len(chosen) == self.batch_size:
                break
            if call_key(request.spec) != key:
                continue
            if request.row is not None:
                if request.row in rows_taken:
                    continue
                rows_taken.add(request.row)
            elif free_rows - len(rows_taken) - sum(r.row is None for _, r in chosen) <= 0:
                continue
            chosen.append((who, request))
        for who, _ in chosen:
            del self.waiting[who]
        self.calls.append(len(chosen))
        return chosen

    def rows_in_use(self) -> float:
        """The share of rows that carried a request, over all calls so far."""
        return sum(self.calls) / (len(self.calls) * self.batch_size) if self.calls else 0.0
