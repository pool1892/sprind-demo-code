"""The probe figure (lane P; `loop.exhibit.figures.probe_figure`):
commanded against measured from `probe_steps` rows, conditions side by side. Synthetic rows:
a prompt response and one held back by a fixed number of rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fleetlab.loop.exhibit.figures import _first_moves, probe_figure


def rows(delay: int, steps: int = 30, issued: int = 5, size: float = 0.03) -> list[dict[str, Any]]:
    """`probe_steps` rows of a step on x issued in row `issued`: the hand follows the command
    `delay` rows late and covers half of what is left every row."""
    out: list[dict[str, Any]] = []
    x = 0.0
    for k in range(steps + 1):
        target = None if k == steps else [size if k >= issued else 0.0, 0.0, 1.0]
        if k > issued + delay:
            x += (size - x) / 2
        out.append({"step": k, "eef_pos": [x, 0.0, 1.0], "target_pos": target})
    return out


def test_the_figure_draws_conditions_side_by_side_with_the_first_moves(tmp_path: Path) -> None:
    prompt, late = [rows(0), rows(0)], [rows(4), rows(4)]
    assert _first_moves(prompt) == "1" and _first_moves(late) == "5"
    assert _first_moves([rows(2), rows(4)]) == "3/5"
    path = tmp_path / "probe.png"
    probe_figure(path, [("prompt", prompt), ("late", late)], "a step", "toy", reference=rows(0))
    assert path.stat().st_size > 10_000
    again = tmp_path / "again.png"
    probe_figure(again, [("prompt", prompt), ("late", late)], "a step", "toy", reference=rows(0))
    assert again.read_bytes() == path.read_bytes()  # no date in the file
