"""Draw success/rescues against operator use and supervisor idle on separate axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def draw(table: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = json.loads(table.read_text())["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    def group(r: dict[str, Any]) -> tuple[str, int, str, str, str]:
        code = r.get("measured_code") or {}
        revision = str(code.get("commit", "unrecorded"))
        if code.get("dirty"):
            revision += "+dirty"
        return (
            str(r.get("world", "world")),
            int(r["operators"]),
            str(r["rule"]),
            str(r.get("latency_route", "subscription")),
            revision,
        )

    for key in sorted({group(r) for r in rows}):
        world, operators, rule, route, revision = key
        points = sorted(
            [r for r in rows if group(r) == key], key=lambda r: r["estimated_window_operator_s"]
        )
        displayed = revision if revision == "unrecorded" else revision[:7]
        if revision.endswith("+dirty"):
            displayed += "+dirty"
        label = f"{world}, {rule}, {operators} operators, {route}, {displayed}"
        x = [r["estimated_window_operator_s"] / 60 for r in points]
        line = axes[0].plot(
            x, [100 * r["estimated_window_success"] for r in points], "o-", label=label
        )[0]
        axes[1].plot(
            x, [r["estimated_window_net_gain"] for r in points], "o-", color=line.get_color()
        )
        if rule == "supervisor":
            for point in points:
                axes[1].axhline(
                    point["hold_control_net"], color=line.get_color(), linestyle="--", alpha=0.35
                )
            axes[1].legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        color="gray",
                        linestyle="--",
                        label="Recorded-hold control net (outside mark)",
                    )
                ],
                fontsize=7,
            )
        for axis_key, marker in (
            ("estimated_window_subscription_s", "o"),
            ("estimated_window_model_s", "x"),
        ):
            timed = [r for r in points if r[axis_key] is not None]
            axes[2].scatter(
                [r[axis_key] / 60 for r in timed],
                [100 * r["estimated_window_success"] for r in timed],
                marker=marker,
                color=line.get_color(),
            )
    axes[0].set(
        ylabel="Estimated whole-window success (%)", xlabel="Estimated consumed operator minutes"
    )
    axes[1].set(
        ylabel="Window net: rescued minus broken", xlabel="Estimated consumed operator minutes"
    )
    axes[2].set(
        ylabel="Estimated whole-window success (%)",
        xlabel="Decision minutes (subscription / model response)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
    axes[2].legend(
        handles=[
            Line2D([], [], color="gray", marker="o", linestyle="", label="Subscription"),
            Line2D([], [], color="gray", marker="x", linestyle="", label="Model response estimate"),
        ],
        fontsize=7,
    )
    fig.suptitle("Claim 4: recorded operating points; human settings are assumptions")
    legend_rows = (len(labels) + 1) // 2
    fig.tight_layout(rect=(0, min(0.4, 0.045 * legend_rows + 0.02), 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    draw(args.table, args.out)


if __name__ == "__main__":
    main()
