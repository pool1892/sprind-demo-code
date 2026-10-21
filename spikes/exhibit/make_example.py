"""Assemble the example exhibit of the real miniature investigation p1-astra-2049.

Round 1 is the real run (lane driver, the analysis agent on w10, two real experiments). Round 2,
the reserved evaluation and the reuse comparison do not exist for it; they are filled
from `fleetlab.loop.exhibit.fixtures` under a FIXTURE marker, so every page, table and
figure built from them says so. The annotations carry an independent check of the public trace as
the lead relayed it on 13 October 2026, attributed as a reviewer's labels.

    python make_example.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fleetlab.loop.exhibit.fixtures import write_fixture

RUN = "/var/tmp/robots-loop/driver/p1-astra-2049/run.json"
W10 = {"r01": "s1", "r02": "s1", "r03": "s1", "r04": "s1",
       "r05": "s2", "r06": "s2", "r07": "s2", "r08": "s2"}
BY = "an independent check of the run's public trace, relayed by the lead on 13 October 2026"


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    engagement = write_fixture(
        out / "fixture", world_id="w10", robots=W10, rounds=(2,), first_session=3,
        days_per_round=1, first_day=3, per_robot_day=13, weak={"r04": 0.35, "r06": 0.6},
        seed=10, videos=False,
    )
    padding = "padding forced by the old format (at least ten rows)"
    labels = [
        {"section": "ranked", "rank": 1, "label": "substantive: the site delay reversion. Its "
         "37.5 points are a site forecast transported from r08's experiment, not a measured "
         "all-site gain"},
        {"section": "ranked", "rank": 2, "label": "substantive: the camera recalibration on "
         "r04. Its 6.88 points scale the r04 experiment by exposure: a forecast"},
        *[{"section": "ranked", "rank": r, "label": f"overlapping delay fallback; {padding}"}
          for r in range(3, 8)],
        {"section": "ranked", "rank": 8, "label": f"the camera-edit alternative that did not "
         f"help in x0001; {padding}"},
        {"section": "ranked", "rank": 9, "label": f"untested cleaning; {padding}"},
        {"section": "ranked", "rank": 10, "label": f"service, which did not help in x0002; {padding}"},
    ]
    annotations = {
        "by": BY,
        "note_label": "Reviewer's note",
        "run_caption": "Miniature investigation plumbing on the eight-robot development world "
        "w10: one real investigation with two real experiments, in the older single-round "
        "format. It is not evidence of a retained loop or of adaptation. Round 2, the reserved "
        "evaluation and the reuse comparison on these pages are fixtures.",
        "labels": labels,
        "notes": [
            {"page": "round-1", "section": "decision", "text": "Only two recommendations are "
             "substantive: reverting site s2's action delay and recalibrating r04's scene "
             "camera, 0.6 operator-hours together by the catalogue's estimate. Ranks 3 to 10 "
             "are what the old format's ten-row minimum forced; they are not eight further "
             "supported improvements."},
            {"page": "round-1", "section": "working", "text": "The agent left three milk "
             "failures from before the delay, and one bowl failure, unexplained, and made no "
             "unsupported claim of drift or retraining."},
            {"page": "round-1", "section": "experiments", "text": "The trace agrees with these "
             "counts (high confidence). How far the delay result on r08 carries to the rest "
             "of site s2 is a moderate-confidence transfer. The paired images were retrieved "
             "by the agent; the reviewer did not judge their contents."},
        ],
    }
    (out / "annotations.json").write_text(json.dumps(annotations, indent=1) + "\n")
    manifest = {
        "format": "fleetlab.exhibit-manifest/1",
        "label": "Miniature investigation p1-astra-2049 on w10",
        "rounds": [
            {"investigation": RUN},
            {"engagement": str(engagement.relative_to(out)), "round": 2},
        ],
        "evaluation": "fixture/evaluation",
        "one_look_list": "fixture/engagement/one-look-list.json",
        "reuse": "fixture/reuse/comparison.json",
        "truth_world": "~/.cache/robots-probes/mini/w10-v2",
        "annotations": "annotations.json",
    }
    (out / "exhibit.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(out / "exhibit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
