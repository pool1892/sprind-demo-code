from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from fleetlab.film import charts, story


def film_data() -> dict[str, Any]:
    """A small film-data.json: one job on three locked layouts, a fleet of four robots."""
    starts = [30, 31, 32]
    chain, job = [True, False, False], [True, True, True]
    clips = {}
    for kind, outcomes in (("chain", chain), ("job", job)):
        for s, ok in zip(starts, outcomes, strict=True):
            clips[f"r1-{kind}-{s}-tile"] = {
                "start": s,
                "delivered": ok,
                "run": f"probes:jobs/{kind}/run",
                "episode": f"e{s}",
            }
    probe_cmd = [0.0] * 5 + [3.0] * 20 + [0.0] * 20
    engagement = [{"record": "eng", "id": "probes:fleet/engagement.json", "sha256": "ab" * 32}]
    table = (
        "| Experiment | Change tested | Without it | With it | Rescued | Broken |\n"
        "|---|---|---|---|---|---|\n"
        "| x1 | a latency compensation of 4 control steps at site s1 "
        "| 5 of 10 | 9 of 10 | 4 | 0 |\n"
        "| x1 | a latency compensation of 8 control steps at site s1 "
        "| 5 of 10 | 7 of 10 | 3 | 1 |\n"
        "| x2 | a service of robot r03's controller | 4 of 8 | 8 of 8 | 4 | 0 |"
    )
    return {
        "numbers": {
            "r1_delivered": "3",
            "r1_release_n": "3",
            "r1_release_k": "2",
            "r1_naive_delivered": "1",
            "r1_rescued": "2",
            "r1_broken": "0",
            "b_round1_count": "2",
            "b_eval_original_k": "6",
            "b_eval_final_k": "9",
            "b_eval_final_n": "10",
            "b_eval_rescued": "4",
            "b_eval_broken": "1",
        },
        "programs": {
            "r1": [
                {
                    "job": "j1",
                    "label": "plain",
                    "role": "plain",
                    "outcomes": "010",
                    "starts": [0, 1, 2],
                    "k": 1,
                    "n": 3,
                    "source": "probes:jobs/session-1/episodes.jsonl",
                },
                {
                    "job": "j2",
                    "label": "worse",
                    "role": "other",
                    "outcomes": "000",
                    "starts": [0, 1, 2],
                    "k": 0,
                    "n": 3,
                    "source": "probes:jobs/session-2/episodes.jsonl",
                },
                {
                    "job": "j3",
                    "label": "better",
                    "role": "winner",
                    "outcomes": "111",
                    "starts": [0, 1, 2],
                    "k": 3,
                    "n": 3,
                    "source": "probes:jobs/session-2/episodes.jsonl",
                },
            ]
        },
        "clips": clips,
        "mosaics": {
            "r1-chain": [f"r1-chain-{s}-tile" for s in starts],
            "r1-job": [f"r1-job-{s}-tile" for s in starts],
        },
        "fleet": {
            "regular_work": {
                "sites": {
                    "s1": {
                        "robots": {"r01": {"k": 5, "n": 10}, "r02": {"k": 6, "n": 10}},
                        "k": 11,
                        "n": 20,
                    },
                    "s2": {
                        "robots": {"r03": {"k": 9, "n": 10}, "r04": {"k": 10, "n": 10}},
                        "k": 19,
                        "n": 20,
                    },
                },
                "source": "probes:fleet/episodes.parquet",
            },
            "b_linked_site": {"value": "s1"},
            "b_round1": {
                "value": "a latency compensation of 4 control steps at site s1; "
                "a service of robot r03's controller",
                "inputs": engagement,
            },
            "b_experiments_table": {
                "value": table,
                "inputs": [{"id": "probes:fleet/w80-x/experiments.jsonl", "sha256": "cd" * 32}],
            },
            "b_probe": {
                "r01": {"site": "s1", "repeats": 1, "first_move": "3 and 5", "half_way": "4 and 6"},
                "how": "read from the recorded steps",
                "inputs": [{"id": "probes:fleet/probe_episodes.parquet", "sha256": "ef" * 32}],
            },
            "experiments": [
                {
                    "experiment": "x1",
                    "change": "a latency compensation of 4 control steps at site s1",
                    "without": {"k": 5, "n": 10},
                    "with": {"k": 9, "n": 10},
                    "rescued": 4,
                    "broken": 0,
                },
                {
                    "experiment": "x1",
                    "change": "a latency compensation of 8 control steps at site s1",
                    "without": {"k": 5, "n": 10},
                    "with": {"k": 7, "n": 10},
                    "rescued": 3,
                    "broken": 1,
                },
                {
                    "experiment": "x2",
                    "change": "a service of robot r03's controller",
                    "without": {"k": 4, "n": 8},
                    "with": {"k": 8, "n": 8},
                    "rescued": 4,
                    "broken": 0,
                },
            ],
            "probe": {
                "step_s": 0.05,
                "source": "probes:fleet/probe_episodes.parquet",
                "axis_s": [0, 1, 2],
                "origin_cm": 0.0,
                "robots": {
                    # out 3 control steps late, back 5 late
                    "r01": {
                        "site": "s1",
                        "commanded_cm": probe_cmd,
                        "measured_cm": [0.0] * 8 + [3.0] * 22 + [0.0] * 15,
                    },
                    "r04": {
                        "site": "s2",
                        "commanded_cm": probe_cmd,
                        "measured_cm": [0.0] * 6 + [3.0] * 20 + [0.0] * 19,
                    },
                },
            },
            "evaluation": {
                "before": 6,
                "after": 9,
                "attempts": 10,
                "pairs": {"rescued": 4, "broken": 1},
            },
        },
        "provenance": {"inputs": []},
    }


STORY = """---
title: A test story
subtitle: In two parts
kicker: Test
healthy: r04
job_r1: Job one
---

<!-- a note to the author, never shown -->

The answer: {r1_delivered} of {r1_release_n}, from {r1_naive_delivered}.

## First part {#first data-toc="First"}

The fleet at {b_link_site} waits {b_probe_r01_out_s} seconds; robot {b_overload_robot} is
serviced; {b_comp_settings} settings tested; healthy: {b_healthy_robots}, {b_healthy_k} of
{b_healthy_n}. Intervals at {ci_level}.

::: figure notebook r1
The notebook
Each dot is a layout.
:::

::: figure exam r1
The exam
One dot per layout.
:::

## Second part {#second}

::: figure fleet
The fleet
By robot.
:::

::: figure probe r01="a delay" r04="healthy"
The probe
:::
"""


def write(
    tmp_path: Path, text: str = STORY, data: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    t, d = tmp_path / "story.md", tmp_path / "film-data.json"
    t.write_text(text)
    d.write_text(json.dumps(data or film_data()))
    return t, d


def test_lags_time_each_step_from_the_recorded_motion() -> None:
    d = film_data()["fleet"]["probe"]["robots"]["r01"]
    assert charts.lags(d["commanded_cm"], d["measured_cm"]) == [(5, 3), (25, 5)]


def test_the_story_fills_every_number_and_draws_its_charts(tmp_path: Path) -> None:
    text = (
        STORY
        + "\n::: table repairs\nThe repairs\n:::\n\n::: figure fleet-result\nThe result\n:::\n"
        + "\n::: figure notebook r1 mute=other\nThe short notebook\n:::\n"
    )
    told = story.render(*write(tmp_path, text))
    assert told.title == "A test story" and told.kicker == "Test"
    assert "3 of 3, from 1" in told.lede and "note to the author" not in told.lede + told.body
    assert told.toc == [("first", "First"), ("second", "Second part")]
    body = told.body
    flat = " ".join(body.split())
    assert "site 1 waits 0.15 seconds" in flat and "robot r03 is" in flat
    assert "2 settings tested" in flat and "healthy: r04, 10 of 10" in flat
    assert "Intervals at 95%." in flat
    assert "simulated controller service for r03" in body
    # the short notebook: the other programs wait behind the full list
    short = body.split("The short notebook", 1)[1]
    assert '<details class="all-rows"><summary>All 3 programs, in the order' in short
    assert short.split("<details", 1)[0].count('class="dot-row') == 2
    assert [c["kind"] for c in told.charts][:3] == ["notebook", "exam", "fleet"]
    assert told.charts[0]["values"][0]["k"] == 1 and told.charts[0]["sources"]
    assert "{" not in body and ":::" not in body
    assert body.count('<figure class="chart') == 6 and told.figures == 7
    assert "2 improved, 0 worsened" in body  # the exam's note, from the bound numbers
    first_two = body.split("The short notebook", 1)[0]
    assert len(re.findall(r'<circle class="ok"[^>]*><title>', first_two)) == 1 + 3 + 1 + 3
    assert 'class="adopted"' in body and 'class="declined"' in body
    assert body.count('class="cell-new" x=') == 4  # rescued, in the grid after
    assert len(re.findall(r'class="cell-lost" x="\d+" y="\d+" width="6"', body)) == 1
    assert told.used["b_probe_r01_out_s"].value == "0.15"
    assert told.used["r1_delivered"].source.startswith("numbers")


GLANCE = """glance.r1: Experiment one · job one | Bowl on the cabinet, / drawer open | completed
glance.fleet: Experiment two · fleet | A fleet | success on {b_eval_final_n} attempts / unseen
glance.marks.jobs: plain chain | the job
glance.marks.fleet: as it was | with the {b_round1_count} changes
glance.legend: completed | failed | a share
"""


def test_the_glance_draws_every_result_in_each_layout_from_the_records(tmp_path: Path) -> None:
    text = STORY.replace("job_r1: Job one\n", "job_r1: Job one\n" + GLANCE)
    text += "\n::: figure glance r1 fleet\nAt a glance\n:::\n"
    told = story.render(*write(tmp_path, text))
    fig = told.body.split('class="chart chart-glance"', 1)[1].split('<details class="numbers"')[0]
    assert [f"glance-{x}" in fig for x in ("rows", "stack")] == [True, True]
    # the job on its three locked layouts: the chain completes one, the job all three; both
    # layouts of the image draw the same marks
    assert fig.count("plain chain, final test layout 31: failed") == 2
    assert fig.count(">1 of 3<") == 2 and fig.count(">3 of 3<") == 2
    # the fleet as shares, its counts by the bars; the key in the page's own legend
    assert fig.count(">60%<") == 2 and fig.count(">90%<") == 2 and fig.count(">9 of 10<") == 2
    assert '<p class="legend">' in fig and "a share" in fig
    assert "with the 2 changes" in fig and ">drawer open<" in fig and ">unseen<" in fig
    drawn = next(c for c in told.charts if c["kind"] == "glance")
    assert [v["after"] for v in drawn["values"]] == ["3", "90%"] and drawn["sources"]
    # a bound count the tiles do not show stops the build
    short = text.split("## First part", 1)[0] + "## Glance\n\n::: figure glance r1\nIt\n:::\n"
    data = film_data()
    data["numbers"]["r1_naive_delivered"] = "2"
    with pytest.raises(story.StoryError, match="r1_naive_delivered: bound 2, the glance 1"):
        story.render(*write(tmp_path, short, data))


def test_an_unknown_placeholder_fails(tmp_path: Path) -> None:
    with pytest.raises(story.StoryError, match=r"\{r9_typo\}"):
        story.render(*write(tmp_path, STORY + "\nAnd {r9_typo}.\n"))


def test_banned_words_fail(tmp_path: Path) -> None:
    with pytest.raises(story.StoryError, match="only"):
        story.render(*write(tmp_path, STORY + "\nIt delivered only twice.\n"))


def test_a_count_that_disagrees_with_its_table_fails(tmp_path: Path) -> None:
    data = film_data()
    data["numbers"]["r1_rescued"] = "3"
    with pytest.raises(story.StoryError, match="r1_rescued"):
        story.render(*write(tmp_path, data=data))
    data = film_data()
    data["fleet"]["experiments"][0]["change"] = "a new kind of repair at site s1"
    with pytest.raises(story.StoryError, match="plain words"):
        story.render(*write(tmp_path, data=data))


def test_the_fleet_record_counts_with_intervals_and_sources() -> None:
    record = story.fleet_record(film_data(), ["r04"])
    body = record.body
    # from 20 attempts on, a count carries its Wilson interval; below, its share alone
    assert '11 of 20<br><span class="ci">55% (34% to 74%)</span>' in body
    assert '10 of 10<br><span class="ci">100%</span>' in body
    assert "Healthy reference" in body and "simulated controller service" in body
    assert "<td>3 and 5</td><td>4 and 6</td>" in body
    assert "probes:fleet/engagement.json" in body and "4 improved, 1 worsened" in body
    assert [r["robot"] for r in record.tables["regular_work"]][-1] == "r04"
    assert record.tables["repairs"][2]["decision"] == "adopted"


def test_the_fleet_record_refuses_an_interval_the_record_does_not_state() -> None:
    data = film_data()
    x = data["fleet"]["experiments"][1]
    x["without"] = {"k": 10, "n": 20}
    data["fleet"]["b_experiments_table"]["value"] = data["fleet"]["b_experiments_table"][
        "value"
    ].replace("| 5 of 10 | 7 of 10 |", "| 10 of 20 (50%; 95% interval 20% to 80%) | 7 of 10 |")
    with pytest.raises(story.StoryError, match="Wilson"):
        story.fleet_record(data, ["r04"])
    data["fleet"]["b_experiments_table"]["value"] = data["fleet"]["b_experiments_table"][
        "value"
    ].replace("20% to 80%", "30% to 70%")
    assert "10 of 20" in story.fleet_record(data, ["r04"]).body
