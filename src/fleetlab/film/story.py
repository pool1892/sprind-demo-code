"""The story page: the site's narrative, every number filled from the records at build time.

A site description (`fleetlab.film-site/1`) may name a story:

    "story": {"text": "../story/story.md",
              "data": "../explainer/film/public/data/film-data.json"}

`text` is the page in Markdown. Its front matter gives `title`, `subtitle`, `kicker` and the
editorial facts the charts need (`healthy`: the robots no planted problem touches;
`job_<prefix>`: a job's name in the exam chart). The part before the first `##` heading is
the answer under the title; every `##` heading opens a section (`## Title {#id}`).

No number is typed into the text: each is a placeholder `{name}`, filled from `data`, the
film's bound data (`site/explainer/build_data.py`): its `numbers` as the bound script has
them, and values derived here from its recorded tables (the fleet's regular work, the probe's
recorded steps, the tested repairs, the evaluation). An unknown placeholder fails the build,
and so do words the page must not use ("only", "just", "merely", "anecdotal").

A block

    ::: figure <kind> [args]
    The figure's title
    Its caption, in Markdown, placeholders allowed.
    :::

becomes a chart drawn from the same data (`fleetlab.film.charts`): `exam r1 r2`,
`notebook r1 [mute=other]`, `fleet`, `probe r01="tag" r06="tag" ...`, `fleet-result`,
`glance r1 r2 fleet` (the results at a glance, its words in the front matter's `glance.*`);
`::: table repairs` is the fleet's tested repairs. Every value the page used is written to
`story-values.json` beside the page, with where it came from.
"""

from __future__ import annotations

import html
import inspect
import json
import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.film import charts, public
from fleetlab.film.record import MIN_RUNS_FOR_INTERVAL, Z95, proportion
from fleetlab.film.sources import sha256
from fleetlab.loop.probes import step_lags

# The travel that counts as the hand's first movement in the probe records: the threshold
# the agent's instrument measured with (step_lags's default), not a number typed here.
PROBE_MOVE_M: float = inspect.signature(step_lags).parameters["threshold_m"].default

__all__ = ["Story", "StoryError", "Value", "derive", "fill", "fleet_record", "public_id", "render"]


class StoryError(ValueError):
    """The story cannot be built as written: an unknown placeholder, a missing record, a
    count that disagrees with its table, or a word the page must not use."""


@dataclass
class Value:
    value: str
    source: str


@dataclass
class Story:
    title: str
    subtitle: str
    kicker: str
    lede: str  # HTML
    lede_text: str  # plain text, for previews
    body: str  # HTML: the sections
    toc: list[tuple[str, str]]  # (id, heading)
    used: dict[str, Value] = field(default_factory=dict)
    data_sha256: str = ""
    text_sha256: str = ""
    figures: int = 0
    values: dict[str, Value] = field(default_factory=dict)  # every value the page could use
    charts: list[dict[str, Any]] = field(default_factory=list)  # each chart's numbers
    data: dict[str, Any] = field(default_factory=dict)  # the film data it was built from
    meta: dict[str, str] = field(default_factory=dict)  # the front matter, filled


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
BLOCK = re.compile(r"^::: *(figure|table) +([^\n]*)\n(.*?)^:::[ \t]*$", re.M | re.S)
BANNED = re.compile(r"\b(only|just|merely|anecdotal(?:ly)?)\b", re.I)
SITE_ID = re.compile(r"^s(\d+)$")


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end() :]


def _pct(k: int, n: int) -> str:
    return f"{round(100 * k / n)}%" if n else "n/a"


def site_label(site: str) -> str:
    """`s1` -> `site 1`: the fleet's public site id in words."""
    match = SITE_ID.match(site)
    if not match:
        raise StoryError(f"not a site id: {site!r}")
    return f"site {match.group(1)}"


def _seconds(steps: int, step_s: float) -> str:
    return f"{steps * step_s:.2f}".rstrip("0").rstrip(".")


# ---- the values ------------------------------------------------------------------------------

REPAIRS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^a latency compensation of (?P<steps>\d+) control steps at site (?P<site>s\d+)$"
        ),
        "latency compensation, {steps} control steps",
    ),
    (
        re.compile(r"^a rollback of the software update at site (?P<site>s\d+)$"),
        "roll back the software update",
    ),
    (
        re.compile(r"^a rollback of the scene camera's calibration change at site (?P<site>s\d+)$"),
        "roll back the camera calibration change",
    ),
    (
        re.compile(r"^a service of robot (?P<robot>r\d+)'s controller$"),
        "simulated controller service for {robot}",
    ),
]


def _robot_sites(data: dict[str, Any]) -> dict[str, str]:
    sites = data["fleet"]["regular_work"]["sites"]
    return {robot: site for site, s in sites.items() for robot in s["robots"]}


def _repair(change: str, robot_sites: dict[str, str]) -> tuple[str, str, dict[str, str]]:
    """A recorded change in plain words: (site id, words, the pattern's fields)."""
    for pattern, words in REPAIRS:
        match = pattern.match(change)
        if match:
            fields = match.groupdict()
            site = fields.get("site") or robot_sites.get(fields.get("robot", ""), "")
            if not site:
                raise StoryError(f"no site for the change {change!r}")
            return site, words.format(**fields), fields
    raise StoryError(f"no plain words for the recorded change {change!r}; add them to REPAIRS")


def derive(data: dict[str, Any], meta: dict[str, str]) -> dict[str, Value]:
    """Every value the story may use: the bound numbers, then values derived from the film
    data's recorded tables. A derived value never replaces a bound one."""
    out = {str(k): Value(str(v), "numbers (the bound script)") for k, v in data["numbers"].items()}

    def add(name: str, value: object, how: str) -> None:
        if name in out:
            raise StoryError(f"the derived value {name} would replace a bound number")
        out[name] = Value(str(value), how)

    fleet = data["fleet"]
    robot_sites = _robot_sites(data)
    sites = fleet["regular_work"]["sites"]
    for site, s in sites.items():
        how = f"fleet.regular_work: {site}"
        add(f"b_{site}_label", site_label(site), "the site's public id in words")
        add(f"b_{site}_k", s["k"], how + ", successes")
        add(f"b_{site}_n", s["n"], how + ", attempts")
        add(f"b_{site}_pct", _pct(s["k"], s["n"]), how + ", share")
        add(f"b_{site}_robots", len(s["robots"]), how + ", robots")
        for robot, r in s["robots"].items():
            add(f"b_{robot}_k", r["k"], f"fleet.regular_work: {robot}, successes")
            add(f"b_{robot}_n", r["n"], f"fleet.regular_work: {robot}, attempts")
    healthy = meta.get("healthy", "").split()
    if healthy:
        unknown = [r for r in healthy if r not in robot_sites]
        if unknown:
            raise StoryError(f"healthy: no such robots in the regular work: {unknown}")
        k = sum(sites[robot_sites[r]]["robots"][r]["k"] for r in healthy)
        n = sum(sites[robot_sites[r]]["robots"][r]["n"] for r in healthy)
        how = f"fleet.regular_work: robots {' '.join(healthy)}"
        add("b_healthy_k", k, how + ", successes")
        add("b_healthy_n", n, how + ", attempts")
        add("b_healthy_pct", _pct(k, n), how + ", share")
        names = healthy[:-1]
        add(
            "b_healthy_robots",
            f"{', '.join(names)} and {healthy[-1]}" if names else healthy[-1],
            "front matter: healthy",
        )
    # the site the customer's remote link serves, by its public note
    add("b_link_site", site_label(fleet["b_linked_site"]["value"]), "fleet.b_linked_site")

    # the tested repairs
    adopted = {a.strip() for a in fleet["b_round1"]["value"].split(";")}
    settings = []
    for x in fleet["experiments"]:
        site, _, fields = _repair(x["change"], robot_sites)
        if "robot" in fields:
            add("b_overload_robot", fields["robot"], "fleet.experiments: the controller service")
            add("b_overload_site", site_label(site), "fleet.experiments: the controller service")
        if x["change"].startswith("a rollback of the software update"):
            add("b_update_site", site_label(site), "fleet.experiments: the software rollback")
        if "steps" in fields:
            settings.append((int(fields["steps"]), x["change"] in adopted))
    kept = [s for s, a in settings if a]
    if settings:
        add("b_comp_settings", len(settings), "fleet.experiments: compensation settings tested")
        add(
            "b_comp_tested",
            ", ".join(str(s) for s, _ in settings[:-1]) + f" and {settings[-1][0]}",
            "fleet.experiments: compensation settings, control steps",
        )
    if len(kept) == 1:
        add("b_comp_kept", kept[0], "fleet.b_round1: the adopted compensation, control steps")
    missing = adopted - {x["change"] for x in fleet["experiments"]}
    if missing:
        raise StoryError(f"adopted changes without a tested experiment: {sorted(missing)}")

    # the probe: waits between a command and the hand's first movement, from its recorded steps
    probe = fleet["probe"]
    step_s = float(probe["step_s"])
    add("b_step_ms", round(step_s * 1000), "fleet.probe.step_s")
    add(
        "ci_level",
        f"{round(100 * math.erf(Z95 / math.sqrt(2)))}%",
        "fleetlab.film.record.Z95: the level of every Wilson interval on the page",
    )
    add("ci_min_runs", MIN_RUNS_FOR_INTERVAL, "fleetlab.film.record.MIN_RUNS_FOR_INTERVAL")
    for robot, trace in probe["robots"].items():
        found = charts.lags(trace["commanded_cm"], trace["measured_cm"])
        if len(found) != 2:
            raise StoryError(f"probe {robot}: expected a step out and a step back, found {found}")
        (_, out_lag), (_, back_lag) = found
        how = f"fleet.probe: {robot}, recorded steps, first 1 mm of movement"
        add(f"b_probe_{robot}_out_steps", out_lag, how)
        add(f"b_probe_{robot}_back_steps", back_lag, how)
        add(f"b_probe_{robot}_out_s", _seconds(out_lag, step_s), how + ", seconds")
        add(f"b_probe_{robot}_back_s", _seconds(back_lag, step_s), how + ", seconds")

    # the evaluation, paired
    ev = fleet["evaluation"]
    pairs = ev["pairs"]
    for name, value in (
        ("b_eval_original_k", ev["before"]),
        ("b_eval_final_k", ev["after"]),
        ("b_eval_final_n", ev["attempts"]),
        ("b_eval_rescued", pairs["rescued"]),
        ("b_eval_broken", pairs["broken"]),
    ):
        if name in out and out[name].value != str(value):
            raise StoryError(f"{name}: the bound {out[name].value} against {value} in the table")
    add("b_eval_both", ev["before"] - pairs["broken"], "fleet.evaluation: before minus broken")
    add(
        "b_eval_neither",
        ev["attempts"] - ev["after"] - pairs["broken"],
        "fleet.evaluation: attempts minus after minus broken",
    )
    return out


# ---- filling ---------------------------------------------------------------------------------


def fill(text: str, values: dict[str, Value], used: dict[str, Value]) -> str:
    """Every `{name}` replaced by its value; an unknown name fails."""
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(text)} - set(values))
    if unknown:
        raise StoryError("unknown placeholders: " + ", ".join("{" + u + "}" for u in unknown))

    def sub(match: re.Match[str]) -> str:
        used[match.group(1)] = values[match.group(1)]
        return values[match.group(1)].value

    return PLACEHOLDER.sub(sub, text)


# ---- charts ----------------------------------------------------------------------------------


def _outcomes(bits: str) -> list[bool]:
    if set(bits) - {"0", "1"}:
        raise StoryError(f"not an outcome string: {bits!r}")
    return [b == "1" for b in bits]


def _exam_rows(data: dict[str, Any], prefix: str) -> tuple[list[charts.DotRow], dict[str, int]]:
    clips, mosaics = data["clips"], data["mosaics"]
    rows = []
    for kind, label, role in (("chain", "Plain chain", ""), ("job", "Agent's job", "win")):
        ids = mosaics.get(f"{prefix}-{kind}")
        if not ids:
            raise StoryError(f"film data: no exam tiles {prefix}-{kind}")
        tiles = [clips[i] for i in ids]
        rows.append(
            charts.DotRow(
                label,
                [bool(t["delivered"]) for t in tiles],
                [int(t["start"]) for t in tiles],
                role,
            )
        )
    chain, job = rows
    if chain.starts != job.starts:
        raise StoryError(f"{prefix}: the chain and the job ran on different layouts")
    counts = {
        "naive_delivered": sum(chain.outcomes),
        "delivered": sum(job.outcomes),
        "release_n": len(job.outcomes),
        "rescued": sum(1 for a, b in zip(chain.outcomes, job.outcomes, strict=True) if b and not a),
        "broken": sum(1 for a, b in zip(chain.outcomes, job.outcomes, strict=True) if a and not b),
    }
    return rows, counts


def _words(meta: dict[str, str], key: str, n: int) -> list[str]:
    """A front matter value of n parts separated by `|`."""
    words = [w.strip() for w in meta.get(key, "").split("|")]
    if len(words) != n or not all(words):
        raise StoryError(f"front matter {key}: {n} parts separated by |")
    return words


def _chart(kind: str, args: list[str], data: dict[str, Any], values: dict[str, Value],
           meta: dict[str, str], used: dict[str, Value]) -> charts.Chart:  # fmt: skip
    def num(name: str) -> str:
        used[name] = values[name]
        return values[name].value

    if kind == "exam":
        jobs = []
        for prefix in args:
            rows, counts = _exam_rows(data, prefix)
            for name, value in counts.items():
                bound = values.get(f"{prefix}_{name}")
                if bound is not None and bound.value != str(value):
                    raise StoryError(
                        f"{prefix}_{name}: bound {bound.value}, the exam tiles say {value}"
                    )
            note = (
                f"{num(prefix + '_rescued')} rescued, {num(prefix + '_broken')} broken · "
                f"the bar: at least {num(prefix + '_release_k')} of {num(prefix + '_release_n')}"
            )
            jobs.append((meta.get(f"job_{prefix}", prefix), rows, note))
        return charts.exam(jobs)
    if kind == "notebook":
        prefix = args[0]
        muted = {a.partition("=")[2] for a in args[1:] if a.startswith("mute=")}
        programs = [
            charts.Program(
                meta.get(f"label.{prefix}.{p['job']}", p["label"]),
                _outcomes(p["outcomes"]),
                list(p["starts"]),
                p["role"],
            )
            for p in data["programs"][prefix]
        ]
        for p, raw in zip(programs, data["programs"][prefix], strict=True):
            if sum(p.outcomes) != raw["k"] or len(p.outcomes) != raw["n"]:
                raise StoryError(f"{prefix} {raw['job']}: outcomes disagree with k of n")
        tried = sum(1 for p in programs if p.role != "confirm")
        summary = f"All {tried} programs, in the order the agent ran them" if muted else ""
        return charts.notebook(programs, muted=muted, summary=summary)
    if kind == "fleet":
        sites = data["fleet"]["regular_work"]["sites"]
        robots = [
            charts.Robot(site_label(site), robot, r["k"], r["n"])
            for site, s in sites.items()
            for robot, r in s["robots"].items()
        ]
        ref = (
            f"healthy robots {num('b_healthy_robots')}",
            int(num("b_healthy_k")),
            int(num("b_healthy_n")),
        )
        return charts.fleet_bars(robots, ref)
    if kind == "probe":
        probe = data["fleet"]["probe"]
        traces = []
        for arg in args:
            robot, _, tag = arg.partition("=")
            trace = probe["robots"].get(robot)
            if trace is None:
                raise StoryError(f"probe: no recorded test motion for {robot}")
            traces.append(
                charts.ProbeTrace(
                    f"{site_label(trace['site'])} · {robot}",
                    tag,
                    [float(v) for v in trace["commanded_cm"]],
                    [float(v) for v in trace["measured_cm"]],
                )
            )
        return charts.probe(
            traces,
            float(probe["step_s"]),
            axis_s=[float(a) for a in probe.get("axis_s") or []],
            origin_cm=float(probe.get("origin_cm") or 0.0),
        )
    if kind == "repairs":
        fleet = data["fleet"]
        robot_sites = _robot_sites(data)
        adopted = {a.strip() for a in fleet["b_round1"]["value"].split(";")}
        repair_rows: list[tuple[str, charts.Repair]] = []
        for x in fleet["experiments"]:
            site, words, _ = _repair(x["change"], robot_sites)
            repair_rows.append(
                (
                    site,
                    charts.Repair(
                        site_label(site),
                        words,
                        (x["without"]["k"], x["without"]["n"]),
                        (x["with"]["k"], x["with"]["n"]),
                        x["rescued"],
                        x["broken"],
                        "adopted" if x["change"] in adopted else "not adopted",
                    ),
                )
            )
        repair_rows.sort(key=lambda r: int(r[0][1:]))  # by site, in test order within a site
        return charts.repairs([r for _, r in repair_rows])
    if kind == "glance":
        # the results at a glance: each job's plain chain against the agent's job on the same
        # locked layouts, and the fleet before and after; the words are the front matter's
        # glance.<arg> (kicker | title | caption, a line break written " / ")

        def agrees(name: str, value: object) -> None:
            if name in values:
                if values[name].value != str(value):
                    raise StoryError(f"{name}: bound {values[name].value}, the glance {value}")
                used[name] = values[name]

        results = []
        for arg in args:
            kicker, title, caption = _words(meta, f"glance.{arg}", 3)
            if arg == "fleet":
                ev = data["fleet"]["evaluation"]
                k0, k1, n = int(ev["before"]), int(ev["after"]), int(ev["attempts"])
                for name, shown in (
                    ("b_eval_original_k", str(k0)),
                    ("b_eval_final_k", str(k1)),
                    ("b_eval_original_n", str(n)),
                    ("b_eval_final_n", str(n)),
                    ("b_eval_original_pct", _pct(k0, n)),
                    ("b_eval_final_pct", _pct(k1, n)),
                ):
                    agrees(name, shown)
                labels = _words(meta, "glance.marks.fleet", 2)
                before, after, unit = _pct(k0, n), _pct(k1, n), ""
                marks = [charts.Marks(labels[0], k=k0, n=n), charts.Marks(labels[1], k=k1, n=n)]
            else:
                (chain, job), counts = _exam_rows(data, arg)
                for name in ("naive_delivered", "delivered", "release_n"):
                    agrees(f"{arg}_{name}", counts[name])
                labels = _words(meta, "glance.marks.jobs", 2)
                before, after = str(counts["naive_delivered"]), str(counts["delivered"])
                unit = f"of {counts['release_n']}"
                marks = [
                    charts.Marks(labels[0], chain.outcomes, chain.starts),
                    charts.Marks(labels[1], job.outcomes, job.starts),
                ]
            results.append(
                charts.Result(
                    kicker, title.split(" / "), before, after, unit, caption.split(" / "), marks
                )
            )
        legend = _words(meta, "glance.legend", 3)
        return charts.glance(results, (legend[0], legend[1], legend[2]))
    if kind == "fleet-result":
        ev = data["fleet"]["evaluation"]
        return charts.fleet_result(
            ev["before"],
            ev["after"],
            ev["attempts"],
            ev["pairs"]["rescued"],
            ev["pairs"]["broken"],
            ("As the fleet was", f"With the agent's {num('b_round1_count')} changes"),
        )
    raise StoryError(f"no chart of kind {kind!r}")


# ---- the page --------------------------------------------------------------------------------


def _inline(markdown_html: str) -> str:
    """A converted paragraph without its <p> wrapper."""
    text = markdown_html.strip()
    match = re.fullmatch(r"<p>(.*)</p>", text, flags=re.S)
    return match.group(1) if match else text


def render(text_path: Path, data_path: Path) -> Story:
    from fleetlab.film.website import markdown_to_html

    if not text_path.is_file():
        raise StoryError(f"{text_path}: no story text")
    if not data_path.is_file():
        raise StoryError(
            f"{data_path}: no film data; build it with site/explainer/build_data.py first"
        )
    data = json.loads(data_path.read_text(encoding="utf-8"))
    raw = text_path.read_text(encoding="utf-8")
    meta, body = _front_matter(raw)
    body = re.sub(r"<!--.*?-->\n?", "", body, flags=re.S)  # notes to the author stay out
    values = derive(data, meta)
    used: dict[str, Value] = {}
    body = fill(body, values, used)
    meta = {k: fill(v, values, used) for k, v in meta.items()}
    banned = sorted({m.group(0).lower() for m in BANNED.finditer(body + " ".join(meta.values()))})
    if banned:
        raise StoryError(f"the page must not say: {', '.join(banned)}")

    figures: list[str] = []
    drawn: list[dict[str, Any]] = []

    def block(match: re.Match[str]) -> str:
        kind, args, content = match.group(1), match.group(2), match.group(3)
        words = shlex.split(args)
        chart = _chart(words[0], words[1:], data, values, meta, used)
        lines = content.strip().splitlines()
        title = lines[0].strip() if lines else ""
        note = _inline(markdown_to_html("\n".join(lines[1:]))[1]) if len(lines) > 1 else ""
        n = len(figures) + 1
        ident = f"fig-{n}-{words[0]}"
        drawn.append(
            {
                "figure": ident,
                "kind": words[0],
                "title": title,
                "values": chart.values,
                "sources": _sources(words[0], words[1:], data),
            }
        )
        numbers = (
            f'<details class="numbers"><summary>The numbers</summary>'
            f'<div class="scroll">{chart.table}</div></details>'
            if chart.table
            else ""
        )
        cap = (
            f'<figcaption><span class="fig-title" id="{ident}-title">{html.escape(title)}</span>'
            + (f' <span class="fig-note">{note}</span>' if note else "")
            + "</figcaption>"
        )
        tag = "figure"
        figures.append(
            f'<{tag} class="chart chart-{words[0]}" id="{ident}" aria-labelledby="{ident}-title">'
            f"{cap}{chart.html}{numbers}</{tag}>"
            if kind == "figure"
            else f'<div class="table-figure" id="{ident}">'
            f'<p class="fig-title" id="{ident}-title">{html.escape(title)}</p>'
            + (f'<p class="fig-note">{note}</p>' if note else "")
            + f'<div class="scroll" role="region" tabindex="0" aria-labelledby="{ident}-title">'
            f"{chart.html}</div></div>"
        )
        return f"\n\nSTORYFIGURE{n:03d}\n\n"

    body = BLOCK.sub(block, body)
    leftover = re.search(r"^:::.*$", body, flags=re.M)
    if leftover:
        raise StoryError(f"an unclosed or unknown block: {leftover.group(0)!r}")

    first = re.search(r"^## ", body, flags=re.M)
    lede_md, sections_md = (body[: first.start()], body[first.start() :]) if first else (body, "")
    lede = markdown_to_html(lede_md.strip())[1]
    lede_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", lede)).strip()
    sections = markdown_to_html(sections_md)[1]
    for i, fig in enumerate(figures, 1):
        token = f"STORYFIGURE{i:03d}"
        count = sections.count(f"<p>{token}</p>")
        sections = sections.replace(f"<p>{token}</p>", fig)
        if count != 1:
            raise StoryError(f"figure {i} did not land in its own paragraph")

    toc: list[tuple[str, str]] = []
    parts = re.split(r"(?=<h2[ >])", sections)
    wrapped = []
    for part in parts:
        heading = re.match(r"<h2([^>]*)>(.*?)</h2>", part, flags=re.S)
        if not heading:
            if part.strip():
                wrapped.append(f'<section class="story-section">{part}</section>')
            continue
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', heading.group(1)))
        text = re.sub(r"<[^>]+>", "", heading.group(2)).strip()
        ident = attrs.get("id") or re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        # a heading may carry a shorter name for the contents line: {#id data-toc="Short"}
        toc.append((ident, html.unescape(attrs.get("data-toc") or text)))
        wrapped.append(
            f'<section class="story-section" id="section-{ident}" aria-labelledby="{ident}">'
            f"{part}</section>"
        )
    return Story(
        title=meta.get("title", ""),
        subtitle=meta.get("subtitle", ""),
        kicker=meta.get("kicker", ""),
        lede=lede,
        lede_text=lede_text,
        body="\n".join(wrapped),
        toc=toc,
        used=dict(sorted(used.items())),
        data_sha256=sha256(data_path),
        text_sha256=sha256(text_path),
        figures=len(figures),
        values=values,
        charts=drawn,
        data=data,
        meta=meta,
    )


# ---- sources and the fleet record ------------------------------------------------------------


def public_id(ident: str) -> str:
    """A film-data identifier (`probes:...`) under the site's configured public rules: a
    fleet's private world id inside a run directory's name becomes its public one."""
    label, _, rest = ident.partition(":")
    for root, name in public.ROOTS:
        if name == label and rest:
            return public.public_ref(root / rest)
    return ident


def _digests(data: dict[str, Any]) -> dict[str, str]:
    """Every identifier the film data names with a SHA-256, to that SHA-256."""
    out: dict[str, str] = {}
    for entry in data.get("fleet", {}).values():
        if isinstance(entry, dict):
            for i in entry.get("inputs") or []:
                out[str(i["id"])] = str(i.get("sha256", ""))
    for i in data.get("provenance", {}).get("inputs") or []:
        out.setdefault(str(i["path"]), str(i.get("sha256", "")))
    return out


def _inputs(entry: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"id": str(i["id"]), "sha256": str(i.get("sha256", ""))} for i in entry.get("inputs") or []
    ]


def _sources(kind: str, args: list[str], data: dict[str, Any]) -> list[dict[str, str]]:
    """The records a chart's numbers were read from, by identifier (and SHA-256 where the film
    data names it)."""
    fleet, sums = data.get("fleet", {}), _digests(data)

    def ident(i: str) -> dict[str, str]:
        return {"id": i, "sha256": sums.get(i, "")}

    if kind == "notebook":
        return [
            ident(i) for i in dict.fromkeys(str(p["source"]) for p in data["programs"][args[0]])
        ]
    if kind == "exam":
        out = []
        for prefix in args:
            for which in ("chain", "job"):
                for c in data["mosaics"][f"{prefix}-{which}"]:
                    tile = data["clips"][c]
                    out.append({"id": str(tile["run"]), "episode": str(tile["episode"])})
        return out
    if kind == "fleet":
        return [ident(str(fleet["regular_work"]["source"]))]
    if kind == "probe":
        return [ident(str(fleet["probe"]["source"])), *_inputs(fleet["b_probe"])]
    if kind == "repairs":
        return _inputs(fleet["b_experiments_table"]) + _inputs(fleet["b_round1"])
    if kind == "fleet-result":
        return _inputs(fleet["b_round1"])  # the engagement record holds the evaluation
    if kind == "glance":
        jobs = [a for a in args if a != "fleet"]
        fleet_part = _sources("fleet-result", [], data) if "fleet" in args else []
        return _sources("exam", jobs, data) + fleet_part
    return []


@dataclass
class FleetRecord:
    title: str
    body: str  # HTML
    tables: dict[str, list[dict[str, Any]]]


def _count(k: int, n: int) -> tuple[str, dict[str, Any]]:
    """A cell: k of n, and under it the share with its Wilson interval from the record's code
    (from MIN_RUNS_FOR_INTERVAL runs on)."""
    p = proportion(k, n)
    under = p["pct"] + (f" ({p['low_pct']} to {p['high_pct']})" if "low_pct" in p else "")
    cell = f'{k} of {n}<br><span class="ci">{html.escape(under)}</span>'
    return cell, {k2: p[k2] for k2 in ("k", "n", "pct", "low_pct", "high_pct") if k2 in p}


def _source_line(items: list[dict[str, str]], what: str = "Source") -> str:
    shown = []
    for i in items:
        digest = i.get("sha256", "")
        shown.append(
            f"<code>{html.escape(public_id(i['id']))}</code>"
            + (f" (SHA-256 <code>{digest[:12]}</code>)" if digest else "")
        )
    if len(shown) <= 2:
        return f'<p class="source">{what}: {"; ".join(shown)}.</p>'
    return (
        f'<details class="source"><summary>{what}: {len(shown)} records</summary><ul>'
        + "".join(f"<li>{x}</li>" for x in shown)
        + "</ul></details>"
    )


def fleet_record(data: dict[str, Any], meta_healthy: list[str]) -> FleetRecord:
    """Experiment two, counted: the regular work robot by robot, the healthy reference, the
    probe's waits as recorded, the tested repairs and the evaluation, each count with its
    share and Wilson interval, each table with the records it was read from. Every number
    comes from the film data; the intervals come from fleetlab.film.record and are checked
    against the ones the record itself states."""
    fleet = data["fleet"]
    sums = _digests(data)
    level = f"{round(100 * math.erf(Z95 / math.sqrt(2)))}%"
    tables: dict[str, list[dict[str, Any]]] = {}
    out = [
        "<h1>The fleet record</h1>",
        f'<p class="record-lede">Every count behind experiment two: the fleet of '
        f"{sum(len(s['robots']) for s in fleet['regular_work']['sites'].values())} simulated "
        f"arms at {len(fleet['regular_work']['sites'])} sites, the planted problems hidden from "
        f"the agent. Each count gives its share and, where it rests on "
        f"{MIN_RUNS_FOR_INTERVAL} or more attempts, its {level} Wilson interval in brackets. "
        "Each table names the records it was read from by a stable identifier and SHA-256; "
        "the files stay with the run records.</p>",
    ]

    # the regular work
    sites = fleet["regular_work"]["sites"]
    rows, vals = [], []
    for site, sv in sites.items():
        for robot, r in sv["robots"].items():
            cell, v = _count(r["k"], r["n"])
            rows.append(
                f"<tr><td>{site_label(site).capitalize()}</td><td>{robot}</td><td>{cell}</td></tr>"
            )
            vals.append({"site": site_label(site), "robot": robot, **v})
        cell, v = _count(sv["k"], sv["n"])
        rows.append(
            f'<tr class="total"><td>{site_label(site).capitalize()}</td><td>all robots</td>'
            f"<td>{cell}</td></tr>"
        )
        vals.append({"site": site_label(site), "robot": "all", **v})
    robot_sites = _robot_sites(data)
    if meta_healthy:
        k = sum(sites[robot_sites[r]]["robots"][r]["k"] for r in meta_healthy)
        n = sum(sites[robot_sites[r]]["robots"][r]["n"] for r in meta_healthy)
        cell, v = _count(k, n)
        names = ", ".join(meta_healthy)
        rows.append(
            f'<tr class="total"><td>Healthy reference</td><td>{names}</td><td>{cell}</td></tr>'
        )
        vals.append({"site": "healthy reference", "robot": names, **v})
    # the record states the noted site and the others with intervals: they must agree
    linked = fleet["b_linked_site"]["value"]
    for key, k, n in (
        ("b_delayed_site", sites[linked]["k"], sites[linked]["n"]),
        (
            "b_other_sites",
            sum(s["k"] for x, s in sites.items() if x != linked),
            sum(s["n"] for x, s in sites.items() if x != linked),
        ),
    ):
        stated = fleet.get(key) or {}
        if stated and (stated.get("k"), stated.get("n")) != (k, n):
            raise StoryError(f"{key}: the record says {stated.get('k')} of {stated.get('n')}")
        if stated and stated.get("interval") and stated["interval"] != proportion(k, n)["interval"]:
            raise StoryError(f"{key}: the record's interval differs from Wilson's")
    tables["regular_work"] = vals
    out += [
        "<h2>The regular work, robot by robot</h2>",
        "<p>Successes in the fleet's regular work before the agent changed anything. The "
        "healthy reference is the robots no planted problem touches.</p>",
        '<table class="record-table"><thead><tr><th>Site</th><th>Robot</th>'
        f"<th>Successes, share ({level} interval)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
        _source_line([{"id": str(fleet["regular_work"]["source"]),
                       "sha256": sums.get(str(fleet["regular_work"]["source"]), "")}]),
    ]  # fmt: skip

    # the probe
    probe, bp = fleet["probe"], fleet["b_probe"]
    step_ms = round(float(probe["step_s"]) * 1000)
    rows, vals = [], []
    for robot, r in bp.items():
        if not isinstance(r, dict) or "first_move" not in r:
            continue
        rows.append(
            f"<tr><td>{robot}</td><td>{site_label(r['site']).capitalize()}</td>"
            f"<td>{r['repeats']}</td>"
            f"<td>{html.escape(str(r['first_move']))}</td>"
            f"<td>{html.escape(str(r['half_way']))}</td></tr>"
        )
        vals.append(
            {"robot": robot, "site": site_label(r["site"]), "repeats": r["repeats"],
             "first_move_steps": r["first_move"], "half_way_steps": r["half_way"]}
        )  # fmt: skip
    tables["probe"] = vals
    out += [
        "<h2>The test motion: waits as recorded</h2>",
        "<p>The agent's test motion with the robot's model switched off: a commanded step out "
        "and back, repeated. For each robot it tested, the control steps between each command "
        f"and the hand's first movement ({PROBE_MOVE_M * 1000:g} mm of travel), and until it was "
        "half-way, as read "
        f"from the recorded steps. One control step is {step_ms} milliseconds.</p>",
        '<table class="record-table"><thead><tr><th>Robot</th><th>Site</th><th>Repeats</th>'
        "<th>First movement after (control steps)</th><th>Half-way after (control steps)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        _source_line(_inputs(bp), "Sources"),
    ]

    # the tested repairs, checked against the record's own table
    stated_rows: dict[tuple[str, str], tuple[str, str]] = {}
    for line in str(fleet["b_experiments_table"]["value"]).splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6:
            stated_rows[(cells[0], cells[1])] = (cells[2], cells[3])
    adopted = {a.strip() for a in fleet["b_round1"]["value"].split(";")}
    rows, vals = [], []
    for x in fleet["experiments"]:
        site, words, _ = _repair(x["change"], robot_sites)
        counts: list[tuple[str, dict[str, Any]]] = []
        for side in ("without", "with"):
            k, n = x[side]["k"], x[side]["n"]
            stated = stated_rows.get((x["experiment"], x["change"]), ("", ""))[side == "with"]
            if stated and stated != f"{k} of {n}{proportion(k, n)['interval']}":
                raise StoryError(f"{x['experiment']}: {stated!r} is not Wilson's interval")
            counts.append(_count(k, n))
        decision = "adopted" if x["change"] in adopted else "not adopted"
        rows.append(
            f"<tr><td>{html.escape(x['experiment'])}</td><td>{site_label(site).capitalize()}</td>"
            f'<td class="change">{html.escape(words)}</td>'
            f"<td>{counts[0][0]}</td><td>{counts[1][0]}</td>"
            f"<td>{x['rescued']}</td><td>{x['broken']}</td><td>{decision}</td></tr>"
        )
        vals.append(
            {"experiment": x["experiment"], "site": site_label(site), "change": words,
             "without": counts[0][1], "with": counts[1][1], "rescued": x["rescued"],
             "broken": x["broken"], "decision": decision}
        )  # fmt: skip
    tables["repairs"] = vals
    out += [
        "<h2>The tested repairs</h2>",
        "<p>Each change the agent tested, against the same attempts without it. Rescued: "
        "failed without the change and succeeded with it; broken: the reverse. Adopted: the "
        "changes the agent kept.</p>",
        '<table class="record-table"><thead><tr><th>Experiment</th><th>Site</th>'
        "<th>Change tested</th><th>Without</th><th>With</th><th>Rescued</th><th>Broken</th>"
        "<th>Decision</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        _source_line(_inputs(fleet["b_experiments_table"]) + _inputs(fleet["b_round1"]),
                     "Sources"),
    ]  # fmt: skip

    # the evaluation
    ev = fleet["evaluation"]
    numbers = data.get("numbers", {})
    rows, vals = [], []
    for label, k, key in (
        ("As the fleet was", ev["before"], "b_eval_original"),
        (f"With the agent's {len(adopted)} changes", ev["after"], "b_eval_final"),
    ):
        cell, v = _count(k, ev["attempts"])
        for bound, mine in (("low", "low_pct"), ("high", "high_pct")):
            if numbers.get(f"{key}_{bound}") not in (None, v.get(mine)):
                raise StoryError(f"{key}_{bound}: bound {numbers[f'{key}_{bound}']}, not Wilson's")
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{cell}</td></tr>")
        vals.append({"state": label, **v})
    pairs = ev["pairs"]
    rows.append(
        f'<tr class="total"><td>The same attempts, compared</td><td>{pairs["rescued"]} rescued, '
        f"{pairs['broken']} broken</td></tr>"
    )
    vals.append({"state": "paired", "rescued": pairs["rescued"], "broken": pairs["broken"]})
    tables["evaluation"] = vals
    out += [
        "<h2>The evaluation on attempts the agent never saw</h2>",
        f"<p>The reserved panel of {ev['attempts']} attempts, run once with the fleet as it "
        "was and once with the agent's adopted changes.</p>",
        '<table class="record-table"><thead><tr><th>Fleet</th>'
        f"<th>Successes, share ({level} interval)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
        _source_line(_inputs(fleet["b_round1"])),
    ]
    return FleetRecord("The fleet record", "\n".join(out), tables)
