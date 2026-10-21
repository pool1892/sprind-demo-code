"""Build the explainer film's data: every number, quotation, chart and clip it shows.

Writes `film/public/data/film-data.json` and `film/public/data/timeline.json`, and links the
footage and narration it names into `film/public/`, so the Remotion composition reads only
from there. Nothing on screen is typed by hand: numbers and quotations come from the bound
script (`derived.json`, `bindings.json`), the story B record's derived values, and the run
tables; charts and dot rows are read from the recorded episodes. Every input file is listed
in the output's `provenance` with its SHA-256.

    uv run --locked python site/explainer/build_data.py \
        --bound ~/.cache/robots-probes/demo/explainer/script-v2 \
        [--voice ~/.cache/robots-probes/demo/explainer/voice/<take>] \
        [--footage ~/.cache/robots-probes/demo/explainer/footage] [--music FILE]

Without `--voice` the timeline is estimated at 150 words a minute; with it, each beat lasts
its recording plus a lead-in and a tail, and cues come from the voice's character timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import duckdb

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "film" / "public"
CACHE = Path.home() / ".cache" / "robots-probes" / "demo"
REL = CACHE / "2026-10-20" / "reliability"
R1 = REL / "replacement-01"
R2 = REL / "plate-cabinet-01"
ENG = CACHE / "2026-10-20" / "instrument" / "engagement-1"
STORY_B_DERIVED = CACHE / "2026-10-20" / "site" / "final-9" / "record" / "derived.json"
TESTS = CACHE / "explainer" / "camera-test"

WPM = 150.0
LEAD_IN = 0.45  # seconds of picture before the voice starts
TAIL = 0.7  # seconds of picture after it ends
SILENT = {"title": 4.0, "end": 12.0}

_sources: dict[str, str] = {}


def sha(path: Path) -> str:
    key = str(path)
    if key not in _sources:
        _sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _sources[key]


def public_id(path: Path) -> str:
    try:
        return "probes:" + str(path.resolve().relative_to(CACHE.parent))
    except ValueError:
        return str(path)


def place(link: Path, target: Path) -> None:
    """Put `target` at `link` inside public/ as a real file (Remotion's bundler copies public/
    and drops symlinks): a hard link where the file system allows it, else a copy."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        os.link(target, link)
    except OSError:
        shutil.copyfile(target, link)


# ---- numbers and quotations ------------------------------------------------------------------


QUOTES = [
    "r1_request",
    "r2_request",
    "r1_switch",
    "r1_diagnosis",
    "r1_said_chain",
    "r1_said_home",
    "r1_said_open",
    "r1_said_finish",
    "r1_said_unisolated",
    "r1_said_ready",
    "r1_skill_description",
    "r1_skill_heights",
    "r1_check_purpose",
    "r2_said_plain",
    "r2_said_wrong_object",
    "r2_said_home",
    "r2_said_continuation",
    "r2_said_loaded",
    "r2_skill_evidence",
    "r2_said_grasp",
    "r2_said_deliberate",
    "r2_said_flicker",
    "r2_said_unisolated",
    "r2_said_library",
    "r2_said_ready",
    "r2_skill_description",
    "r2_check_purpose",
    "r1_proposal",
    "r2_proposal",
]


def all_fields(out: Path) -> dict[str, str]:
    """Every field of fields.json bound through the record binder (the same derivation and
    records as the script), so the screen can show any of them: a page that names them all."""
    from fleetlab.film.record import GENERATED, bind_record

    fields = json.loads((HERE / "fields.json").read_text())["fields"]
    names = [n for n in fields if n not in GENERATED]
    page = out / "all-fields.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, str] = {}
    pending = names
    for _ in range(4):  # drop any field whose value the binder refuses to show, then retry
        page.write_text("# all fields\n\n" + "\n".join(f"- {n}: {{{n}}}" for n in pending) + "\n")
        try:
            bind_record(page, HERE / "fields.json", HERE / "records.json", out / "all-fields")
        except Exception as error:  # a forbidden value: find and drop it
            shown = re.search(r"shows '(.+?)'", str(error))
            bad = [n for n in pending if shown and shown.group(1) in n]
            print(f"all-fields: {error}; dropping {bad or pending[-1:]}")
            pending = [n for n in pending if n not in (bad or pending[-1:])]
            continue
        bindings = json.loads((out / "all-fields" / "bindings.json").read_text())
        for name, item in (bindings.get("values") or {}).items():
            if isinstance(item, dict) and "value" in item:
                values[name] = str(item["value"])
        break
    return values


def numbers_and_quotes(bound: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Every bound placeholder of the script, and every field of the record's fields file
    (counts, names, quotations) the screen may show, from the same derivation."""
    bindings = json.loads((bound / "bindings.json").read_text())
    derived = json.loads((bound / "derived.json").read_text())
    sha(bound / "bindings.json")
    sha(bound / "derived.json")
    numbers: dict[str, str] = {}
    for name, item in (bindings.get("values") or {}).items():
        if isinstance(item, dict) and "value" in item:
            numbers[name] = str(item["value"])
    values = derived.get("values", {})
    fields = json.loads((HERE / "fields.json").read_text())["fields"]
    for name, spec in fields.items():
        if name in numbers or spec.get("record") != "derived":
            continue
        top, _, key = str(spec["field"]).partition(".")
        entry = values.get(top)
        if isinstance(entry, dict) and key in entry and not isinstance(entry[key], dict | list):
            numbers[name] = str(entry[key])
    for name, value in all_fields(CACHE / "explainer" / "data-build").items():
        numbers.setdefault(name, value)
    quotes = {q: numbers[q] for q in QUOTES if q in numbers}
    return numbers, quotes


def story_b(path: Path) -> dict[str, Any]:
    """Story B's derived values from the 20 October record (each with its own inputs)."""
    values = json.loads(path.read_text())["values"]
    sha(path)
    keep = [
        "b_probe",
        "b_delayed_site",
        "b_other_sites",
        "b_linked_site",
        "b_lag_lesson",
        "b_probe_reading",
        "b_round1",
        "b_hours",
        "b_experiments_table",
    ]
    out = {k: values.get(k) for k in keep}
    rows = []
    for line in str(values["b_experiments_table"]["value"]).splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 6:
            continue
        exp, change, without, with_, rescued, broken = cells
        m0 = re.match(r"(\d+) of (\d+)", without)
        m1 = re.match(r"(\d+) of (\d+)", with_)
        rows.append(
            {
                "experiment": exp,
                "change": change,
                "without": {"k": int(m0[1]), "n": int(m0[2])} if m0 else None,
                "with": {"k": int(m1[1]), "n": int(m1[2])} if m1 else None,
                "rescued": int(rescued),
                "broken": int(broken),
            }
        )
    out["experiments"] = rows
    return out


# ---- programs on practice starts (the dot rows) ------------------------------------------------

R1_LABELS = {
    "j0001": ("straight to the drawer (the plain chain)", "plain"),
    "j0002": ("placement runs on, then the drawer", "other"),
    "j0003": ("open the gripper first", "other"),
    "j0004": ("go home first", "other"),
    "j0005": ("open, then go home", "other"),
    "j0006": ("let the placement finish, then go home", "winner"),
    "j0007": ("drawer first, then the bowl", "other"),
    "j0008": ("the same program, layouts never used", "confirm"),
}
R2_LABELS = {
    "j0001": ("plate, then bowl (the plain chain)", "plain"),
    "j0002": ("bowl, then plate", "other"),
    "j0003": ("plate, then bowl, re-plan every {actions_per_chunk} actions", "other"),
    "j0004": ("plate, home, bowl", "other"),
    "j0005": ("bowl, home, plate", "other"),
    "j0006": ("bowl with job 1's trick, home, plate", "other"),
    "j0007": ("plate twice, home, bowl", "other"),
    "j0008": ("plate, home, plate again, home, bowl", "other"),
    "j0009": ("plate, home, bowl, re-plan every {actions_per_chunk}", "other"),
    "j0010": ("bowl with the trick, home, plate, home", "other"),
    "j0011": ("plate push cut short, retry", "other"),
    "j0012": ("push cut short, slide, retry", "other"),
    "j0013": ("cheese loaded into the bowl", "loaded"),
    "j0014": ("cheese loaded, bowl first", "other"),
    "j0015": ("cheese grabbed and set aside", "winner"),
    "j0016": ("set aside, with a lift first", "other"),
    "j0017": ("loaded, bowl named in the instruction", "other"),
    "j0018": ("loaded, rim grasp asked for", "other"),
    "j0019": ("the same program, layouts never used", "confirm"),
}


def programs(run: Path, labels: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    names: dict[str, str] = {}
    record = json.loads((run / "job_run.json").read_text())
    sha(run / "job_run.json")
    for batch in record.get("batches", []):
        for job in batch.get("jobs", []):
            prog = job.get("program")
            names[str(job.get("job_id"))] = str(
                prog.get("name", "") if isinstance(prog, dict) else prog or ""
            )
    recorded: dict[
        str, dict[str, Any]
    ] = {}  # job id -> its recorded program (run.json of its session)
    for rj in sorted((run / "runs").glob("session-*/run/run.json")):
        for job in json.loads(rj.read_text()).get("jobs", []):
            if isinstance(job, dict) and isinstance(job.get("program"), dict):
                recorded[str(job["job_id"])] = {**job["program"], "_source": rj}
    for sess in sorted((run / "runs").glob("session-*/run/episodes.jsonl")):
        sha(sess)
        rows = [json.loads(line) for line in sess.read_text().splitlines() if line.strip()]
        by: dict[str, list[tuple[int, bool]]] = {}
        for r in rows:
            by.setdefault(r["job_id"], []).append(
                (int(r["start"]["init_state"]), bool(r["delivered"]))
            )
        for job, items in sorted(by.items()):
            items.sort()
            label, role = labels.get(job, (job, "other"))
            if (
                "{" in label
            ):  # a parameter of the recorded program, read from its session's run.json
                prog = recorded[job]
                sha(prog["_source"])
                label = label.format(**{k: v for k, v in prog.items() if not k.startswith("_")})
            out.append(
                {
                    "job": job,
                    "label": label,
                    "name": names.get(job, ""),
                    "role": role,
                    "outcomes": "".join("1" if d else "0" for _, d in items),
                    "starts": [s for s, _ in items],
                    "k": sum(d for _, d in items),
                    "n": len(items),
                    "source": public_id(sess),
                }
            )
    return out


# ---- charts from recorded tables --------------------------------------------------------------


def intervals(flags: list[bool]) -> list[list[int]]:
    out, start = [], None
    for i, v in enumerate([*flags, False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append([start, i])
            start = None
    return out


def job_trace(run_dir: Path, episode: str, obj: str, goal: int) -> dict[str, Any]:
    con = duckdb.connect()
    log = run_dir / "log"
    o = con.execute(
        "select step, position[3], in_contact_with_robot "
        f"from read_parquet('{log}/object_steps/{episode}.parquet') "
        f"where object = '{obj}' order by step"
    ).fetchall()
    g = con.execute(
        f"select step, holds from read_parquet('{log}/goal_steps/{episode}.parquet') "
        f"where goal = {goal} order by step"
    ).fetchall()
    ev = con.execute(
        "select program_step, kind, start_step, end_step, outcome, instruction "
        f"from read_parquet('{log}/job_step_events.parquet') "
        f"where episode_id = '{episode}' order by start_step"
    ).fetchall()
    for f in (
        f"{log}/object_steps/{episode}.parquet",
        f"{log}/goal_steps/{episode}.parquet",
        f"{log}/job_step_events.parquet",
    ):
        sha(Path(f))
    return {
        "episode": episode,
        "run": public_id(run_dir),
        "object": obj,
        "height_m": [round(float(z), 4) for _, z, _ in o],
        "touch": intervals([bool(t) for *_, t in o]),
        "goal": intervals([bool(h) for _, h in g]),
        "steps": len(o),
        "events": [
            {
                "program_step": e[0],
                "kind": e[1],
                "from": e[2],
                "to": e[3],
                "outcome": e[4],
                "instruction": e[5],
            }
            for e in ev
        ],
    }


def probe_traces() -> dict[str, Any]:
    con = duckdb.connect()
    base = ENG / "loop" / "rounds" / "r2" / "exports" / "start-02" / "log"
    eps = con.execute(
        "select episode_id, robot_id, site_id, repeat "
        f"from read_parquet('{base}/probe_episodes.parquet') order by robot_id, repeat"
    ).fetchall()
    sha(base / "probe_episodes.parquet")
    out: dict[str, Any] = {
        "robots": {},
        "step_s": 0.05,
        "source": public_id(base / "probe_episodes.parquet"),
    }
    for ep, robot, site, rep in eps:
        if rep != 0:
            continue
        path = base / "probe_steps" / f"{ep}.parquet"
        sha(path)
        rows = con.execute(
            f"select step, target_pos, eef_pos from read_parquet('{path}') order by step"
        ).fetchall()
        last_t = None
        targets, measured = [], []
        for _, t, e in rows:
            last_t = list(t) if t is not None else last_t
            targets.append(last_t)
            measured.append(list(e))
        t0 = next(t for t in targets if t is not None)
        axis = max(range(3), key=lambda i: max(abs((t or t0)[i] - t0[i]) for t in targets))
        e0 = measured[0][axis]
        out["robots"][robot] = {
            "site": site,
            "episode": ep,
            "axis": "xyz"[axis],
            "commanded_cm": [round(((t or t0)[axis] - t0[axis]) * 100, 3) for t in targets],
            "measured_cm": [round((m[axis] - e0) * 100, 3) for m in measured],
        }
    # The diagram's references, from the traces rather than typed in the scene: the commanded
    # position the step starts from, and a tick every 3 s over the longest recorded trace.
    traces = list(out["robots"].values())
    out["origin_cm"] = min(min(r["commanded_cm"]) for r in traces)
    longest = max(len(r["commanded_cm"]) for r in traces) * out["step_s"]
    out["axis_s"] = [3 * k for k in range(int(longest // 3) + 1)]
    return out


def regular_work() -> dict[str, Any]:
    con = duckdb.connect()
    p = ENG / "loop" / "rounds" / "r1" / "exports" / "start-01" / "log" / "episodes.parquet"
    sha(p)
    rows = con.execute(
        "select site_id, robot_id, day, count(*), "
        "sum(case when outcome = 'success' then 1 else 0 end) "
        f"from read_parquet('{p}') group by 1, 2, 3 order by 1, 2, 3"
    ).fetchall()
    sites: dict[str, dict[str, Any]] = {}
    for site, robot, _day, n, k in rows:
        s = sites.setdefault(site, {"robots": {}, "k": 0, "n": 0})
        r = s["robots"].setdefault(robot, {"k": 0, "n": 0})
        r["k"] += int(k)
        r["n"] += int(n)
        s["k"] += int(k)
        s["n"] += int(n)
    return {"sites": sites, "source": public_id(p)}


def fleet_gains() -> dict[str, Any]:
    """The agent's forecast of the gain (points) and the reserved panel's measured gain."""
    eng = json.loads((ENG / "engagement" / "engagement.json").read_text())
    sha(ENG / "engagement" / "engagement.json")
    states = {s["name"]: s for s in eng["evaluation"]["summary"]["states"]}
    before, after = states["original"], states["round-2"]
    measured = 100.0 * (
        after["successes"] / after["attempts"] - before["successes"] / before["attempts"]
    )
    return {
        "forecast_gain_pts": eng["rounds"][0]["decision"]["predicted_set_gain"],
        "measured_gain_pts": round(measured, 1),
        "evaluation": {
            "before": before["successes"],
            "after": after["successes"],
            "attempts": after["attempts"],
            "pairs": eng["evaluation"]["summary"]["pairs"][0],
        },
    }


FLEET_FILES = CACHE / "2026-10-20" / "instrument" / "worlds" / "STORYB" / "files"
CHANGE_WORDS = {
    "software.version": "software update",
    "camera.scene.calibration": "scene camera calibration changed",
    "camera.wrist.calibration": "wrist camera calibration changed",
}


def fleet_console() -> dict[str, Any]:
    """What the fleet's operator console shows the agent: the action catalogue with its
    operator-hour costs and the change log, in plain words (the fleet's public description)."""
    path = FLEET_FILES / "description.json"
    desc = json.loads(path.read_text())
    sha(path)
    catalogue = [
        {"verb": c["verb"], "target": (c["target"] or "").replace("_", " "), "hours": c["hours"]}
        for c in desc["costs"]
        if c["hours"] is not None and c["target"] not in ("config_key", "condition_key", None)
    ]
    for c in desc["costs"]:
        if c["target"] == "config_key":
            catalogue.append({"verb": c["verb"], "target": "a setting", "hours": c["hours"]})
    changes = [
        {
            "day": e["day"],
            "scope": e["scope"],
            "change": CHANGE_WORDS.get(e["key"], "setting changed"),
        }
        for e in desc["fleet"]["change_log"]
    ]
    return {
        "catalogue": catalogue,
        "change_log": changes,
        "maintenance_entries": len(desc["fleet"]["maintenance_log"]),
        "source": public_id(path),
    }


# ---- clips ------------------------------------------------------------------------------------


def probe_video(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "frames": int(s["nb_read_packets"]),
    }


def clip_entry(
    cid: str,
    path: Path,
    *,
    run: Path,
    episode: str,
    label: str,
    verified: bool,
    frame_every: int,
    track: Path | None = None,
    delivered: bool | None = None,
    start: int | None = None,
) -> dict[str, Any]:
    meta = probe_video(path)
    link = PUBLIC / "footage" / f"{cid}{path.suffix}"
    place(link, path)
    entry: dict[str, Any] = {
        "id": cid,
        "src": f"footage/{link.name}",
        "fps": meta["fps"] * 1.0,
        "width": meta["width"],
        "height": meta["height"],
        "frames": meta["frames"],
        "first_step": 0,
        "frame_every": frame_every,
        "track": None,
        "run": public_id(run),
        "episode": episode,
        "verified": verified,
        "label": label,
        "file": public_id(path),
        "sha256": sha(path),
    }
    # a clip's fps is in control steps: 20 steps a second at frame_every 1, 5 frames a second at 4
    entry["fps"] = 20.0
    if track is not None and track.is_file():
        tl = PUBLIC / "footage" / f"{cid}.track.json"
        place(tl, track)
        entry["track"] = f"footage/{tl.name}"
    if delivered is not None:
        entry["delivered"] = delivered
    if start is not None:
        entry["start"] = start
    return entry


def episodes(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "episodes.jsonl"
    sha(path)
    return {
        r["episode_id"]: r
        for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
    }


def clips(footage: Path | None) -> tuple[dict[str, Any], dict[str, list[str]]]:
    out: dict[str, Any] = {}
    r1rel, r2rel = R1 / "run" / "release" / "run" / "run", R2 / "run" / "release" / "run" / "run"
    sess = lambda root, n: root / "run" / "runs" / f"session-{n:02d}" / "run"  # noqa: E731

    def add(cid, path, **kw):
        if path.is_file():
            out[cid] = clip_entry(cid, path, **kw)

    # verified re-renders made for the storyboard and the earlier cut (20 steps a second)
    add(
        "r1-chain-31-scene",
        R1 / "replays-release" / "e000202.rerender.scene.720px.mp4",
        run=r1rel,
        episode="e000202",
        label="job 1, plain chain, reserved start 31",
        verified=True,
        frame_every=1,
    )
    add(
        "r1-job-31-scene",
        R1 / "replays-release" / "e000102.rerender.scene.720px.mp4",
        run=r1rel,
        episode="e000102",
        label="job 1, the saved job, reserved start 31",
        verified=True,
        frame_every=1,
    )
    add(
        "r1-chain-31-wrist",
        R1 / "replays-release-camera-fixed" / "e000202.rerender.wrist.720px.mp4",
        run=r1rel,
        episode="e000202",
        label="job 1, plain chain, wrist camera",
        verified=True,
        frame_every=1,
    )
    add(
        "r1-chain-31-side",
        TESTS / "e000202.rerender.sideview.720px.mp4",
        run=r1rel,
        episode="e000202",
        label="job 1, plain chain, side camera",
        verified=True,
        frame_every=1,
    )
    add(
        "r1-job-31-side",
        TESTS / "r1-release" / "e000102.rerender.sideview.720px.mp4",
        run=r1rel,
        episode="e000102",
        label="job 1, the saved job, side camera",
        verified=True,
        frame_every=1,
    )
    screen = REL / "screen-01"
    add(
        "single-cabinet-side",
        TESTS / "singles" / "single-cabinet" / "e000102.rerender.sideview.720px.mp4",
        run=screen / "single-cabinet",
        episode="e000102",
        label="the cabinet skill alone",
        verified=True,
        frame_every=1,
    )
    add(
        "single-drawer-side",
        TESTS / "singles" / "single-drawer" / "e000102.rerender.sideview.720px.mp4",
        run=screen / "single-drawer",
        episode="e000102",
        label="the drawer skill alone",
        verified=True,
        frame_every=1,
    )
    add(
        "r2-chain-31-scene",
        R2 / "replays-release" / "e000202.rerender.scene.720px.mp4",
        run=r2rel,
        episode="e000202",
        label="job 2, plain chain, reserved start 31",
        verified=True,
        frame_every=1,
    )
    add(
        "r2-job-31-scene",
        R2 / "replays-release" / "e000102.rerender.scene.720px.mp4",
        run=r2rel,
        episode="e000102",
        label="job 2, the saved job, reserved start 31",
        verified=True,
        frame_every=1,
    )
    add(
        "r2-loaded-0-front",
        TESTS / "r2-s4" / "e001301.rerender.frontview.720px.mp4",
        run=sess(R2, 4),
        episode="e001301",
        label="job 2, cheese loaded, practice start 0",
        verified=True,
        frame_every=1,
    )
    add(
        "r2-relocate-0-front",
        TESTS / "r2-s5" / "e001501.rerender.frontview.720px.mp4",
        run=sess(R2, 5),
        episode="e001501",
        label="job 2, cheese set aside, practice start 0",
        verified=True,
        frame_every=1,
    )
    add(
        "r2-chain-dev1-top",
        TESTS / "r2-s1" / "e000102.rerender.birdview.720px.mp4",
        run=sess(R2, 1),
        episode="e000102",
        label="job 2, plain chain, practice start 1, top camera",
        verified=True,
        frame_every=1,
    )
    # original recordings (360 px, every fourth control step)
    for cid, root, n, ep, label in [
        ("r1-chain-practice2", R1, 1, "e000103", "job 1, plain chain, practice start 2"),
        ("r1-openhome-2", R1, 3, "e000503", "job 1, open then home, practice start 2"),
        ("r1-finishhome-2", R1, 3, "e000603", "job 1, finish then home, practice start 2"),
        ("r2-chain-dev1", R2, 1, "e000102", "job 2, plain chain, practice start 1"),
        ("r2-chain-dev5", R2, 1, "e000106", "job 2, plain chain, practice start 5"),
        ("r2-loaded-0", R2, 4, "e001301", "job 2, cheese loaded, practice start 0"),
        ("r2-relocate-0", R2, 5, "e001501", "job 2, cheese set aside, practice start 0"),
    ]:
        run = sess(root, n)
        add(
            cid,
            run / "log" / "video" / f"{ep}.scene.mp4",
            run=run,
            episode=ep,
            label=label + " (original recording)",
            verified=True,
            frame_every=4,
        )
    # the wrist camera of the agent's own first runs, as the agent could read its frames
    add(
        "r1-chain-practice2-wrist",
        sess(R1, 1) / "log" / "video" / "e000103.wrist.mp4",
        run=sess(R1, 1),
        episode="e000103",
        label="job 1, plain chain, practice start 2, wrist camera (original recording)",
        verified=True,
        frame_every=4,
    )
    # the final exam's tiles: every reserved run of both jobs, the job and the plain chain
    mosaics: dict[str, list[str]] = {}
    tiles: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    mosaic_index = footage / "mosaic" / "index.json" if footage else None
    if mosaic_index and mosaic_index.is_file():
        sha(mosaic_index)
        for e in json.loads(mosaic_index.read_text()).get("entries", []):
            if e.get("accepted") and e.get("status") == "rendered":
                tiles[(str(Path(e["run_dir"]).resolve()), e["episode_id"])] = (
                    mosaic_index.parent / e["video"],
                    e,
                )
    for job, run in [("r1", r1rel), ("r2", r2rel)]:
        eps = episodes(run)
        for jid, kind in [("j0001", "job"), ("j0002", "chain")]:
            ids = []
            for ep, row in sorted(eps.items(), key=lambda kv: kv[1]["start"]["init_state"]):
                if row["job_id"] != jid:
                    continue
                init = row["start"]["init_state"]
                cid = f"{job}-{kind}-{init}-tile"
                tile = tiles.get((str(run.resolve()), ep))
                if tile is not None:
                    add(
                        cid,
                        tile[0],
                        run=run,
                        episode=ep,
                        label=f"{job} {kind}, reserved start {init} (verified re-render)",
                        verified=True,
                        frame_every=1,
                        delivered=bool(row["delivered"]),
                        start=int(row["start"]["init_state"]),
                    )
                    if cid in out:
                        out[cid]["hashes_compared"] = tile[1].get("hashes_compared")
                        out[cid]["hash_mismatches"] = tile[1].get("hash_mismatches")
                else:
                    add(
                        cid,
                        run / "log" / "video" / f"{ep}.scene.mp4",
                        run=run,
                        episode=ep,
                        label=f"{job} {kind}, reserved start {init} (original recording)",
                        verified=True,
                        frame_every=4,
                        delivered=bool(row["delivered"]),
                        start=int(row["start"]["init_state"]),
                    )
                ids.append(cid)
            mosaics[f"{job}-{kind}"] = ids
    # framed re-renders (batch-1): hash-verified, 1920x1080, with overlay tracks; registered
    # under the batch's ids and under readable aliases
    for batch in ("batch-1", "batch-2"):
        add_batch(out, footage / batch / "index.json" if footage else None)
    return out, mosaics


def add_batch(out: dict[str, Any], index_path: Path | None) -> None:
    """A framed batch's accepted clips, under their ids and readable aliases."""
    if index_path and index_path.is_file():
        index = json.loads(index_path.read_text())
        sha(index_path)
        for e in index.get("entries", []):
            if not e.get("accepted") or e.get("status") != "rendered":
                continue
            folder = index_path.parent
            entry = clip_entry(
                e["id"],
                folder / e["video"],
                run=Path(e["run_dir"]),
                episode=e["episode_id"],
                label=e.get("note") or e["id"],
                verified=True,
                frame_every=int(e.get("frame_every") or 1),
                delivered=e.get("delivered"),
            )
            entry["first_step"] = int(e.get("from_step") or 0)
            entry["camera"] = e.get("camera")
            entry["report"] = public_id(folder / e["report"]) if e.get("report") else None
            entry["hashes_compared"] = e.get("hashes_compared")
            entry["hash_mismatches"] = e.get("hash_mismatches")
            if e.get("track"):
                entry["track"] = convert_track(folder / e["track"], e["id"])
            out[e["id"]] = entry
            alias = FRAMED_ALIASES.get(e["id"])
            if alias:
                out[alias] = {**entry, "id": alias}


FRAMED_ALIASES = {
    "j1-chain-e000202-hero": "r1-chain-31-hero",
    "j1-job-e000102-hero": "r1-job-31-hero",
    "j1-job-e000102-establishing": "r1-job-31-establishing",
    "j1-practice2-e000503-openhome-hero": "r1-openhome-2-hero",
    "j1-practice2-e000603-finishhome-hero": "r1-finishhome-2-hero",
    "skill-cabinet-e000102-hero": "single-cabinet-hero",
    "skill-drawer-e000102-hero": "single-drawer-hero",
    "j2-chain-s01-e000102-hero": "r2-chain-dev1-hero",
    "j2-chain-s01-e000106-hero": "r2-chain-dev5-hero",
    "j2-chain-s01-e000108-hero": "r2-chain-dev7-hero",
    "j2-cheese-loaded-s04-e001301-hero": "r2-loaded-0-hero",
    "j2-cheese-aside-s05-e001501-hero": "r2-relocate-0-hero",
    "j2-chain-e000202-hero": "r2-chain-31-hero",
    "j2-job-e000102-hero": "r2-job-31-hero",
    "j2-cheese-aside-s05-e001501-top": "r2-relocate-0-top",
}


def convert_track(path: Path, cid: str) -> str:
    """The batch's track (fleetlab.film-track/1) in the film's overlay format: per frame the
    control step, each target's projected pixel and visibility, the gripper, the recorded
    contact flags, goals, gripper opening, program step and instruction."""
    sha(path)
    t = json.loads(path.read_text())
    frames = []
    for r in t["rows"]:
        objects = {}
        gripper = None
        for name, p in (r.get("targets") or {}).items():
            if p.get("u") is None or p.get("v") is None:
                continue
            item = {
                "x": p["u"],
                "y": p["v"],
                "depth": p.get("depth"),
                "visible": bool(p.get("in_frame")),
                "cx": p.get("cu"),
                "cy": p.get("cv"),
                "bbox": p.get("bbox"),
                "outline": p.get("outline"),
            }
            if name == "gripper":
                gripper = item
            else:
                objects[name] = item
        frames.append(
            {
                "step": r["step"],
                "objects": objects,
                "gripper": gripper,
                "touch": r.get("contact") or {},
                "goals": r.get("goals"),
                "gripper_opening": r.get("gripper_opening"),
                "program_step": r.get("program_step"),
                "instruction": r.get("instruction"),
            }
        )
    out = PUBLIC / "footage" / f"{cid}.track.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "format": "mayfly.explainer-track/1",
                "source": public_id(path),
                "source_sha256": sha(path),
                "width": t["width"],
                "height": t["height"],
                "camera": t.get("camera"),
                "goals": t.get("goals"),
                "program": t.get("program"),
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )
    return f"footage/{out.name}"


# ---- timeline ---------------------------------------------------------------------------------


def beats_of(script: str) -> list[tuple[str, str]]:
    body = re.sub(r"<!--.*?-->", "", script, flags=re.S)
    out = []
    for part in re.split(r"^## ", body, flags=re.M)[1:]:
        head, _, text = part.partition("\n")
        out.append((head.split(":")[0].strip(), " ".join(text.split())))
    return out


def timeline(bound: Path, voice: Path | None, music: Path | None) -> dict[str, Any]:
    cue_file = json.loads((HERE / "cues.json").read_text())
    cues = cue_file["beats"]
    holds = cue_file.get("holds", {})  # seconds of picture after the voice, per beat
    script = (bound / "page" / "index.md").read_text()
    beats = []
    for beat, text in beats_of(script):
        entry: dict[str, Any] = {
            "id": beat,
            "text": text,
            "voice": None,
            "voice_offset": 0.0,
            "voice_seconds": 0.0,
            "cues": {},
        }
        if not text:
            entry["seconds"] = SILENT.get(beat, 4.0)
            beats.append(entry)
            continue
        words = text.split()
        align = None
        audio = voice / f"{beat}.mp3" if voice else None
        if audio and audio.is_file():
            meta = voice / f"{beat}.json"
            info = json.loads(meta.read_text()) if meta.is_file() else {}
            seconds = float(info.get("seconds") or probe_audio(audio))
            align = info.get("alignment")
            link = PUBLIC / "audio" / "voice" / audio.name
            place(link, audio)
            entry["voice"] = f"audio/voice/{audio.name}"
            sha(audio)
        else:
            seconds = len(words) / WPM * 60
        entry["voice_offset"] = LEAD_IN
        entry["voice_seconds"] = round(seconds, 3)
        entry["seconds"] = round(LEAD_IN + seconds + TAIL + float(holds.get(beat, 0.0)), 3)
        for name, phrase in cues.get(beat, {}).items():
            at = cue_time(text, phrase, seconds, align)
            if at is not None:
                entry["cues"][name] = round(LEAD_IN + at, 3)
        beats.append(entry)
    out: dict[str, Any] = {
        "format": "mayfly.explainer-timeline/1",
        "source": "voice" if voice else "estimate",
        "beats": beats,
        "music": None,
        "voice_dir": str(voice) if voice else None,
    }
    if music and music.is_file():
        link = PUBLIC / "audio" / f"music{music.suffix}"
        place(link, music)
        out["music"] = f"audio/{link.name}"
        sha(music)
    return out


def cue_time(text: str, phrase: str, seconds: float, align: dict[str, Any] | None) -> float | None:
    i = text.lower().find(phrase.lower())
    if i < 0:
        return None
    if align and align.get("characters"):
        chars = "".join(align["characters"])
        j = chars.lower().find(phrase.lower())
        if j >= 0:
            return float(align["character_start_times_seconds"][j])
    before = len(text[:i].split())
    return seconds * before / max(1, len(text.split()))


def probe_audio(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


# ---- main -------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--bound",
        type=Path,
        required=True,
        help="the bound script's folder (page/index.md, derived.json, bindings.json)",
    )
    ap.add_argument(
        "--voice",
        type=Path,
        help="narration folder: <beat>.mp3 and <beat>.json (seconds, alignment)",
    )
    ap.add_argument("--footage", type=Path, default=CACHE / "explainer" / "footage")
    ap.add_argument("--music", type=Path)
    args = ap.parse_args()
    bound = args.bound.expanduser()
    (PUBLIC / "data").mkdir(parents=True, exist_ok=True)
    numbers, quotes = numbers_and_quotes(bound)
    clip_index, mosaics = clips(args.footage.expanduser() if args.footage else None)
    film = {
        "format": "mayfly.explainer-data/1",
        "generated": subprocess.run(
            ["date", "-Iseconds"], capture_output=True, text=True
        ).stdout.strip(),
        "numbers": numbers,
        "quotes": quotes,
        "programs": {"r1": programs(R1 / "run", R1_LABELS), "r2": programs(R2 / "run", R2_LABELS)},
        "traces": {
            "r1-start31-chain": job_trace(
                R1 / "run" / "release" / "run" / "run", "e000202", "akita_black_bowl_1", 0
            ),
            "r1-start31-job": job_trace(
                R1 / "run" / "release" / "run" / "run", "e000102", "akita_black_bowl_1", 0
            ),
            "r1-practice2-chain": job_trace(
                R1 / "run" / "runs" / "session-01" / "run", "e000103", "akita_black_bowl_1", 0
            ),
            "r1-practice2-job": job_trace(
                R1 / "run" / "runs" / "session-03" / "run", "e000603", "akita_black_bowl_1", 0
            ),
        },
        "clips": clip_index,
        "mosaics": mosaics,
        "fleet": {
            **story_b(STORY_B_DERIVED),
            "probe": probe_traces(),
            "regular_work": regular_work(),
            **fleet_gains(),
            "console": fleet_console(),
        },
        "provenance": {},
    }
    sha(ENG / "engagement" / "engagement.json")
    tl = timeline(
        bound,
        args.voice.expanduser() if args.voice else None,
        args.music.expanduser() if args.music else None,
    )
    film["provenance"] = {
        "inputs": [{"path": public_id(Path(p)), "sha256": h} for p, h in sorted(_sources.items())]
    }
    (PUBLIC / "data" / "film-data.json").write_text(
        json.dumps(film, indent=1, ensure_ascii=False) + "\n"
    )
    (PUBLIC / "data" / "timeline.json").write_text(
        json.dumps(tl, indent=1, ensure_ascii=False) + "\n"
    )
    total = sum(b["seconds"] for b in tl["beats"])
    print(
        f"film-data: {len(numbers)} numbers, {len(quotes)} quotes, {len(clip_index)} clips, "
        f"{sum(len(v) for v in film['programs'].values())} programs; "
        f"timeline {tl['source']}: {len(tl['beats'])} beats, {total:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
