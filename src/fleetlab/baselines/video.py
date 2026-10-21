"""Longest failure videos, with end-effector travel breaking duration ties."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from fleetlab.agents.run import AgentResult
from fleetlab.baselines.common import (
    BaselineAnswer,
    budget_label,
    export_catalogue,
    ranking,
    static_description,
    write_json,
)
from fleetlab.baselines.models import (
    VALUE_INSTRUCTIONS,
    Choice,
    catalogue_prompt,
    model_call,
    validate_actions,
)
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, cut
from fleetlab.contracts.env import Camera
from fleetlab.log.reader import LogReader

FRAMES_PER_CAMERA = 16


def dramatic_failures(export: Path) -> list[dict[str, Any]]:
    """Rank only logged failures with videos: duration, total EEF travel, episode ID."""
    with LogReader(export) as reader:
        return list(
            reader.query("""
            WITH motions AS (
                SELECT s.episode_id, eef_pos,
                    lag(eef_pos) OVER (PARTITION BY s.episode_id ORDER BY step) AS prev
                FROM steps s JOIN episodes e USING (episode_id)
                WHERE e.outcome = 'failure'
            ), travel AS (
                SELECT episode_id, sum(sqrt(
                    pow(eef_pos[1]-prev[1], 2) + pow(eef_pos[2]-prev[2], 2) +
                    pow(eef_pos[3]-prev[3], 2))) AS motion_m
                FROM motions GROUP BY episode_id
            )
            SELECT e.episode_id, robot_id, site_id, duration_s, steps, task_text,
                video_scene, video_wrist, coalesce(motion_m, 0) AS motion_m
            FROM episodes e LEFT JOIN travel USING (episode_id)
            WHERE outcome = 'failure' AND (video_scene IS NOT NULL OR video_wrist IS NOT NULL)
            ORDER BY duration_s DESC, motion_m DESC, episode_id
        """).to_pylist()
        )


def filmstrip(reader: LogReader, clip: dict[str, Any], target: Path) -> None:
    """Two camera timelines, all the way from start to end, as runtime-supported images."""
    episode = clip["episode_id"]
    logged = reader.query(
        "SELECT step FROM steps WHERE episode_id = ? AND frame_index IS NOT NULL ORDER BY step",
        [episode],
    )["step"].to_pylist()
    if not logged:
        raise ValueError(f"failure {episode} has no recorded video frames")
    wanted = [int(logged[int(i)]) for i in np.linspace(0, len(logged) - 1, FRAMES_PER_CAMERA)]
    cameras = [c for c in Camera if clip[f"video_{c.value}"] is not None]
    width, height, columns = 192, 214, 4
    rows = 4 * len(cameras)
    sheet = Image.new("RGB", (width * columns, height * rows + 36), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), f"{episode}  robot {clip['robot_id']}  {clip['duration_s']:g}s", fill="black")
    for ci, camera in enumerate(cameras):
        for i, frame in enumerate(reader.frames(episode, wanted, camera)):
            x, y = (i % columns) * width, 36 + (ci * 4 + i // columns) * height
            image = Image.fromarray(frame.image).convert("RGB")
            image.thumbnail((width, 192))
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + 193), f"{camera.value} step {frame.frame_step}", fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        sheet.save(handle, format="PNG")


def dramatic_videos(
    export: Path,
    inputs: Path,
    *,
    run_root: Path | None = None,
    caller: Callable[..., AgentResult] | None = None,
) -> BaselineAnswer:
    """One six-clip answer per repeat; the count-3 condition uses its first three clips."""
    clips = dramatic_failures(export)[:6]
    if not clips:
        raise ValueError("no logged failure videos; dramatic-videos baseline is unavailable")
    inputs.mkdir(parents=True, exist_ok=False)
    write_json(inputs / "selection.json", clips)
    images = []
    with LogReader(export) as reader:
        for clip in clips:
            path = inputs / f"{clip['episode_id']}.png"
            filmstrip(reader, clip, path)
            images.append(path)
    entries = export_catalogue(export)
    filmed = {c[key] for c in clips for key in ("robot_id", "site_id")}
    candidates = [a for a in entries if a.target.scope in filmed]
    prompt = (
        f"Watch the {len(clips)} selected dramatic failure videos as chronological "
        "filmstrips, 16 frames per available camera including start and end. Each image "
        "names its robot and episode; camera timelines are separated. For EACH video, "
        "choose the ONE most plausible intervention from the catalogue for that filmed "
        "robot (components or configuration keys) or its site (conditions or site-scoped "
        "configuration keys), with the verb offered for that target. Return nominations "
        "in the same order as the clips, exactly one per episode, with a brief visual "
        "rationale. No telemetry, health, change logs or skills are available. "
        "This same answer serves all budgets: the 3-action condition uses the first "
        "three clips in this order; the 6-action and 4/8-hour conditions use all six. "
        + VALUE_INSTRUCTIONS
        + "Choose the best visual hypothesis even if uncertain.\n"
        + json.dumps(
            {
                "system": static_description(export),
                "clips": [
                    {
                        key: c[key]
                        for key in ("episode_id", "robot_id", "site_id", "task_text", "duration_s")
                    }
                    for c in clips
                ],
                "catalogue": catalogue_prompt(candidates, values=False),
            },
            separators=(",", ":"),
        )
    )
    raw, metadata = model_call(
        prompt,
        images=images,
        run_root=run_root,
        caller=caller,
        label="baseline dramatic-videos all budgets",
        schema=VideoNominations,
    )
    return video_answer(entries, clips, raw, metadata, inputs)


def video_answer(
    entries: list[Action],
    clips: list[dict[str, Any]],
    raw: Any,
    metadata: dict[str, Any],
    inputs: Path,
) -> BaselineAnswer:
    """Validate and cut one six-clip answer, also used for recorded runtime reruns."""
    filmed = {c[key] for c in clips for key in ("robot_id", "site_id")}
    candidates = [a for a in entries if a.target.scope in filmed]
    nominations = VideoNominations.model_validate(raw).nominations
    if [n.episode_id for n in nominations] != [c["episode_id"] for c in clips]:
        raise ValueError("vision must nominate once per selected clip, in order; no retry")
    nominated = []
    for nomination, clip in zip(nominations, clips, strict=True):
        if nomination.index >= len(candidates):
            raise ValueError("vision nominated an index outside the action catalogue")
        action = candidates[nomination.index]
        if action.target.scope not in {clip["robot_id"], clip["site_id"]}:
            raise ValueError("vision nominated an action for a different robot or site")
        nominated.append(action)
    validate_actions(nominated, entries)
    selections, rankings = {}, {}
    for k in (3, 6):
        chosen, reasons, seen = [], [], set()
        # Prefix the clips BEFORE deduplicating: later clips cannot fill count-3 vacancies.
        for action, nomination in zip(nominated[:k], nominations[:k], strict=True):
            key = action.model_dump_json()
            if key not in seen:
                chosen.append(action)
                reasons.append(nomination.rationale)
                seen.add(key)
        selections[k] = chosen
        padding = [a for a in entries if a.model_dump_json() not in seen][
            : max(0, 10 - len(chosen))
        ]
        rankings[k] = ranking(
            chosen + padding,
            reasons
            + [
                "FORMAT PADDING ONLY; not a video nomination; never included in a FixSet."
                for _ in padding
            ],
        )
    return BaselineAnswer(
        "dramatic_videos",
        rankings[6],
        by_budget={"count-3": rankings[3]},
        explicit_fix_sets={
            budget_label(b.kind, b.amount): cut(
                selections[3 if b.kind == "count" and b.amount == 3 else 6], b
            )
            for b in SCORING_BUDGETS
        },
        metadata={
            "calls": [
                {
                    "videos": len(clips),
                    "nominations": [n.model_dump() for n in nominations],
                    **metadata,
                }
            ],
            "inputs": str(inputs),
            "selected": clips,
            "cost_budget_videos": 6,
            "values": "recorded-standard; hidden from model",
            "frames_per_camera": FRAMES_PER_CAMERA,
            "explicit_fix_sets_authoritative": True,
            "format_padding_never_selected": True,
            "count3_prefix_before_deduplication": True,
        },
    )


class VideoNomination(Choice):
    episode_id: str


class VideoNominations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nominations: list[VideoNomination] = Field(min_length=1)
