# Footage for the explainer film: framed, verified re-renders with overlay tracks

`python -m fleetlab.film frame` re-executes recorded job episodes and draws them at film
size from chosen camera poses; `python -m fleetlab.film stills` draws stills of several
cameras from one replay. Both live in `src/fleetlab/film/framing.py`.

## What makes a clip trustworthy

Every clip is an exact replay of a recorded run, checked the way
`python -m fleetlab.jobs replay` checks its re-renders (they share `load_record` and
`replay` in `fleetlab.jobs.replay`): the episode's recorded actions are executed again in a
scene built from its recorded identity, MuJoCo's state hash after the reset and after every
control step is compared with the recorded one, and the delivery verdict at the end with the
recorded verdict. The whole episode is always replayed and checked, also when only a step
range is drawn. A clip is accepted only when every hash and the verdict match; then its
video, track and debug still are kept and its report carries their SHA-256. Otherwise the
clip keeps only `<id>.rejected-rerender.json`, which says where the replay left the record;
an earlier acceptance of the clip is removed with its outputs, also when the run cannot be
read at all.

A later run skips a clip only while its report's spec, the SHA-256 of its video and track,
and `source_sha256`, a digest of what it was drawn from (the episode's replay record and
identity under `truth/jobs/`, its row of `episodes.jsonl`, its per-step log tables, its
rows of `job_step_events` and the run's request state), all still match; a corrected
source is drawn again. The reports of batches 1 and 2 and of the mosaic were written before
the digest existed and carry none, so a rerun of their manifests would draw those clips
again; their outputs stand as verified, and they are not to be redrawn.

Cameras never touch the physics: a placed camera is written into MuJoCo's scene after it is
updated (the two eyes and the headlight), never into the model. The hashes prove it.

## Commands

Each command is one light simulation job; run one at a time.

```sh
# a batch from a manifest; clips accepted for the same spec and sources are skipped
tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film frame \
    --clips site/explainer/clips-batch-1.json \
    --out ~/.cache/robots-probes/demo/explainer/footage/batch-1 [--only ID,ID] [--force]

# one clip without a manifest (draft size, a step range, an extra tracked body)
tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film frame \
    --run ~/.cache/robots-probes/demo/2026-10-20/reliability/replacement-01/run/release/run/run \
    --episode e000202 --camera hero-cabinet --width 1280 --height 720 \
    --from-step 60 --to-step 160 --body wooden_cabinet_1_cabinet_middle --out DIR

# stills of several cameras at chosen steps, from one verified replay (kept only when it
# is accepted: a rejected or unreadable record removes earlier stills of the same names)
tools/heavy-run --light -m 4G uv run --locked python -m fleetlab.film stills \
    --run RUN_DIR --episode e000202 --steps 0,82,145 \
    --camera hero-cabinet --camera top --camera '{"label": "low", "position": [0.8, 0, 1.1],
    "look_at": [0, 0, 0.95], "fovy": 30}' [--poses CLIPS.json] --out DIR
```

A 1920 x 1080 clip of 400 control steps takes about 15 seconds (replay, drawing,
segmentation and encoding together).

## The clips manifest

`site/explainer/clips.example.json` is a documented example; `load_clips` in the module
describes every field. In short: `{"format": "fleetlab.film-clips/1", "poses": {...},
"defaults": {...}, "clips": [...]}`, where a clip has an `id` (used in the file names),
`run`, `episode`, `camera`, and optionally `width`, `height` (even), `frame_every` (divides
20; the video runs at 20 / frame_every frames a second, real time), `from_step`, `to_step`,
`bodies` (more things to track) and a `note`. A camera is the name of a pose (the
manifest's own, then the proposed ones below), `scene` or `wrist` (the run's observation
cameras), another camera of the LIBERO scene by its MuJoCo name (`frontview`, `sideview`,
`birdview`, ...), or a pose written out: `{"position": [x, y, z], "look_at": [x, y, z],
"fovy": degrees, "roll": degrees, "up": [x, y, z]}` in world metres. `fovy` is the vertical
field of view; `up` (default +z) is the world direction that points up in the picture;
`roll` turns the picture counterclockwise.

## Outputs, per clip

- `<id>.rerender.mp4`: H.264 (CRF 18, preset slow, repeatable encode), one frame per drawn
  control step.
- `<id>.rerender.json`: the report: the clip's spec, the verification (hashes compared,
  mismatches, the first mismatching step, both verdicts), the camera with its intrinsics
  and extrinsics, the SHA-256 of video and track, `source_sha256` (above), and the largest
  coordinate difference in
  metres between the replayed positions the track projects and the logged ones (0 for the
  objects; up to about 2 mm for the gripper, see below).
- `<id>.track.json`: the overlay track (below).
- `<id>.debug.png`: four frames with every target's point (dot), the centre of its visible
  geometry (cross), its visible box and the goals' regions (yellow outlines) drawn, to see
  that the projections sit on the objects.
- `index.json`: every clip of the manifest with its status (`rendered`, `skipped`,
  `rejected`, `error`, `not run`), `accepted`, hash counts, the recorded delivery and file
  names. It is written once, when a run is complete: a run that draws removes the old index
  first, so an index is there only after a complete run, and a rerun that draws nothing
  leaves an index listing the same outputs untouched.

## The overlay track

`fleetlab.film-track/1`. The header names the run, episode, video and its SHA-256, the frame
rate and size, the camera (`view`: `K`, `fx`, `fy`, `cx`, `cy`, `world_to_camera`,
position, forward, up, right, near, far), the tracked targets, the requested goals, the
program's steps with their control steps (`program`), and the recorded delivery. Then one
row per video frame:

`t` is the control step's time in the episode (step / 20 s); a row's time in the video is
`frame` / fps.

```json
{"frame": 22, "step": 82, "t": 4.1,
 "targets": {"akita_black_bowl_1": {"u": 796.8, "v": 418.6, "depth": 0.8964,
                                   "cu": 798.5, "cv": 371.7, "in_frame": true,
                                   "visible_px": 14328, "bbox": [690, 302, 888, 434]},
             "gripper": {...},
             "wooden_cabinet_1_top_side": {"u": ..., "v": ..., "depth": ..., "in_frame": true,
                                           "outline": [[u, v], [u, v], [u, v], [u, v]]}},
 "contact": {"akita_black_bowl_1": true, ...}, "touching": {"akita_black_bowl_1": [...]},
 "goals": [true, false], "gripper_opening": 0.008, "gripper_command": 1.0,
 "program_step": 1, "instruction": "open the middle drawer of the cabinet", "phase": "program"}
```

- Pixels: origin at the top-left corner, `u` right, `v` down; the pixel in column j and row
  i has its centre at (j + 0.5, i + 0.5). The camera is an OpenCV pinhole:
  `world_to_camera` takes world metres to x right, y down, z along the line of sight, and
  u = fx x / z + cx, v = fy y / z + cy. `depth` is z in metres; `u`, `v` are null behind
  the camera.
- Targets by default: every object of the episode's `object_steps` (its body origin, which
  the log records exactly), `gripper` (the grip site between the fingertips; the log's
  `eef_pos` is sampled one 2 ms physics substep earlier, so it trails the frame by up to
  about 2 mm when the hand moves fast: at most 0.84 mm in batch 1, 1.77 mm in the mosaic),
  and each requested goal's region when it is a site (`outline`: the four corners of the
  face of its box that looks most upward, a footprint on the table or a drawer's opening).
  A clip's `bodies` add objects, MuJoCo bodies or sites by name.
- `cu`, `cv`: the centre of the target's visible geometry (better than the origin for a
  ring around a bowl). `in_frame`: in front of the camera and inside the picture.
  `visible_px` and `bbox`: the pixels where the target is actually seen, from a
  segmentation pass (occlusion counted; drawn at 1/k of the frame's size with k = width //
  960, half size at 1920 x 1080, and the numbers scaled back to full-size pixels).
- `contact`, `touching`, `goals`, `gripper_*`, `program_step`, `instruction`, `phase`: the
  run's own log tables for that control step, unchanged (`instruction` is null during
  motion, wait and home steps and at the end row; `program_step` is null in the tail, the
  policy going on after the program until the delivery check holds, and at the end row,
  whose `phase` is `end`).

## Orientation: frames are the true view

Frames are upright and not mirrored: MuJoCo reads rows bottom-up and the rows are flipped,
nothing else. LIBERO's own images, the policy's input, the recorded scene and wrist videos
and `fleetlab.jobs.replay`'s re-renders turn the read-back image by 180 degrees, which for
the scene's upright cameras is the kitchen mirrored left to right (the stove, at +y, is on
the left of the recorded scene video and on the right of a framed `scene` clip). Cutting a
recorded clip against a framed one swaps the sides of the table; a mosaic from framed clips
is consistent within itself.

## Proposed camera poses

In `fleetlab.film.framing.POSES`, world metres (the table top at z = 0.90, the robot's base
at x = -0.66, the cabinet at y = -0.24 with its drawers facing +y, the stove at y = +0.20).
Stills at 1280 x 720 are in `~/.cache/robots-probes/demo/explainer/footage/cameras/`.

| Pose | Position | Looks at | fovy | For |
|---|---|---|---|---|
| `hero-cabinet` | (0.60, 0.26, 1.53) | (-0.03, -0.14, 1.09) | 35 | job 1: the cabinet top where the bowl lands, the middle drawer and the hand, close, three-quarter from the front right; the drawer stays readable when the arm comes in to open it |
| `hero-plate` | (0.50, 0.30, 1.62) | (-0.09, -0.03, 1.00) | 37 | job 2: cream cheese, bowl and plate, the push to the stove front left to right, and the cabinet top where the bowl goes last |
| `establishing` | (1.45, 1.05, 1.80) | (-0.30, -0.05, 1.02) | 38 | the whole station and the robot, three-quarter |
| `top` | (-0.16, -0.08, 2.55) | (-0.16, -0.08, 0.90), up -x | 24 | straight down on the working area, robot at the top, +y right, for drawn diagrams |
| `top-tilt` | (0.41, -0.48, 2.40) | (-0.16, -0.08, 0.90), up -x | 26 | the same area 25 degrees off vertical from the front left, robot still at the top: straight down, the arm hides the cream cheese while the hand works it (0 visible pixels through e001501's slide); from here it stays in sight beside the fingers |

## Batches

- `site/explainer/clips-batch-1.json` -> `footage/batch-1/`: 15 clips at 1920 x 1080,
  20 frames a second, of job 1 (release chain and saved job on start 31, practice start 2,
  the two single skills) and job 2 (practice-start chain failures, the cheese loaded and
  set aside, the release chain and saved job, and one `top` clip).
- `footage/mosaic/clips-mosaic.json` -> `footage/mosaic/`: all 80 release episodes of both
  jobs from the scene camera at 480 x 480, for the final-exam mosaics.
- `site/explainer/clips-batch-2.json` -> `footage/batch-2/`: review fixes at 1920 x 1080: job
  1's plain chain on practice start 2 (`hero-cabinet`), and job 2's overhead clips of
  practice starts 1 and 5 and of the cheese loaded, from `top` and from `top-tilt`, with the
  cheese set aside (e001501) from `top-tilt`.

Whether a target can be seen is in the track: `visible_px` per frame (for example the
cream cheese while `contact` says the hand touches it).
