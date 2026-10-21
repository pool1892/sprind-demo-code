# The reliability film: sources and bindings

Working files for the revised demo film. Nothing here is published by
the agents that build it. The existing cuts, scripts and fields under `site/manifests/` and
`site/narration/` stay as they are; this film uses its own files:

- `site/reliability/script.md`: the narration of the final cut, one section per beat, every
  number a placeholder, a `source:` note under each beat.
- `site/reliability/fields.json`: what each placeholder means and which record field holds
  it. `r1_*` is job 1 (replacement-01), `r2_*` job 2 (plate-cabinet-01), `b_*` story B,
  copied unchanged from `site/narration/fields.json`.
- `site/manifests/reliability-final-2-2026-10-20.json`: the current final cut, about three
  and a half minutes, with `.records.json` (the files behind every record) and
  `.provenance.json` (the build's provenance, committed after the build).
- `site/manifests/reliability-final-1-2026-10-20.json`: final cut 1, kept unchanged
  (script as of b50d84e, rendered as `film/final-1`). Cut 2 repairs its opening line,
  splits job 2's experiment into `r2-loaded` and `r2-relocated` so the cheese is grasped
  and slid aside while that sentence is read, and states job 2's counts as deliveries.
- Earlier private drafts of the first case, kept unchanged:
  `reliability-first-case-2026-10-20.json` (draft 1, script at commit 82222f9, rendered as
  `film/first-case-draft-1`) and `reliability-first-case-2-2026-10-20.json` (draft 2,
  script at 7cb3947, rendered as `film/first-case-draft-2`), each with its provenance.

The final cut sets `"sources_on_screen": "viewer"`: no run ids or digests in the frames;
`provenance.json` keeps them all.

## Build

From the checkout, with the environment from `make setup`, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and a disk-backed `TMPDIR`:

```sh
out=~/.cache/robots-probes/demo/2026-10-20/reliability/film/final-2
cut=site/manifests/reliability-final-2-2026-10-20
film="tools/heavy-run --light -m 2G uv run --locked python -m fleetlab.film"

# 1. bind the script's numbers; the manifest's "script" names this bound copy
$film bind site/reliability/script.md --fields site/reliability/fields.json \
    --records $cut.records.json --out "$out/narration" --final

# 2. resolve every clip, count and re-render; print the timeline
$film check $cut.json

# 3. the silent proof (captions at reading pace)
$film build $cut.json --out "$out/silent"

# 4. the voice (key from the environment, model given explicitly): Brian reads the BOUND
#    script. Do not pass --records: film-narrate would bind with
#    site/narration/fields.json, which lacks these names. Without a key it runs dry.
tools/film-narrate render --script "$out/narration/script.md" --model <model> \
    --out "$out/voice/brian"

# 5. the final voiced cut
$film build $cut.json --out "$out/cut" --narration "$out/voice/brian" --final
```

The pairs on screen were chosen by rule with `film pairs` (the first rescued start):

```sh
R1=~/.cache/robots-probes/demo/2026-10-20/reliability/replacement-01/run
R2=~/.cache/robots-probes/demo/2026-10-20/reliability/plate-cabinet-01/run
$film pairs --before $R1/release/run/run --after $R1/release/run/run \
    --before-where job_id=j0002 --after-where job_id=j0001          # start 31, 19 / 0
$film pairs --before $R2/release/run/run --after $R2/release/run/run \
    --before-where job_id=j0002 --after-where job_id=j0001          # start 31, 15 / 0
$film pairs --before $R2/runs/session-01/run --after $R2/runs/session-05/run \
    --before-where job_id=j0001 --after-where job_id=j0015          # start 1, 3 / 0
$film pairs --before $R2/runs/session-04/run --after $R2/runs/session-05/run \
    --before-where job_id=j0013 --after-where job_id=j0015          # start 0, 6 / 0
```

## The final cut, element by element

`R1` and `R2` as above. Both investigations: gpt-6-astra at xhigh through the job driver,
allocation profile `pool-v1`, no order rule; X-VLA unchanged.

| Beat | On screen | Source |
|---|---|---|
| open | title card | none (no number) |
| r1-request | the lead's request, first two sentences, verbatim; the plain chain placing the bowl, steps 0 to 82 | `R1/job_run.json` `request.request_text` (exact prefix, checked when the manifest was written); `R1/release/run/run` e000202, reserved start 31, verified 720px scene re-render |
| r1-failure | the bowl knocked off as the hand turns to the drawer, steps 82 to 230 | e000202 as above; spoken 5 of 10 from session 1, j0001 |
| r1-experiment | card: "does the bowl stay in place?", a labelled summary; open-then-home beside placement-runs-on-then-home, practice start 2, steps 30 to 200 | check `handoff-retention-and-delivery` `purpose`/`expects`; `R1/runs/session-03/run` e000503, e000603, original 360px; spoken 5 and 10 of 10 (j0005, j0006), 30 steps (saved job step 2) |
| r1-release | plain chain beside the saved job, reserved start 31; card 20 of 20, plain chain 1, 19 rescued, 0 broken | e000202 / e000102 verified 720px scene re-renders; receipt, reconciled in `replacement-01/coordinator-verification.json`; confirmation 20 of 20 from session 4, j0008 |
| r1-kept | the learned skill's description, verbatim | `R1/library/skills/quiet-handoff-versus-path-clearance/SKILL.md`, line 3 |
| r2-request | the lead's request, first two sentences, verbatim; the plain chain starting, steps 0 to 60 | `R2/job_run.json` `request.request_text` (exact prefix); `R2/runs/session-01/run` e000102, practice start 1, original 360px |
| r2-failure | the plate push running out of time, steps 60 to 299 | the same development episode (plate step times out at 300, bowl step skipped); spoken 7 of 10 from session 1, j0001 |
| r2-loaded | cheese loaded into the bowl, practice start 0: loading and plate (steps 0 to 301, speed 2), then the bowl failing (301 to 637, speed 1.5) | `R2/runs/session-04/run` e001301 (j0013), original 360px; spoken 8 and 4 of 10 delivered (j0004, j0013); the check extension from the account (`station-goal-and-handoff-audit`) |
| r2-relocated | the same start with the cheese grasped, slid aside and released (steps 0 to 130, speed 1, under that sentence), then plate and bowl (130 to 420, speed 3) | `R2/runs/session-05/run` e001501 (j0015), original 360px; spoken 10 of 10 delivered |
| r2-release | card 20 of 20, plain chain 5, 15 rescued, 0 broken; plain chain beside the saved job, reserved start 31 | receipt, reconciled in `plate-cabinet-01/coordinator-verification.json`; e000202 / e000102 verified 720px scene re-renders (`plate-cabinet-01/replay-film-reader-verification.json`); confirmation 20 of 20 from session 6, j0019 |
| r2-kept | the learned skill's description, verbatim | `R2/library/skills/scene-preparation-can-move-the-bottleneck/SKILL.md`, line 3 |
| b-instrument | the fleet's success by site and day; the agent's probe, commanded against measured | `instrument/engagement-1/exhibit/figures/r1-fleet-sites.png`; `film/rehearsal-3/figures/probe-first-fleet.png`; as in the earlier final cuts |
| b-result | reserved panel by state; card 558 of 576, before 451, 113 rescued, 6 broken | `exhibit/figures/panel-states.png`; `loop/evaluation/summary.json` (`b_*`) |
| close | card: both jobs, both plain chains, the fleet | the bindings above |

Two different failures of job 2 are shown and kept apart: the development clip (practice
start 1) is the plate push timing out; the reserved plain chain (start 31) reaches the
plate goal at step 145, then its bowl step times out at 445 with both goals false. The
cheese is moved aside on purpose; the film does not claim it was the only change or an
isolated cause. No claim is made that reusing the first job's library made the second
faster.

Fields defined but not spoken: the other job-1 variants (`r1_finish_only`, `r1_open_only`,
`r1_home_only`, `r1_drawer_first`), `r*_raw_verified`, `r*_sessions`, `r*_model_minutes`
and `r*_practice_runs`. A field is bound only when the script speaks it, so a number put on
a card must also be spoken.

## Known gaps

- Development footage is the original 360px at a frame every fourth control step; only the
  release pairs are 720px re-renders. Wrist re-renders exist for both release pairs but the
  cut uses the scene view.
- Until root records the voice, the cut is silent and its captions follow a reading pace
  of 150 words a minute; the voiced cut's beat lengths follow the recorded audio.
