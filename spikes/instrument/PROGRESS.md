# Lane P spike progress (story B, the instrument)

Timebox three hours from about 15:27 PDT, 19 October 2026. Results under
`~/.cache/robots-probes/demo/2026-10-20/instrument/`.

## 15:40 PDT

- Done: the step probe (hold 10, +3 cm on x for 40, back for 40, hold 10;
  three repetitions) on robot r01 of w10-v2 under twelve conditions, light
  job, 3 workers, 70 s, `chain_ok` true in every run
  (`probe_shapes.py`). Healthy reproduces section 10's check (first move 1
  row, half-way 4 rows).
- Done: the alignment lag of `spikes/adaptation/lag.sql` and per-episode
  variants over the recorded stage-2/3 calibration runs, 3,300 episodes, no
  simulation (`telemetry_lag.py`). No heavy job is needed.
- Done: the figure (`figures/probe-shapes.png`) and the table.
- Seen so far: remote link is a fixed dead time; controller overload is a
  dead time that grows within the probe and varies between repetitions;
  arm torque loss is prompt but sags and settles short of every target; the
  four image-side conditions leave the probe identical to healthy, row for
  row.
- Next: the catalogue and learned-skill answers, the README and its verdict;
  possibly a longer step train to show the overload's growth rate.

## 15:52 PDT

- Done: README with the verdict (measured mechanism per condition, the
  recommended lookalike set with levels), the telemetry table, the catalogue
  table with hours, and the learned-skill answer with the smallest additive
  change (not made).
- Done: the step train (four steps, six conditions) and the same probe in two
  more scenes (libero_spatial 5, libero_object 0); the signatures hold.
  Under arm torque loss the hand already starts 18 mm back and 30 mm low
  (0.4), the only condition that moves the start.
- Ready, not run: `overload_compensation.py`, the one heavy job that would
  settle whether compensation also rescues the overloaded controller (lane S
  holds the GPU; the command is in README section 7; manifest checked on CPU).
- Next: a final read of the README against the outputs, then the handoff.

## 15:56 PDT: handed over

- Done: `catalogue_check.py` confirms with `contracts.actions.resolve_all`
  which action removes which fault (README section 5); the verdict's
  mechanisms as one line each.
- Not run: the heavy compensation-under-overload check (README section 7);
  it is the integrator's to queue.
- Every step of the assignment and the coordinator's three corrections are
  answered in README.md. No path outside `spikes/instrument/` was touched.

## 16:48 PDT: lane P's build (second assignment)

- Done and committed: the probe field and its recipe (9193204); the recipe
  command, its page and the probe figure (e545e52); the base skill, the
  driver's mounting with probes and the two story B fleet recipes (8439303);
  the plan's form of the command (77741a7); README section 10 (58f28e4).
- Evidence: the recipe command on w10-v2 (r05 against r01) as a light job:
  first move 18 and 1 rows, half-way 21 and 4, its SQL the same. Both story B
  worlds composed on the CPU under results/worlds/ (STORYB 04f239f1...,
  STORYB-LATER 9ad7d6fb...).
- Known: the skills roster test fails until the story skills are registered
  (with lane K's).
- Next: a targeted test batch (running), then the full make check under the
  machine-wide lock and make smoke-world on the heavy queue, then the handover.
