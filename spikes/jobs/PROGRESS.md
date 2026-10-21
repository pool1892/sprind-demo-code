# Lane S progress (spike: chaining goal-suite skills)

Branch `demo/spike`, worktree `/home/cs/code/robots-wt-spike`, base `main` at
`2c3da47`. Results under `~/.cache/robots-probes/demo/2026-10-20/spike/`.
Times are Pacific.

## 15:38

- Ran: the light goal sanity check (`goals_check.py`): every goal-suite base
  task builds, the two unary goals included (turn on the stove, open the middle
  drawer); this runner's predicate evaluation agrees with the env's own LIBERO
  check; LIBERO's `On` needs contact (the cheese held 6 cm above the bowl does
  not count); after settling, cheese-in-bowl holds while bowl-on-plate does not,
  and the reverse. A model-free dry run exercised the three switch variants.
- Running: the heavy batch `batches/naive10.json`: the eight single skills on
  start 0, then J1 to J10 naive on starts 0 to 9.
- Next: the report script (tables, strips) while the batch runs; then the
  confirmation runs on the two most promising combinations (20 starts, naive,
  placed switch, release transition).
- Blockers: none. Findings for the integrator are collected in the README.

## 15:50

- Ran (heavy, 527 s, load average about 2): single skills on start 0 (S6, S4,
  S9, S8, S3, S7 delivered; S1 bowl-into-top-drawer timed out at 300 steps;
  S2 turn-on-stove reaches native success but keeps gripping the knob, so it
  fails the release rule), then J1 to J10 naive on init states 0 to 9.
- Naive delivered of 10: J1 4, J2 1, J6 3, all others 0 (J3, J4, J5, J7, J8,
  J9, J10). First skills reach native success 9 or 10 of 10 everywhere; the
  second skill fails from the state the first left. At the switch the hand is
  20 to 43 cm from where the job started (mostly 19 to 26 cm lower). In J1 and
  J2 the cheese is already released at the switch (gripper commanded open in
  10 of 10), so "held at switch" is not the mechanism there; the hand fails to
  grasp the loaded bowl. J8 (independent steps) fails 10 of 10: the hand never
  approaches the bowl from the drawer.
- Throughput: 132 to 145 ms per policy call (batch 1, fp16_res32), 4 to 5 s
  per chained start (300 to 425 control steps), about 6 ms per control step
  for the simulator, rendering, privileged state and predicates.
- Running: confirmations on J1 and J2, 20 starts (init 0 to 19): naive, placed
  switch, release (5 cm), plus `home` (bounded ramps back to the start
  position) and the host-only `teleport` diagnostic; J8 as a control.
- Next: tables and recommendation in README.md; strips are in figures/.

## 16:43

- Ran: confirmations on J1 and J2, 20 starts (init 0 to 19). Delivered of 20:
  J1 naive 5, placed 10, release 8, home 0, teleport 1; J2 naive 1, placed 6,
  release 8, home 2, teleport 2. Paired against naive: J1 placed rescues 5 and
  breaks none; J2 release rescues 7 and breaks none. Returning the hand (home)
  or the whole arm (teleport) to the start makes J1 worse: the obstacle is the
  loaded bowl (no grasp from the start pose), not the hand's position. J8: 0 of
  10 in every variant (the open drawer defeats bowl-on-plate even from home).
- Ran: J3 and J5 at 20 starts: 0 of 20 in naive, placed and release. J3: the
  policy puts the cheese where the bowl used to be (4 to 7 cm from the bowl's
  start position, 8 to 12 cm from the bowl). J5: after the release the push
  tips the bowl off the plate (11 of 20). J7 at 10: 0 naive, 0 placed, 0
  release (opening the hand on the knob turns the stove off), 3 home.
- Ran: J1 with step 2 rephrased ("pick up the black bowl and place it on the
  plate"): placed 7 of 20, release 5 of 20; no better than the task's own words.
- Interruption: the second batch was stopped at 131 of 220 runs at about 16:32
  by the harness's 30-minute limit on a background command (it had waited for
  lane P's heavy run, about 16:07 to 16:20); resumed at 16:41 with --resume.
- Running: the rest of the second batch (J7 home, J4, J9, J6, J10 variants at
  10), then batch 4: J1 and J2 `retry` (placed switch, two 150-step attempts
  at step 2 with a release between) at 20, and six spike-only pairs that share
  no object (K1 to K6) at 10, to look for combinations that just work.
- Next: README tables, strips and recommendation; handover.
- No blocker. No combination reaches four fifths delivered in any variant so far.

## 17:30

- Ran: the second batch's remaining 89 runs (J7 home, J4, J9, J6, J10 variants),
  batch 4 (J1 and J2 retry at 20; six spike-only pairs K1 to K6 naive at 10),
  batch 5 (K1, K4 at 20 in naive, placed, release; home, K5, K6 at 10),
  batch 6 (extend30 on J1, J2 at 20 and J9 at 10), batch 7 (J9 at 30 starts:
  naive, extend30; placed and release at 20; J6 extend30; J1, J2 extend60),
  batch 8 (K7 = J9 reversed; J10 extend30). 1,040 chained runs in all.
- Delivered: no combination reaches four fifths in any variant. Best: J9 extend30
  (a second 30-step policy step under the cheese instruction, expressible today)
  20 of 30 against naive 2 of 30 on the same starts (19 rescued, 1 broken); J1
  placed or extend60 10 of 20 (naive 5); J2 release or extend30 8 of 20 (naive 1);
  J4 home 5 of 10 (outside the vocabulary). Retry, rephrasing, home and teleport do
  not fix the loaded carry. No pair just works (best naive K5 5 of 10). Reverse
  orders J3 and K7 fail 0 of 20 and 0 of 10: the cheese goes where the bowl was.
- Running: batch 9, J9 with extend15, extend45, extend60 on starts 0 to 29.
- Next: README summary, tables and recommendation; handover.

## 17:42

- Ran: batch 9 (J9 with 15, 30, 45 and 60 extra steps of the cheese skill, starts
  0 to 29): 9, 20, 16 and 22 of 30 delivered against naive 2 of 30 (extend60: 20
  rescued, none broken). 1,130 chained runs and 8 single skills in all.
- Done: README.md with the tables (native reached, native at end, placed at end,
  delivered), strips, paired flips, throughput, recommendation (job one J9, job two
  J1; the order rule does not bite; nothing just works) and findings for the
  integrator; runs.csv with every run's start identity. ruff check and format pass on
  spikes/jobs; the dry run, the goal check and a regenerated report agree with the
  committed tables.
- Not done: make check (this branch changes only spikes/jobs/, which ruff, mypy and
  pytest do not cover); release starts 30 to 49 untouched.

## 17:53

- An independent review found the gate-1
  numbers sound and five descriptive errors in the README; all five corrected after
  recounting from the traces: the expressibility table against lane J's contract on
  `demo/jobs` (`home` exists with a different motion, so this spike's home results are
  unverified for it; `retry` and `placed-pickup` depend on placed; `release-pickup`
  separate); J9 step outcomes kept apart from tags (naive bowl step times out in 26 of
  30, tag `second_skill_failed` 22) and residuals given per length; J1 home and
  teleport close on the bowl but lift it less than 2 cm in 19 and 15 of 20; the cheese
  is usually, not always, released at native success (J9 starts 10, 19, 22), and the
  switch height does not explain `extend60` (10.6 cm); J10's push step runs on 4 starts
  and never achieves its goal. Also corrected: J9 `extend30` loses the cheese during
  the bowl step, not during the extra steps.
