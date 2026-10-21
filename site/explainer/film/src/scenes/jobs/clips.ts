// The footage of the job beats, by clip id from film-data `clips`. Every id is a recorded run
// or its hash-verified re-render: the framed 1920x1080 re-renders with overlay tracks
// (`-hero`, `-top`) where they exist, else the 720 px re-renders. Switching a beat's footage is
// one line here; the scenes read nothing else about it.
export const CLIPS = {
  // job 1, reserved start 31: the plain chain (hero and wrist cameras)
  r1Chain: 'r1-chain-31-hero',
  r1ChainWrist: 'r1-chain-31-wrist',
  // job 1, practice start 2: open then home (loses the bowl), finish then home (keeps it)
  r1OpenHome: 'r1-openhome-2-hero',
  r1FinishHome: 'r1-finishhome-2-hero',
  // job 1, plain chain on practice start 2 (session 1), the run r1-measure charts (until a
  // re-render exists the chart shows alone)
  r1ChainPractice2: 'r1-chain-practice2-hero',
  // job 2, plain chain on practice starts 1 and 5 (both fail at the plate): the hero camera,
  // and from above, tilted 25 degrees so the cheese stays in sight under the hand, for the
  // moments the hand would hide it (the hero clip stands in if a tilted clip is missing)
  r2Fail1: 'r2-chain-dev1-hero',
  r2Fail5: 'r2-chain-dev5-hero',
  r2Fail1Top: 'r2-chain-dev1-top-tilt',
  r2Fail5Top: 'r2-chain-dev5-top-tilt',
  // job 2, practice start 0: the cheese loaded into the bowl, and the cheese set aside (hero
  // camera, and from above, tilted, for the slide and the release)
  r2Loaded: 'r2-loaded-0-hero',
  r2Relocate: 'r2-relocate-0-hero',
  r2AsideTop: 'r2-aside-0-top-tilt',
} as const;
// The final exams play film-data `mosaics` (r1-chain, r1-job, r2-chain, r2-job): every
// reserved run of both jobs, in start order.

/**
 * Control steps at which each program step of a clip's run begins, read from the run's
 * `job_step_events` table (plate-cabinet-01 session-05 e001501, session-04 e001301;
 * replacement-01 session-03 e000503 and e000603). They only time highlights and are never
 * shown. A clip with an overlay track carries `program_step` itself, which takes precedence.
 */
export const PROGRAM_STEPS = {
  // grab with the cheese skill, slide aside, let go, home, plate, home, bowl, (tail)
  r2Relocate: [0, 60, 80, 100, 130, 279, 316, 405],
  // cheese into the bowl, home, plate, home, bowl (times out)
  r2Loaded: [0, 120, 151, 301, 337],
  // bowl, open the gripper, home, drawer
  r1OpenHome: [0, 85, 105, 138],
  // bowl, let the placement finish, home, drawer
  r1FinishHome: [0, 85, 115, 148],
} as const;
