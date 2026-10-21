// The recorded moments of a hand-off (job 1's plain chain), read from a run's trace exactly as
// recorded: contact and goal intervals are never joined or smoothed. Shared by r1-fail and
// r1-measure; also maps a practice episode to the layout (start) it ran on.
import type {FilmData} from '../../data';

export type HandoffTrace = {
  episode?: string;
  height_m: number[];
  touch: [number, number][];
  goal: [number, number][];
  events: {program_step: number | null; kind: string; from: number; to: number; instruction: string | null}[];
};

export type Handoff = {
  switchStep: number; // the next skill starts (the bowl skill's goal first holds)
  holdEnd: number; // end of the recorded contact that spans the switch
  release: number; // the longest recorded gap in contact before the fall: its start...
  recontact: number; // ...and its end, where the hand touches the bowl again
  fallStart: number; // last step at the cabinet top before the fall
  fallEnd: number; // first step down near the table
  cabinet: number; // bowl height at the switch, metres
  table: number; // bowl height at the start, metres
  instructions: string[];
};

export const handoffMoments = (t: HandoffTrace): Handoff => {
  const policy = t.events.filter((e) => e.kind === 'policy');
  const switchStep = policy[1]?.from ?? t.goal[0]?.[0] ?? 0;
  const h = t.height_m;
  const cabinet = h[switchStep] ?? h[0];
  const table = h[0];
  // the fall: first step after the switch down near the table, and the last step before it at
  // the cabinet top
  const low = table + (cabinet - table) * 0.2;
  let fallEnd = h.length - 1;
  for (let s = switchStep; s < h.length; s++) {
    if (h[s] < low) {
      fallEnd = s;
      break;
    }
  }
  let fallStart = switchStep;
  for (let s = fallEnd; s > switchStep; s--) {
    if (h[s] >= cabinet - 0.004) {
      fallStart = s;
      break;
    }
  }
  // contact as recorded: the interval spanning the switch, and the longest gap between the
  // switch and the fall (the hand lets go, then comes back)
  const spanning = t.touch.find(([a, b]) => a <= switchStep && b > switchStep);
  const holdEnd = spanning ? spanning[1] : switchStep;
  const window = t.touch.filter(([a, b]) => b > switchStep && a < fallStart);
  let gap: [number, number] = [holdEnd, holdEnd];
  for (let i = 0; i + 1 < window.length; i++) {
    const g: [number, number] = [window[i][1], window[i + 1][0]];
    if (g[1] - g[0] > gap[1] - gap[0]) gap = g;
  }
  return {switchStep, holdEnd, release: gap[0], recontact: gap[1], fallStart, fallEnd, cabinet, table, instructions: policy.map((e) => e.instruction ?? '')};
};

/**
 * The drag of the plain chain, as the run recorded it: the contact that spans the switch
 * (`holdEnd`), the first sustained contact after it (`resume`, at least `sustain` steps, up to
 * `contactEnd`), the step the bowl's goal is lost for good (`leave`) and the drop to the table
 * (`fallEnd`). Between `holdEnd` and `resume` the recorded contact only flickers (the fingers
 * opening against the bowl).
 */
export type Drag = Handoff & {resume: number; contactEnd: number; leave: number};

export const dragMoments = (t: HandoffTrace, sustain = 10): Drag => {
  const m = handoffMoments(t);
  const sustained = t.touch.find(([a, b]) => a >= m.holdEnd && b - a >= sustain);
  const resume = sustained ? sustained[0] : m.holdEnd;
  const contactEnd = sustained ? sustained[1] : m.holdEnd;
  const lost = t.goal.filter(([a]) => a < m.fallEnd);
  const leave = lost.length ? lost[lost.length - 1][1] : m.switchStep;
  return {...m, resume, contactEnd, leave};
};

/** Whether `step` lies in one of the recorded [from, to) intervals (no joining). */
export const recordedAt = (intervals: [number, number][] | undefined, step: number): boolean =>
  (intervals ?? []).some(([a, b]) => step >= a && step < b);

/**
 * The practice layout (start) a practice episode ran on: its job's recorded per-start list,
 * indexed by the episode's number within the job (e000103 is job j0001's third episode).
 * Null when the episode does not resolve, so a label can drop the number rather than guess.
 */
export const practiceStartOf = (film: FilmData, job: 'r1' | 'r2', episode: string | undefined): number | null => {
  if (!episode || !/^e\d{6}$/.test(episode)) return null;
  const jobId = `j${episode.slice(1, 5)}`;
  const idx = Number(episode.slice(5)) - 1;
  const p = (film.programs[job] ?? []).find((x) => x.job === jobId && x.role !== 'confirm');
  return p && idx >= 0 && idx < p.starts.length ? p.starts[idx] : null;
};

/** A clip id if film-data has it (optionally only with an overlay track), else the fallback. */
export const clipOr = (film: FilmData, id: string, fallback: string, needTrack = false): string => {
  const c = film.clips[id];
  return c && (!needTrack || c.track) ? id : fallback;
};
