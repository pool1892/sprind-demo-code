// r2-fix: set the cheese aside. The saved program as seven steps, each lit while the recorded
// run (practice start 0) is in it: the grab on the hero camera, the slide and the release from
// above (tilted, so the cheese stays in sight under the hand), a hold on the cheese on the table
// and the plate's cleared path (drawn from the plate's recorded start and end of its push), then
// the plate and the bowl on the hero camera, fast-forwarded (marked on screen) so the program
// completes on the voice; the winner's dots; then the job-2 final exam: the agent's job, then
// the plain chain, every tile a recorded run; rescued and broken small.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useBeat, useCue, useFilm, useNumber, type Clip} from '../../data';
import {color, FPS, shadow} from '../../theme';
import {CLIPS, PROGRAM_STEPS} from './clips';
import {Chip, JobStage, Label, OutcomeDots, TrackedPin, Verdict, keyed, objectBox, orderKeys, programStarts, ramp, stepOfKeys, useArrive, useTrackFile, type Key, type TrackFile} from './common';
import {clipOr} from './moments';
import {EXAM, ExamPanel, VERDICT_RIPPLE} from './exam';
import {DotLegend} from './DotLegend';

const FOOT = {x: 120, y: 214, w: 1040, h: 585};
const CHEESE = 'cream_cheese_1';
const ramp01 = (p: number) => Math.max(0, Math.min(1, (p - 0.6) / 0.4));
const PLATE = 'plate_1';
const COL = {x: 1220, y: 214, w: 580};

const StepChip: React.FC<{text: React.ReactNode; state: 'idle' | 'active' | 'done'; arrive: number}> = ({text, state, arrive}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        height: 60,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '0 22px 0 14px',
        borderRadius: 16,
        background: color.card,
        boxShadow: state === 'active' ? `0 0 0 4px ${color.agent}, ${shadow.card}` : shadow.card,
        opacity: arrive * (state === 'idle' ? 0.6 : 1),
        transform: `translateX(${(1 - arrive) * 24}px)`,
      }}
    >
      {state === 'done' ? (
        <Verdict ok size={34} style={{boxShadow: 'none'}} />
      ) : (
        <div style={{width: 34, height: 34, borderRadius: '50%', flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `inset 0 0 0 4px ${state === 'active' ? color.agent : color.line}`}}>
          {state === 'active' ? <div style={{width: 14, height: 14, borderRadius: '50%', background: color.agent, opacity: 0.55 + 0.45 * Math.sin(frame / 4)}} /> : null}
        </div>
      )}
      <div style={{fontSize: 28, fontWeight: state === 'active' ? 760 : 600, color: color.ink, whiteSpace: 'nowrap'}}>{text}</div>
    </div>
  );
};

/**
 * From above, after the release: the cheese labelled where the hand set it down, and the plate's
 * path drawn from the plate's recorded position now to where the recorded push left it, as wide
 * as the plate; the cheese sits outside it.
 */
const ClearPath: React.FC<{
  ctx: {step: number; toBox: (x: number, y: number) => {x: number; y: number}; scale: number; box: {w: number; h: number}};
  track: TrackFile | null;
  plateEndStep: number;
  asideIn: number;
  pathIn: number;
}> = ({ctx, track, plateEndStep, asideIn, pathIn}) => {
  const cheese = objectBox(track, ctx.step, CHEESE);
  const plate = objectBox(track, ctx.step, PLATE);
  const plateEnd = objectBox(track, plateEndStep, PLATE);
  let path: React.ReactNode = null;
  let label: React.ReactNode = null;
  if (plate && plateEnd && pathIn > 0) {
    const a = ctx.toBox(plate.cx, plate.cy);
    const b0 = ctx.toBox(plateEnd.cx, plateEnd.cy);
    const b = {x: a.x + (b0.x - a.x) * pathIn, y: a.y + (b0.y - a.y) * pathIn};
    const w = Math.max(plate.w, plate.h) * ctx.scale;
    const ang = Math.atan2(b0.y - a.y, b0.x - a.x);
    const head = 26;
    const hx = (sgn: number) => b.x - Math.cos(ang) * head + sgn * Math.sin(ang) * head * 0.7;
    const hy = (sgn: number) => b.y - Math.sin(ang) * head - sgn * Math.cos(ang) * head * 0.7;
    path = (
      <svg style={{position: 'absolute', left: 0, top: 0, width: ctx.box.w, height: ctx.box.h, overflow: 'visible'}}>
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="rgba(18,162,108,0.22)" strokeWidth={w} strokeLinecap="butt" />
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={color.ok} strokeWidth={6} strokeDasharray="16 12" strokeLinecap="round" />
        {pathIn > 0.95 ? (
          // the arrow head only: no circle around the plate or its destination
          <path d={`M${hx(1)},${hy(1)} L${b.x},${b.y} L${hx(-1)},${hy(-1)}`} fill="none" stroke={color.ok} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
        ) : null}
      </svg>
    );
    // the label under the end of the path, kept inside the frame
    const lw = 340;
    const lx = Math.max(16, Math.min(ctx.box.w - lw - 16, b0.x - lw / 2));
    const ly = Math.min(ctx.box.h - 72, b0.y + w / 2 + 14);
    label = (
      <div style={{position: 'absolute', left: lx, top: ly, opacity: ramp01(pathIn)}}>
        <Chip tone="ok" size={28}>
          the plate's way is clear
        </Chip>
      </div>
    );
  }
  return (
    <>
      {path}
      <TrackedPin toBox={ctx.toBox} box={cheese} dx={-330} dy={-110} tone="light" show={asideIn}>
        cheese set aside
      </TrackedPin>
      {label}
    </>
  );
};

/** Where to centre the overhead push-in: the cheese, the plate and where the plate's push ends. */
const pathFocus = (track: TrackFile | null, clip: Clip | null, step: number, plateEndStep: number): {x: number; y: number} => {
  const pts = [objectBox(track, step, CHEESE), objectBox(track, step, PLATE), objectBox(track, plateEndStep, PLATE)].filter((b): b is NonNullable<typeof b> => b !== null);
  if (!clip || pts.length === 0) return {x: 0.5, y: 0.5};
  const x = pts.reduce((a, b) => a + b.cx, 0) / pts.length;
  const y = pts.reduce((a, b) => a + b.cy, 0) / pts.length;
  return {x: x / clip.width, y: y / clip.height};
};

export const R2Fix: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const last = beat.frames - 1;
  const graspSteps = useNumber('r2_grasp_steps');
  const slideCm = useNumber('r2_slide_cm');
  const nSteps = useNumber('r2_steps');
  const k = useNumber('r2_relocate_k');
  const n = useNumber('r2_relocate_n');
  const winner = (film.programs.r2 ?? []).find((p) => p.role === 'winner');

  const answerF = useCue('answer', 0.03);
  const grabF = useCue('grab', 0.1);
  const asideF = useCue('aside', 0.16);
  const plateF = useCue('plate', 0.24);
  const bowlF = useCue('bowl', 0.3);
  const examF = useCue('exam', 0.47);
  const chainF = useCue('chain', 0.65);

  // the run's program, from its recorded track (else the run's event table)
  const track = useTrackFile(CLIPS.r2Relocate);
  const starts = programStarts(track, PROGRAM_STEPS.r2Relocate);
  const at = (i: number) => starts[Math.min(i, starts.length - 1)];
  const clip = useClip(CLIPS.r2Relocate);
  const end = clip ? (clip.frames - 1) * clip.frame_every : at(6) + 100;
  // from above for the slide and the release (the framed overhead clip of the same run, when it
  // has a track), then a hold on the cleared path; the hero camera for the rest
  const topId = clipOr(film, CLIPS.r2AsideTop, CLIPS.r2Relocate, true);
  const topClip = useClip(topId);
  const topTrack = useTrackFile(topId);
  const topFrom = grabF;
  const topTo = plateF + 8; // hold on the cleared path into "Then the plate"
  const homeEnd = at(4) - 2;
  // the grab and the set-aside near real time on the voice; the plate and the bowl
  // fast-forwarded, so the program completes before the exam
  const doneF = Math.max(plateF + 60, examF - 12);
  const keys: Key[] = orderKeys(
    [
      [answerF - 9, 0],
      [topFrom, at(1) - 10],
      [asideF + 4, at(1) + 24],
      [asideF + 20, at(3) + 2],
      [topTo, homeEnd],
      [topTo + 2, at(4)],
      [Math.round(topTo + 2 + (doneF - topTo - 2) * 0.5), at(5)],
      [Math.round(topTo + 2 + (doneF - topTo - 2) * 0.62), at(6)],
      [doneF, end],
    ],
    last,
  );
  const play = keyed(clip, keys);
  // how fast the run plays at this frame, against real time (marked when well above it)
  const seg = keys.findIndex(([f], i) => i > 0 && frame < f);
  const rate = seg > 0 ? (keys[seg][1] - keys[seg - 1][1]) / Math.max(1, keys[seg][0] - keys[seg - 1][0]) / ((clip?.fps ?? 20) / FPS) : 0;
  const fastForward = rate > 2.5;
  const fromAbove = topId !== CLIPS.r2Relocate && frame >= topFrom && frame < topTo;
  const asideIn = ramp(frame, asideF - 8, asideF + 4);
  const pathIn = ramp(frame, asideF + 10, asideF + 22);
  // where the plate's push ends, from the run's track (the step the plate step finishes)
  const plateEndStep = at(5);
  const step = stepOfKeys(frame, keys);
  const active = starts.reduce((acc, s, i) => (step >= s ? i : acc), 0);
  const finished = step >= end - 1;

  const labels: React.ReactNode[] = [
    <>grab with the cheese skill · {graspSteps} steps</>,
    <>slide it {slideCm} cm aside</>,
    <>let go</>,
    <>home</>,
    <>push the plate to the stove front</>,
    <>home</>,
    <>the empty bowl onto the cabinet</>,
  ];

  const phase2 = ramp(frame, examF - 12, examF + 4);
  const enter = useArrive(0);
  const headIn = useArrive(answerF);
  const winIn = useArrive(bowlF + 24, 'lively');

  return (
    <JobStage job={2} step={frame >= examF - 4 ? 3 : 2}>
      <div style={{position: 'absolute', inset: 0, opacity: 1 - phase2}}>
        <div style={{position: 'absolute', left: FOOT.x, top: 136, opacity: headIn * (1 - ramp(frame, examF - 16, examF - 8)), transform: `translateY(${(1 - headIn) * 16}px)`}}>
          <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>Set the cheese aside</div>
        </div>
        <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y, opacity: enter, transform: `scale(${0.97 + 0.03 * enter})`}}>
          {fromAbove ? (
            <Footage
              clip={topId}
              segments={play.segments}
              startAt={play.startAt}
              width={FOOT.w}
              height={FOOT.h}
              zoom={{from: 1.3, to: 1.45, at: [topFrom, topTo], focus: pathFocus(topTrack, topClip, homeEnd, plateEndStep)}}
              label={<Chip size={26}>from above · the same run</Chip>}
            >
              {(ctx) => <ClearPath ctx={ctx} track={topTrack} plateEndStep={plateEndStep} asideIn={asideIn} pathIn={pathIn} />}
            </Footage>
          ) : (
            <Footage clip={CLIPS.r2Relocate} segments={play.segments} startAt={play.startAt} width={FOOT.w} height={FOOT.h} label={<Chip size={26}>the saved program · practice layout</Chip>}>
              {() => (
                <>
                  <div style={{position: 'absolute', right: 22, top: 22, opacity: fastForward ? 1 : 0}}>
                    <Chip tone="dark" size={26}>
                      ▸▸ fast-forward
                    </Chip>
                  </div>
                  <div style={{position: 'absolute', right: 22, bottom: 22, opacity: finished ? 1 : 0}}>
                    <Chip ok size={28}>
                      delivered
                    </Chip>
                  </div>
                </>
              )}
            </Footage>
          )}
        </div>
        <div style={{position: 'absolute', left: COL.x, top: COL.y - 52}}>
          <Label>the saved program · {nSteps} steps</Label>
        </div>
        <div style={{position: 'absolute', left: COL.x, top: COL.y, width: COL.w, display: 'flex', flexDirection: 'column', gap: 12}}>
          {labels.map((t, i) => (
            <StepChip key={i} text={t} state={step >= at(i) && i < active ? 'done' : i === active && !finished ? 'active' : finished ? 'done' : 'idle'} arrive={ramp(frame, answerF + i * 3, answerF + i * 3 + 10)} />
          ))}
        </div>
        <div style={{position: 'absolute', left: COL.x, top: COL.y + 7 * 72 + 22, opacity: Math.min(1, winIn)}}>
          <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
            <OutcomeDots outcomes={winner?.outcomes ?? ''} start={bowlF + 20} size={26} gap={8} />
            <span style={{fontSize: 44, fontWeight: 840, color: color.ink, fontVariantNumeric: 'tabular-nums', transform: `scale(${0.8 + 0.2 * Math.min(1.1, winIn)})`, display: 'inline-block'}}>
              {k} of {n}
            </span>
          </div>
          <div style={{fontSize: 24, fontWeight: 600, color: color.ink2, marginTop: 8}}>on the practice layouts</div>
        </div>
        <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y + FOOT.h + 30, opacity: Math.min(1, winIn)}}>
          <DotLegend small />
        </div>
      </div>

      {frame >= examF - 14 ? (
        <div style={{position: 'absolute', inset: 0, opacity: phase2}}>
          <div style={{position: 'absolute', left: EXAM.x, top: 116, opacity: ramp(frame, examF, examF + 10)}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>The final exam</div>
          </div>
          <ExamPanel
            job="r2"
            appearAt={examF - 12}
            chainPlay={[examF - 8, Math.max(30, chainF - 40 - (examF - 8))]}
            jobPlay={[examF - 12, 22]}
            chainVerdictAt={chainF - 36}
            jobVerdictAt={examF + 12}
            footerAt={Math.min(last - 18, chainF - 36 + VERDICT_RIPPLE + 12)}
            dim="chain"
          />
        </div>
      ) : null}
    </JobStage>
  );
};
