// why: skills are not jobs. Two graphics in turn, never both at full size.
// First, small and quick, with the first paragraph: skill tiles tick green and snap into a
// job; the seam cracks; change cards send it back to the start. Then, from "Turning skills
// into reliable work", the engineering loop alone, centred and at full size: each of the four
// steps arrives and lights exactly on its words (cues watch, cause, change, prove). On
// "specialists" a specialist takes the seat on the loop and works it round by hand, a clock
// and a cost meter running in its middle. The loop's geometry is the one `question` opens on.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {useBeat, useCue} from '../../data';
import {color, shadow, type} from '../../theme';
import {arrive, mix, span} from './kit';
import {Clock, CostMeter, Crack, JobTiles, Person, StatusPill} from './parts';

const SKILLS = ['pick up', 'place', 'push', 'open'];
export const ENGINEERING_STEPS = ['watch the failures', 'find the cause', 'change the system', 'prove it'];
const CHANGES = ['new product', 'new packing rule', 'a worn gripper'];

// graphic one: compact, in the middle of the frame
const G1X = 300;
const SKILL_Y = 370;
const JOB_Y = 550;
const JOB_W = 320;
const JOB_H = 96;
const CARD_X = 1350;

// graphic two: the loop, as `question` draws it (centre, radius, a seat on the left)
const CX = 960;
const CY = 560;
const R = 300;

/**
 * Where each step's pill sits: centred on the top and bottom of the ring; on the right and
 * left straddling the ring, mostly outside it, so the centre stays free (the left one moves
 * below the seat once it is taken). `ax` is the pill's horizontal anchor.
 */
const pillAt = (i: number, seated: number): {x: number; y: number; ax: number} =>
  [
    {x: CX, y: CY - R, ax: -50},
    {x: CX + R - 40, y: CY, ax: 0},
    {x: CX, y: CY + R, ax: -50},
    {x: CX - R + 40, y: mix(seated, CY, CY + 120), ax: -100},
  ][i];

export const Why: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const beat = useBeat();
  const cSkills = useCue('skills', 0.02);
  const cJob = useCue('job', 0.14);
  const cChanging = useCue('changing', 0.31);
  const cEng = useCue('engineering', 0.38);
  const cIsEng = useCue('isEng', 0.45);
  const cSteps = [useCue('watch', 0.5), useCue('cause', 0.56), useCue('change', 0.61), useCue('prove', 0.66)];
  const cSpec = useCue('specialists', 0.7);
  const cEvery = useCue('every', 0.78);
  const cSlow = useCue('slow', 0.89);

  // ---- graphic one: skills snap into a job, it cracks, changes send it back ----
  const g1 = 1 - span(frame, cEng - 2, cEng + 12);
  const crackAt = cJob + 40;
  const crack = span(frame, crackAt, crackAt + 14);
  const backAt = cChanging + 12;
  const status =
    frame >= backAt
      ? {tone: 'bad' as const, text: '↺ back to the start', at: backAt}
      : frame >= crackAt
        ? {tone: 'bad' as const, text: '✗ not reliable', at: crackAt}
        : null;

  // ---- graphic two: the engineering loop ----
  const g2 = span(frame, cEng + 2, cEng + 16);
  const draw = span(frame, cEng + 4, cEng + 40);
  const circ = 2 * Math.PI * R;
  // the progress arc reaches each step as it is named
  const arc = 0.25 * (span(frame, cSteps[1] - 8, cSteps[1] + 4) + span(frame, cSteps[2] - 8, cSteps[2] + 4) + span(frame, cSteps[3] - 8, cSteps[3] + 4));
  const seated = span(frame, cSpec - 2, cSpec + 14);
  const spec = arrive(frame, fps, cSpec);
  // by hand: a marker works round the loop, one step every `per` frames
  const t0 = cSpec + 14;
  const per = 20;
  const handStep = frame >= t0 ? Math.floor((frame - t0) / per) % ENGINEERING_STEPS.length : -1;
  const marker = span(frame, t0 - 6, t0 + 4);
  const mAngle = -Math.PI / 2 + (2 * Math.PI * Math.max(0, frame - t0)) / (per * ENGINEERING_STEPS.length);
  let stepNow = -1;
  cSteps.forEach((c, i) => {
    if (frame >= c) stepNow = i;
  });
  const slow = span(frame, cSlow, cSlow + 12);
  const run = span(frame, t0, beat.frames, (t) => t);
  const centreA = span(frame, cEng + 6, cEng + 20) * (1 - span(frame, cSpec - 10, cSpec + 2));
  const isEng = arrive(frame, fps, cIsEng);
  const every = arrive(frame, fps, cEvery);
  const slowP = arrive(frame, fps, cSlow, 'lively');

  return (
    <Stage act="the problem">
      {/* graphic one */}
      {g1 > 0.01 ? (
        <div style={{position: 'absolute', inset: 0, opacity: g1, transform: `scale(${mix(g1, 0.92, 1)})`, transformOrigin: '50% 50%'}}>
          <div style={{position: 'absolute', left: G1X, top: SKILL_Y - 46, ...type.kicker, opacity: arrive(frame, fps, cSkills - 6)}}>skills</div>
          <div style={{position: 'absolute', left: G1X, top: SKILL_Y, display: 'flex', gap: 20}}>
            {SKILLS.map((s, i) => {
              const p = arrive(frame, fps, cSkills + i * 5);
              const tick = arrive(frame, fps, cSkills + 16 + i * 6, 'lively');
              return (
                <div key={s} style={{width: 210, height: 82, borderRadius: 20, background: color.card, boxShadow: shadow.card, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 22px', boxSizing: 'border-box', opacity: p, transform: `translateY(${(1 - p) * 24}px)`}}>
                  <div style={{fontSize: 32, fontWeight: 740, color: color.ink}}>{s}</div>
                  <div style={{width: 38, height: 38, borderRadius: '50%', background: color.ok, color: '#fff', fontSize: 22, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', transform: `scale(${Math.min(1.15, tick)})`, opacity: Math.min(1, tick * 1.5)}}>✓</div>
                </div>
              );
            })}
          </div>
          <div style={{position: 'absolute', left: G1X, top: JOB_Y - 46, ...type.kicker, opacity: arrive(frame, fps, cJob - 4)}}>a job, done reliably</div>
          <JobTiles x={G1X} y={JOB_Y} at={cJob} crack={crack} w={JOB_W} h={JOB_H} fontSize={29} />
          <Crack x={G1X + JOB_W} y={JOB_Y} p={crack} height={JOB_H} />
          {status ? (
            <StatusPill key={status.at} x={G1X + JOB_W * 2 + 50} y={JOB_Y + JOB_H / 2} tone={status.tone} show={arrive(frame, fps, status.at, 'lively')}>
              {status.text}
            </StatusPill>
          ) : null}
          {CHANGES.map((c, i) => {
            const p = arrive(frame, fps, cChanging + i * 6, 'lively');
            const y = mix(Math.min(p, 1.1), -260, SKILL_Y + 20 + i * 104);
            return (
              <div key={c} style={{position: 'absolute', left: CARD_X + (i % 2) * 30, top: y, width: 300, boxSizing: 'border-box', padding: '18px 24px', borderRadius: 18, background: color.card, borderLeft: `10px solid ${color.lead}`, boxShadow: shadow.card, fontSize: 30, fontWeight: 740, color: color.ink, transform: `rotate(${[-3, 2.5, -1.5][i]}deg)`, opacity: p > 0 ? 1 : 0, whiteSpace: 'nowrap'}}>
                {c}
              </div>
            );
          })}
        </div>
      ) : null}

      {/* graphic two: the engineering loop, alone, full size */}
      {g2 > 0.01 ? (
        <div style={{position: 'absolute', inset: 0, opacity: g2}}>
          <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible'}}>
            <circle cx={CX} cy={CY} r={R} fill="none" stroke={color.line} strokeWidth={8} strokeDasharray={circ} strokeDashoffset={circ * (1 - draw)} transform={`rotate(-90 ${CX} ${CY})`} strokeLinecap="round" />
            <circle cx={CX} cy={CY} r={R} fill="none" stroke={color.ink2} strokeWidth={8} strokeDasharray={circ} strokeDashoffset={circ * (1 - arc)} transform={`rotate(-90 ${CX} ${CY})`} strokeLinecap="round" opacity={(stepNow >= 1 ? 1 : 0) * (1 - marker)} />
            {marker > 0.01 ? <circle cx={CX + R * Math.cos(mAngle)} cy={CY + R * Math.sin(mAngle)} r={14} fill={color.ink2} opacity={marker} /> : null}
          </svg>

          {/* the centre: what the loop is, then who works it and what that costs */}
          <div style={{position: 'absolute', left: CX - 240, top: CY - 110, width: 480, height: 220, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', opacity: centreA}}>
            <div style={{...type.body, fontSize: 32, color: color.ink2, lineHeight: 1.25}}>
              turning skills
              <br />
              into reliable work
            </div>
            <div style={{fontSize: 58, fontWeight: 820, letterSpacing: '-0.025em', color: color.ink, marginTop: 14, opacity: Math.min(1, isEng * 1.3), transform: `translateY(${(1 - Math.min(1, isEng)) * 14}px)`}}>is engineering</div>
          </div>
          <div style={{position: 'absolute', left: CX - 250, top: CY - 150, width: 500, height: 300, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18, textAlign: 'center', opacity: Math.min(1, spec * 1.3)}}>
            <div style={{fontSize: 36, fontWeight: 760, color: color.ink, lineHeight: 1.2}}>
              a specialist, by hand
              <div style={{opacity: Math.min(1, every * 1.3)}}>for every change</div>
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
              <Clock size={74} turns={run * 6} tone={slow > 0.5 ? color.bad : color.ink} />
              <CostMeter width={220} fill={run} tone={slow > 0.5 ? color.bad : color.ink} />
            </div>
            <div style={{fontSize: 38, fontWeight: 800, color: color.bad, whiteSpace: 'nowrap', opacity: Math.min(1, slowP * 1.4), transform: `scale(${0.9 + 0.1 * Math.min(1, slowP)})`}}>slow and expensive</div>
          </div>

          {/* the seat on the loop, taken by the specialist */}
          <div style={{position: 'absolute', left: CX - R - 70, top: CY - 70, opacity: Math.min(1, spec * 1.4), transform: `translateY(${(1 - Math.min(1, spec)) * 30}px)`}}>
            <Person size={140} />
          </div>

          {/* the four steps, each arriving and lighting on its words */}
          {ENGINEERING_STEPS.map((s, i) => {
            if (frame < cSteps[i] - 3) return null;
            const p = arrive(frame, fps, cSteps[i] - 3, 'lively');
            const lit = handStep >= 0 ? handStep === i : stepNow === i;
            const {x, y, ax} = pillAt(i, seated);
            return (
              <div
                key={s}
                style={{
                  position: 'absolute',
                  left: x,
                  top: y,
                  transform: `translate(${ax}%, -50%) scale(${(0.8 + 0.2 * Math.min(1.05, p)) * (lit ? 1.06 : 1)})`,
                  opacity: Math.min(1, p * 1.5),
                  padding: '14px 28px',
                  borderRadius: 999,
                  background: lit ? color.ink : color.card,
                  color: lit ? '#fff' : color.ink,
                  boxShadow: shadow.pill,
                  fontSize: 34,
                  fontWeight: 740,
                  whiteSpace: 'nowrap',
                  transformOrigin: `${-ax}% 50%`,
                }}
              >
                {s}
              </div>
            );
          })}
        </div>
      ) : null}
    </Stage>
  );
};
