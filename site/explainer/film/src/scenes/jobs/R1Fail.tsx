// r1-fail: the plain chain on practice layout 2, one of the agent's own first runs. The bowl
// counts as placed and the drawer skill starts while the fingers still hold the bowl; the
// fingers open against it, the hand keeps hold and drags it off the cabinet. The label on the
// bowl and the "recorded in the run" panel read one flag, the run's recorded contact at the
// step on screen, never smoothed: the label shows only where the record has sustained contact
// (it hides while the contact merely flickers as the fingers open). The label follows the
// bowl through the clip's overlay track; no ring is drawn on the bowl. No
// wrist inset: the wrist camera exists only for another run.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useBeat, useCue, useFilm} from '../../data';
import {color, shadow} from '../../theme';
import {CLIPS} from './clips';
import {Chip, JobStage, Label, TrackedPin, Verdict, keyed, objectBox, orderKeys, ramp, stepOfKeys, touchingAt, useArrive, useTrackFile, type Key} from './common';
import {dragMoments, practiceStartOf, recordedAt, type HandoffTrace} from './moments';

const BOWL = 'akita_black_bowl_1';
const FOOT = {x: 120, y: 136, w: 1240, h: 698};
const COL = {x: 1410, y: 150, w: 400};

type Phase = 'none' | 'holding' | 'flicker' | 'dragging' | 'falling' | 'fallen';

/** What the label on the bowl says in each phase (shown only while the record has contact). */
const PHASE_LABEL: Record<Phase, string | null> = {
  none: null,
  holding: 'fingers still on the bowl',
  flicker: null,
  dragging: 'drags it off',
  falling: null,
  fallen: 'dragged off the cabinet',
};

/** Whether the run recorded goal `i` holding at this step (from the clip's track). */
const goalAt = (track: {frames: {step: number; goals?: boolean[]}[]}, step: number, i: number): boolean => {
  const fr = track.frames;
  if (fr.length === 0) return false;
  const every = fr.length > 1 ? Math.max(1, fr[1].step - fr[0].step) : 1;
  const k = Math.max(0, Math.min(fr.length - 1, Math.round((step - fr[0].step) / every)));
  return Boolean(fr[k].goals?.[i]);
};

const SkillChip: React.FC<{text: string; state: 'idle' | 'active' | 'done'; arrive: number}> = ({text, state, arrive}) => {
  const frame = useCurrentFrame();
  const active = state === 'active';
  return (
    <div
      style={{
        opacity: arrive,
        transform: `translateY(${(1 - arrive) * 20}px)`,
        background: color.card,
        borderRadius: 20,
        boxShadow: active ? `0 0 0 4px ${color.agent}, ${shadow.card}` : shadow.card,
        padding: '16px 24px 16px 18px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        whiteSpace: 'nowrap',
      }}
    >
      {state === 'done' ? (
        <Verdict ok size={40} style={{boxShadow: 'none'}} />
      ) : (
        <div style={{width: 40, height: 40, borderRadius: '50%', flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `inset 0 0 0 4px ${active ? color.agent : color.line}`}}>
          {active ? <div style={{width: 16, height: 16, borderRadius: '50%', background: color.agent, opacity: 0.55 + 0.45 * Math.sin(frame / 4)}} /> : null}
        </div>
      )}
      <div style={{fontSize: 28, fontWeight: 650, color: color.ink}}>“{text}”</div>
    </div>
  );
};

/** One recorded flag of the run at the step on screen. */
const Readout: React.FC<{label: string; on: boolean; kind: 'contact' | 'goal'}> = ({label, on, kind}) => {
  const frame = useCurrentFrame();
  const pulse = 0.75 + 0.25 * Math.sin(frame / 3);
  return (
    <div style={{display: 'flex', alignItems: 'center', gap: 16, height: 58}}>
      {kind === 'contact' ? (
        <div
          style={{
            width: 34,
            height: 34,
            flex: 'none',
            borderRadius: '50%',
            background: on ? color.bad : 'transparent',
            boxShadow: on ? `0 0 ${18 * pulse}px ${color.bad}` : `inset 0 0 0 4px ${color.line}`,
            transform: `scale(${on ? 0.9 + 0.12 * pulse : 1})`,
          }}
        />
      ) : (
        <Verdict ok={on} size={34} style={{boxShadow: 'none'}} />
      )}
      <div style={{fontSize: 28, fontWeight: on ? 700 : 520, color: on ? color.ink : color.ink2}}>{label}</div>
    </div>
  );
};

export const R1Fail: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const trace = film.traces['r1-practice2-chain'] as HandoffTrace;
  const m = dragMoments(trace);
  const layout = practiceStartOf(film, 'r1', trace.episode);
  const last = beat.frames - 1;

  const chainF = useCue('chain', 0.04);
  const switchF = useCue('switch', 0.24);
  const holdingF = useCue('holding', 0.49);
  const dragF = useCue('drag', 0.62);

  // the run's own steps, on the narration: the placement up to the switch, a near hold while
  // the banner is read, slow motion while the fingers still hold the bowl, the flicker of the
  // opening fingers at real time, the hand holding again, the drag at real time, the drop at
  // half speed, and the bowl held on the table
  const startF = Math.max(4, switchF - 80);
  const flickerF = Math.max(holdingF + 22, dragF - 26);
  const resumeF = flickerF + Math.round(((m.resume - (m.holdEnd - 0.5)) / 20) * 30);
  const leaveF = resumeF + Math.round(((m.leave - m.resume) / 20) * 30);
  const dropF = leaveF + Math.round(((m.contactEnd - m.leave) / 20) * 30);
  const landF = dropF + Math.round(((m.fallEnd + 4 - m.contactEnd) / 20) * 30 * 2);
  const keys: Key[] = orderKeys(
    [
      [startF, Math.max(0, m.switchStep - 70)],
      [switchF, m.switchStep],
      [switchF + 24, m.switchStep + 1.5],
      [holdingF, m.holdEnd - 6],
      [flickerF, m.holdEnd - 0.5],
      [resumeF, m.resume],
      [leaveF, m.leave],
      [dropF, m.contactEnd],
      [landF, m.fallEnd + 4],
      [last, m.fallEnd + 10],
    ],
    last,
  );
  const clip = useClip(CLIPS.r1ChainPractice2);
  const track = useTrackFile(CLIPS.r1ChainPractice2);
  const play = keyed(clip, keys);
  const step = stepOfKeys(frame, keys);

  // one flag for the label and the panel: the recorded contact at the step on screen
  const atStep = Math.round(step);
  const contact = track ? touchingAt(track, atStep, BOWL) : recordedAt(trace.touch, atStep);
  const goalNow = track ? goalAt(track, atStep, 0) : recordedAt(trace.goal, atStep);
  const phase: Phase =
    atStep < m.switchStep
      ? 'none'
      : atStep < m.holdEnd
        ? 'holding'
        : atStep < m.resume
          ? 'flicker'
          : atStep < m.leave
            ? 'holding'
            : atStep < m.contactEnd
              ? 'dragging'
              : atStep < m.fallEnd
                ? 'falling'
                : 'fallen';
  const label = PHASE_LABEL[phase];
  const labelShown = label !== null && (phase === 'fallen' || contact);

  const enter = useArrive(0);
  const titleIn = useArrive(chainF);
  const chip1In = useArrive(chainF + 4);
  const chip2In = useArrive(chainF + 10);
  const bannerIn = useArrive(switchF - 4, 'lively');
  const bannerOut = ramp(frame, holdingF - 6, holdingF + 8);
  const pinIn = useArrive(holdingF);
  const readIn = useArrive(switchF + 6);
  const fallIn = useArrive(landF - 6, 'lively');

  const state1 = step >= m.switchStep ? 'done' : 'active';
  const state2 = step >= m.switchStep ? 'active' : 'idle';

  return (
    <JobStage job={1} step={0}>
      {/* the run, large */}
      <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y, opacity: enter, transform: `scale(${0.97 + 0.03 * enter})`}}>
        <Footage
          clip={CLIPS.r1ChainPractice2}
          segments={play.segments}
          startAt={play.startAt}
          width={FOOT.w}
          height={FOOT.h}
          zoom={{from: 1, to: 1.2, at: [switchF - 24, switchF + 24], focus: {x: 0.45, y: 0.5}}}
          label={<Chip size={26}>{layout === null ? 'the plain chain · a practice layout' : `the plain chain · practice layout ${layout}`}</Chip>}
        >
          {(ctx) => {
            const box = objectBox(track, ctx.step, BOWL);
            const pinUp = phase !== 'fallen';
            return (
              labelShown ? (
                <TrackedPin toBox={ctx.toBox} box={box} dx={pinUp ? -440 : -470} dy={pinUp ? -150 : 40} tone="bad" show={phase === 'fallen' ? fallIn : pinIn}>
                  {label}
                </TrackedPin>
              ) : null
            );
          }}
        </Footage>
        {/* the switch, as the run recorded it */}
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: 96,
            display: 'flex',
            justifyContent: 'center',
            opacity: Math.min(1, bannerIn) * (1 - bannerOut),
            transform: `translateY(${(1 - Math.min(1, bannerIn)) * -16}px) scale(${0.9 + 0.1 * Math.min(1, bannerIn)})`,
          }}
        >
          <div style={{display: 'flex', alignItems: 'center', gap: 16, background: color.night, color: '#fff', borderRadius: 999, padding: '14px 32px 14px 16px', fontSize: 32, fontWeight: 700, boxShadow: shadow.pill}}>
            <Verdict ok size={46} style={{boxShadow: 'none'}} />
            bowl on cabinet
            <span style={{color: '#8A90A0', fontWeight: 500}}>→</span>
            next skill: open the drawer
          </div>
        </div>
      </div>

      {/* the chain: two skills, one after the other */}
      <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y + FOOT.h + 34, display: 'flex', alignItems: 'center', gap: 18}}>
        <SkillChip text={m.instructions[0] ?? ''} state={state1} arrive={chip1In} />
        <div style={{fontSize: 40, color: color.ink3, opacity: chip2In}}>→</div>
        <SkillChip text={m.instructions[1] ?? ''} state={state2} arrive={chip2In} />
      </div>

      {/* what the run recorded */}
      <div style={{position: 'absolute', left: COL.x, top: COL.y, width: COL.w}}>
        <div style={{opacity: titleIn, transform: `translateY(${(1 - titleIn) * 20}px)`}}>
          <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.05, color: color.ink}}>The plain chain</div>
          <div style={{fontSize: 32, fontWeight: 480, color: color.ink2, marginTop: 10, lineHeight: 1.3}}>the two skills, one after the other</div>
        </div>
        <div
          style={{
            marginTop: 44,
            padding: '20px 26px',
            borderRadius: 22,
            background: color.card,
            boxShadow: shadow.card,
            opacity: readIn,
            transform: `translateY(${(1 - readIn) * 20}px)`,
          }}
        >
          <Label style={{marginBottom: 8}}>recorded in the run</Label>
          <Readout kind="contact" label="hand on the bowl" on={contact} />
          <Readout kind="goal" label="bowl on the cabinet" on={goalNow} />
        </div>
      </div>
    </JobStage>
  );
};
