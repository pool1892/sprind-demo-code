// question: the intention, said once. A robot installation with its engineering loop; the
// specialist's seat in the loop is taken by the agent. Then the question as a card, its three
// verbs lighting (find out → change → prove). Then the two experiment cards side by side, each
// with its reason as the voice gives it.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {AgentGlyph} from '../../components/agent';
import {useCue} from '../../data';
import {color, shadow, type} from '../../theme';
import {ExperimentCard} from './ExperimentCard';
import {arrive, mix, span} from './kit';
import {Person, RobotCell} from './parts';
import {ENGINEERING_STEPS} from './Why';

const VERBS = ['find out', 'change', 'prove'];

/** The engineering loop around an installation: four steps on a ring and a seat on its left. */
const InstallationLoop: React.FC<{frame: number; fps: number; agentAt: number; start: number}> = ({frame, fps, agentAt, start}) => {
  const R = 300;
  const cx = 960;
  const cy = 560;
  const draw = span(frame, start, start + 36);
  const circ = 2 * Math.PI * R;
  const swap = arrive(frame, fps, agentAt, 'lively');
  const spin = (frame / 30) * 0.5; // the loop turns slowly
  const angles = ENGINEERING_STEPS.map((_, i) => -Math.PI / 2 + (i * Math.PI) / 2);
  const seat = {x: cx - R, y: cy};
  return (
    <>
      <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible'}}>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={color.line} strokeWidth={8} strokeDasharray={circ} strokeDashoffset={circ * (1 - draw)} transform={`rotate(-90 ${cx} ${cy})`} strokeLinecap="round" />
        {/* a running marker: the loop is worked through again and again */}
        <circle cx={cx + R * Math.cos(spin * 2 * Math.PI - Math.PI / 2)} cy={cy + R * Math.sin(spin * 2 * Math.PI - Math.PI / 2)} r={12} fill={swap > 0.5 ? color.agent : color.ink2} opacity={draw} />
      </svg>
      <div style={{position: 'absolute', left: cx - 170, top: cy - 120, opacity: draw}}>
        <RobotCell width={340} t={frame / 30} />
      </div>
      {angles.map((a, i) => {
        if (i === 3) return null;
        const p = span(frame, start + 10 + i * 8, start + 22 + i * 8);
        const x = cx + R * Math.cos(a);
        const y = cy + R * Math.sin(a);
        return (
          <div key={i} style={{position: 'absolute', left: x, top: y, transform: 'translate(-50%,-50%)', opacity: p, padding: '12px 24px', borderRadius: 999, background: color.card, boxShadow: shadow.pill, fontSize: 30, fontWeight: 720, color: color.ink, whiteSpace: 'nowrap'}}>
            {ENGINEERING_STEPS[i]}
          </div>
        );
      })}
      {/* the fourth step sits on the left; the seat is drawn just inside it */}
      <div style={{position: 'absolute', left: seat.x - 30, top: seat.y + 90, transform: 'translate(-50%,0)', opacity: span(frame, start + 34, start + 46), padding: '12px 24px', borderRadius: 999, background: color.card, boxShadow: shadow.pill, fontSize: 30, fontWeight: 720, color: color.ink, whiteSpace: 'nowrap'}}>
        {ENGINEERING_STEPS[3]}
      </div>
      <div style={{position: 'absolute', left: seat.x - 70, top: seat.y - 70, width: 140, height: 140}}>
        <div style={{position: 'absolute', inset: 0, opacity: (1 - Math.min(1, swap)) * draw, transform: `translateX(${-Math.min(1, swap) * 90}px)`}}>
          <Person size={140} />
        </div>
        <div style={{position: 'absolute', inset: 0, opacity: Math.min(1, swap * 1.5), transform: `translateY(${(1 - Math.min(1, swap)) * -160}px)`}}>
          <AgentGlyph size={140} />
        </div>
      </div>
    </>
  );
};

export const Question: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cMayfly = useCue('mayfly', 0.02);
  const cAgents = useCue('agents', 0.13);
  const cBefore = useCue('before', 0.19);
  const cQuestion = useCue('question', 0.38);
  const cVerbs = [useCue('findout', 0.41), useCue('change', 0.5), useCue('prove', 0.53)];
  const cTwo = useCue('two', 0.57);
  const cKinds = useCue('kinds', 0.6);
  const cOne = useCue('one', 0.64);
  const cWhy1 = useCue('why1', 0.7);
  const cFleet = useCue('fleet', 0.79);
  const cWhy2 = useCue('why2', 0.88);

  const phaseA = 1 - span(frame, cQuestion - 24, cQuestion - 6);
  const sim = arrive(frame, fps, cBefore + 10, 'lively');
  const q = arrive(frame, fps, cQuestion - 4);
  const up = span(frame, cTwo - 6, cTwo + 18);
  const kinds = arrive(frame, fps, cKinds);
  const e1 = arrive(frame, fps, cOne);
  const e2 = arrive(frame, fps, cFleet);
  // why these two, each reason on its words
  const why1 = arrive(frame, fps, cWhy1);
  const why2 = arrive(frame, fps, cWhy2);

  return (
    <Stage act="what we set out to test">
      {phaseA > 0.01 ? (
        <div style={{position: 'absolute', inset: 0, opacity: phaseA}}>
          <div style={{position: 'absolute', left: 0, right: 0, top: 110, textAlign: 'center', opacity: arrive(frame, fps, cMayfly)}}>
            <div style={{...type.h3, fontSize: 46}}>Robot installations, engineered by AI agents</div>
          </div>
          <InstallationLoop frame={frame} fps={fps} agentAt={cAgents} start={cMayfly + 6} />
          <div style={{position: 'absolute', left: 1430, top: 860, opacity: Math.min(1, sim * 1.4), transform: `scale(${0.85 + 0.15 * Math.min(1, sim)})`, transformOrigin: 'left center'}}>
            <div style={{padding: '14px 28px', borderRadius: 999, background: color.teal, color: '#fff', fontSize: 32, fontWeight: 740, boxShadow: shadow.pill, whiteSpace: 'nowrap'}}>tested first in simulation</div>
          </div>
        </div>
      ) : null}

      {/* the question */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: mix(up, 320, 100),
          transform: `translate(-50%, 0) scale(${mix(up, 1, 0.72)})`,
          transformOrigin: 'center top',
          opacity: q,
          width: 1300,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: 30,
          boxShadow: shadow.card,
          padding: '44px 56px 50px',
          textAlign: 'center',
        }}
      >
        <div style={{...type.kicker, marginBottom: 18}}>the question</div>
        <div style={{...type.h2, fontSize: 58}}>Can an AI agent, on its own, …</div>
        <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 26, marginTop: 36}}>
          {VERBS.map((v, i) => {
            const lit = arrive(frame, fps, cVerbs[i], 'lively');
            return (
              <React.Fragment key={v}>
                {i > 0 ? <div style={{fontSize: 48, color: color.ink3}}>→</div> : null}
                <div
                  style={{
                    padding: '18px 40px',
                    borderRadius: 999,
                    fontSize: 50,
                    fontWeight: 800,
                    background: lit > 0.5 ? color.agent : color.paper2,
                    color: lit > 0.5 ? '#fff' : color.ink3,
                    transform: `scale(${1 + 0.06 * Math.sin(Math.min(1, lit) * Math.PI)})`,
                  }}
                >
                  {v}
                </div>
              </React.Fragment>
            );
          })}
        </div>
        <div style={{...type.body, fontSize: 36, marginTop: 30, opacity: span(frame, cVerbs[2] + 6, cVerbs[2] + 18)}}>what needs to change, change it, and prove that it works</div>
      </div>

      {/* the two experiments */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 470, textAlign: 'center', ...type.kicker, fontSize: 28, opacity: up}}>
        two experiments<span style={{opacity: kinds}}> · one for each kind of engineering an installation needs</span>
      </div>
      <div style={{position: 'absolute', left: 300, top: 528, opacity: Math.min(1, e1 * 1.3), transform: `translateY(${(1 - Math.min(1, e1)) * 50}px)`}}>
        <ExperimentCard which={1} width={600} reason="every new product or rule asks for one" reasonShow={why1} />
      </div>
      <div style={{position: 'absolute', left: 1020, top: 528, opacity: Math.min(1, e2 * 1.3), transform: `translateY(${(1 - Math.min(1, e2)) * 50}px)`}}>
        <ExperimentCard which={2} width={600} reason="when working robots start to fail" reasonShow={why2} />
      </div>
    </Stage>
  );
};
