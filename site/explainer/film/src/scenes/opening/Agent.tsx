// agent: the engineer and its sandbox, condensed. One clean box with the
// agent glyph inside; three short groups that appear with the voice and stay small: what it
// sees and what it does (inside the box), what it cannot see (outside, behind a lock). The
// rules as one line on the box's lower edge. On the last two sentences one line under the
// box for the people, which gives way on "After that, no person took part" to the punchline,
// large and held to the end; the agent's own work takes its colour
// on "the agent chose every experiment". No arrows, no extra icons, nothing that moves
// without a word to move it.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {AgentGlyph} from '../../components/agent';
import {useCue} from '../../data';
import {color, shadow, type} from '../../theme';
import {arrive, Padlock, span} from './kit';

const BX = 150; // the sandbox
const BY = 220;
const BW = 1170;
const BH = 480;
const RX = 1380; // what it cannot see, outside the box
const RW = 400;

type Line = {text: string; at: number; sub?: {text: string; at: number}};

/** A small group: a heading and short lines, each line arriving on its own cue. */
const Group: React.FC<{
  heading: React.ReactNode;
  headingTone: string;
  headingAt: number;
  lines: Line[];
  frame: number;
  fps: number;
  ink?: string;
  style?: React.CSSProperties;
}> = ({heading, headingTone, headingAt, lines, frame, fps, ink = color.ink, style}) => {
  const h = arrive(frame, fps, headingAt - 4);
  return (
    <div style={style}>
      <div style={{...type.kicker, fontSize: 26, color: headingTone, opacity: h, display: 'flex', alignItems: 'center', gap: 14}}>{heading}</div>
      <div style={{display: 'flex', flexDirection: 'column', gap: 18, marginTop: 22}}>
        {lines.map((l) => {
          const p = arrive(frame, fps, l.at);
          const s = l.sub ? arrive(frame, fps, l.sub.at) : 0;
          return (
            <div key={l.text} style={{opacity: Math.min(1, p * 1.4), transform: `translateY(${(1 - Math.min(1, p)) * 14}px)`}}>
              <div style={{fontSize: 36, fontWeight: 700, color: ink, whiteSpace: 'nowrap'}}>{l.text}</div>
              {l.sub ? <div style={{fontSize: 27, fontWeight: 560, color: color.ink3, marginTop: 4, whiteSpace: 'nowrap', opacity: s}}>{l.sub.text}</div> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const Agent: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cSandbox = useCue('sandbox', 0.01);
  const cFrames = useCue('frames', 0.15);
  const cMeas = useCue('measurements', 0.2);
  const cHand = useCue('hand', 0.26);
  const cDoes = [useCue('checks', 0.37), useCue('programs', 0.41), useCue('experiments', 0.46)];
  const cRules = useCue('rules', 0.5);
  const cCannot = useCue('cannot', 0.55);
  const cHidden = [useCue('faults', 0.58), useCue('layouts', 0.64)];
  const cPeople = [useCue('humans', 0.73), useCue('wrote', 0.76), useCue('fixed', 0.79)];
  const cAfter = useCue('after', 0.82);
  const cSolutions = useCue('solutions', 0.87);

  const box = arrive(frame, fps, cSandbox - 6);
  const glyph = arrive(frame, fps, cSandbox + 6, 'lively');
  const rules = arrive(frame, fps, cRules);
  const notChangeable = arrive(frame, fps, cCannot);
  const hidden = arrive(frame, fps, cHidden[0] - 6);
  const lock = span(frame, cHidden[0], cHidden[0] + 10);
  const people = cPeople.map((c) => arrive(frame, fps, c));
  const after = span(frame, cAfter, cAfter + 18);
  const own = span(frame, cSolutions, cSolutions + 12) > 0.5;
  const punch = arrive(frame, fps, cAfter + 10);
  const chose = arrive(frame, fps, cSolutions);

  return (
    <Stage act="the engineer">
      {/* the sandbox */}
      <div
        style={{
          position: 'absolute',
          left: BX,
          top: BY,
          width: BW,
          height: BH,
          boxSizing: 'border-box',
          borderRadius: 36,
          background: 'rgba(255,255,255,0.72)',
          border: '4px solid rgba(75,92,240,0.40)',
          boxShadow: shadow.card,
          opacity: box,
          transform: `scale(${0.96 + 0.04 * box})`,
        }}
      >
        <div style={{position: 'absolute', left: 34, top: 26, ...type.kicker, fontSize: 24, color: color.agent}}>sandbox</div>
        <div style={{position: 'absolute', left: '50%', top: 40, transform: `translateX(-50%) scale(${Math.min(1.08, glyph)})`, opacity: Math.min(1, glyph * 1.5), display: 'flex', alignItems: 'center', gap: 22}}>
          <AgentGlyph size={100} />
          <div style={{fontSize: 40, fontWeight: 800, color: color.ink, whiteSpace: 'nowrap'}}>AI agent</div>
        </div>
        <Group
          style={{position: 'absolute', left: 70, top: 176}}
          heading="sees"
          headingTone={color.teal}
          headingAt={cFrames}
          frame={frame}
          fps={fps}
          lines={[
            {text: 'camera frames from its trials', at: cFrames},
            {text: 'measurements', at: cMeas, sub: {text: 'hand, objects, what touches what', at: cHand}},
          ]}
        />
        <Group
          style={{position: 'absolute', left: BW / 2 + 30, top: 176}}
          heading="does"
          headingTone={own ? color.agent : color.ink3}
          headingAt={cDoes[0]}
          frame={frame}
          fps={fps}
          ink={own ? color.agent : color.ink}
          lines={[
            {text: 'writes its own checks', at: cDoes[0]},
            {text: 'builds programs from skills', at: cDoes[1]},
            {text: 'runs experiments', at: cDoes[2]},
          ]}
        />
      </div>

      {/* the rules, one line on the box's lower edge */}
      <div style={{position: 'absolute', left: BX + BW / 2, top: BY + BH, transform: `translate(-50%, -50%) translateY(${(1 - Math.min(1, rules)) * 14}px)`, opacity: Math.min(1, rules * 1.4), padding: '14px 32px', borderRadius: 999, background: color.paper, boxShadow: shadow.pill, whiteSpace: 'nowrap', fontSize: 32, fontWeight: 700, color: color.ink}}>
        <span style={{color: color.ink3, fontWeight: 760}}>rules: </span>
        known
        <span style={{opacity: notChangeable}}>, not changeable</span>
      </div>

      {/* what it cannot see, outside the box, behind a lock */}
      <div style={{position: 'absolute', left: RX, top: BY + 140, width: RW, boxSizing: 'border-box', padding: '30px 34px 36px', borderRadius: 28, background: color.night, boxShadow: shadow.card, opacity: Math.min(1, hidden * 1.3), transform: `translateY(${(1 - Math.min(1, hidden)) * 20}px)`}}>
        <Group
          heading={
            <>
              <Padlock size={34} tone={color.lead} closed={lock} />
              <span>cannot see</span>
            </>
          }
          headingTone={color.nightText}
          headingAt={cHidden[0]}
          frame={frame}
          fps={fps}
          ink="#fff"
          lines={[
            {text: 'planted faults', at: cHidden[0]},
            {text: 'exam layouts', at: cHidden[1], sub: {text: 'until its solution is final', at: cHidden[1] + 24}},
          ]}
        />
      </div>

      {/* the people: one line, which fades when the agent takes over */}
      <div style={{position: 'absolute', left: BX + 70, top: BY + BH + 70, whiteSpace: 'nowrap', fontSize: 34, fontWeight: 650, color: color.ink2, opacity: Math.min(1, people[0] * 1.4) * (1 - after), transform: `translateY(${(1 - Math.min(1, people[0])) * 14}px)`}}>
        <span style={{color: color.lead, fontWeight: 780}}>People: </span>
        built the bench
        <span style={{opacity: people[1]}}>, wrote the requests</span>
        <span style={{opacity: people[2]}}>, fixed the grading</span>
      </div>

      {/* the punchline, in the people line's place, held to the end */}
      <div style={{position: 'absolute', left: BX, top: BY + BH + 62, width: 1640}}>
        <div style={{fontSize: 48, fontWeight: 820, letterSpacing: '-0.015em', color: color.ink, whiteSpace: 'nowrap', opacity: Math.min(1, punch * 1.3), transform: `translateY(${(1 - Math.min(1, punch)) * 16}px)`}}>
          After that, no person took part:
        </div>
        <div style={{fontSize: 48, fontWeight: 820, letterSpacing: '-0.015em', color: color.agent, whiteSpace: 'nowrap', marginTop: 6, opacity: Math.min(1, chose * 1.3), transform: `translateY(${(1 - Math.min(1, chose)) * 16}px)`}}>
          the agent chose every experiment and wrote every solution.
        </div>
      </div>
    </Stage>
  );
};
