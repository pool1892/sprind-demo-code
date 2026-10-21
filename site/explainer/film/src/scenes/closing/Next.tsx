// next: an illustration (not a result): a packing cell with two two-armed
// mobile robots standing on the floor (wheels on the ground, soft shadows), one driving totes
// from the shelf to the packing table, the other boxes from the table to the packaging machine;
// request cards arrive; an "engineer hours per new job" bar with a question mark.
import React from 'react';
import {useCurrentFrame, Easing, interpolate} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue} from '../../data';
import {color, shadow} from '../../theme';
import {ramp} from './parts';

/** Where the floor line of the cell drawing lies: everything stands on it. */
const FLOOR = 484;
const ROBOT_TINT = '#3D4250';

/**
 * A two-armed mobile robot, flat, in a 100x120 box with its wheels' bottom at y = 118.
 * `carry` holds a tote or box between the hands; `roll` turns the wheels (driven distance).
 */
const Carrier: React.FC<{size: number; carry: string | null; lift: number; roll: number}> = ({size, carry, lift, roll}) => {
  const hold = carry !== null;
  // arms: forward and bent to hold a load in front of the torso, else hanging at the sides
  const handY = hold ? 66 - 6 * lift : 76;
  const left = hold ? `M30,36 L20,${56 - 4 * lift} L27,${handY}` : 'M30,36 L16,56 L13,74';
  const right = hold ? `M70,36 L80,${56 - 4 * lift} L73,${handY}` : 'M70,36 L84,56 L87,74';
  const wheel = (cx: number) => (
    <g transform={`translate(${cx} 112) rotate(${roll})`}>
      <circle r={6} fill={color.ink} />
      <line x1={-4.2} y1={0} x2={4.2} y2={0} stroke="#8E929C" strokeWidth={1.6} strokeLinecap="round" />
    </g>
  );
  return (
    <svg width={size} height={size * 1.2} viewBox="0 0 100 120" style={{overflow: 'visible', display: 'block'}}>
      <rect x={18} y={92} width={64} height={16} rx={6} fill={ROBOT_TINT} />
      {wheel(30)}
      {wheel(70)}
      <rect x={40} y={46} width={20} height={48} rx={5} fill={ROBOT_TINT} />
      <rect x={30} y={30} width={40} height={24} rx={8} fill={ROBOT_TINT} />
      <rect x={39} y={14} width={22} height={16} rx={6} fill={ROBOT_TINT} />
      <circle cx={46} cy={22} r={2.4} fill="#fff" />
      <circle cx={54} cy={22} r={2.4} fill="#fff" />
      {hold ? <rect x={27} y={handY - 16} width={46} height={22} rx={3} fill={carry} stroke="rgba(0,0,0,0.12)" strokeWidth={1.2} /> : null}
      <path d={left} stroke={ROBOT_TINT} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d={right} stroke={ROBOT_TINT} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
};

/**
 * One robot's shuttle between two stations, as a loop of `period` frames: load at `from`, drive
 * to `to` carrying, set the load down, drive back empty. Returns its position and load.
 */
const shuttle = (t: number, period: number, offset: number, from: number, to: number) => {
  const u = (((t + offset) % period) + period) % period / period;
  const ease = Easing.inOut(Easing.cubic);
  let x = from;
  let loaded = false;
  let lift = 0;
  if (u < 0.12) {
    x = from;
    loaded = u > 0.06;
    lift = interpolate(u, [0.06, 0.12], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  } else if (u < 0.5) {
    x = from + (to - from) * ease((u - 0.12) / 0.38);
    loaded = true;
  } else if (u < 0.62) {
    x = to;
    loaded = u < 0.56;
    lift = interpolate(u, [0.5, 0.56], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  } else {
    x = to + (from - to) * ease((u - 0.62) / 0.38);
  }
  return {x, loaded, lift};
};

/**
 * A packing cell, flat and illustrative, in a 1400x560 box: shelving with totes,
 * a packing table with boxes, a packaging machine with its outfeed, and two two-armed mobile
 * robots on the floor, each with a soft shadow, shuttling loads. `t` animates it.
 */
const PackingFloor: React.FC<{t: number; width?: number}> = ({t, width = 1400}) => {
  const s = width / 1400;
  const boxes = Array.from({length: 5}, (_, i) => (i * 150 + t * 1.6) % 750);
  const size = 200;
  const half = size / 2;
  const top = FLOOR - size * 1.2 * (118 / 120);
  const robots = [
    {...shuttle(t, 270, 0, 330, 560), carry: color.teal},
    {...shuttle(t, 270, 135, 760, 890), carry: color.wood},
  ];
  return (
    <div style={{position: 'relative', width: 1400 * s, height: 560 * s}}>
      <div style={{position: 'absolute', left: 0, top: 0, width: 1400, height: 560, transform: `scale(${s})`, transformOrigin: 'top left'}}>
        {/* floor */}
        <div style={{position: 'absolute', left: 0, right: 0, top: 452, height: 108, borderRadius: 24, background: color.paper2}} />
        {/* shelving with totes */}
        <div style={{position: 'absolute', left: 30, top: 150, width: 210, height: 322, borderRadius: 14, border: `6px solid ${color.ink3}`, boxSizing: 'border-box'}}>
          {[0, 1, 2].map((r) => (
            <div key={r} style={{position: 'absolute', left: 14, right: 14, top: 22 + r * 100, height: 64, borderRadius: 8, background: r === 1 ? color.teal : '#B9C3C9', opacity: 0.85}} />
          ))}
        </div>
        <div style={{position: 'absolute', left: 30, width: 210, top: 466, height: 14, borderRadius: '50%', background: 'rgba(21,23,28,0.10)', filter: 'blur(3px)'}} />
        {/* packing table with boxes */}
        <div style={{position: 'absolute', left: 300, top: 380, width: 420, height: 22, borderRadius: 6, background: color.wood}} />
        <div style={{position: 'absolute', left: 330, top: 402, width: 16, height: 70, background: color.wood}} />
        <div style={{position: 'absolute', left: 676, top: 402, width: 16, height: 70, background: color.wood}} />
        <div style={{position: 'absolute', left: 310, width: 400, top: 466, height: 12, borderRadius: '50%', background: 'rgba(21,23,28,0.10)', filter: 'blur(3px)'}} />
        {[0, 1, 2].map((i) => (
          <div key={i} style={{position: 'absolute', left: 330 + i * 130, top: 320, width: 100, height: 60, borderRadius: 6, background: color.wood, boxShadow: 'inset 0 -10px 0 rgba(0,0,0,0.12)'}} />
        ))}
        {/* packaging machine with its outfeed belt */}
        <div style={{position: 'absolute', left: 920, top: 190, width: 250, height: 282, borderRadius: 22, background: color.ink2}}>
          <div style={{position: 'absolute', left: 30, top: 34, width: 190, height: 90, borderRadius: 12, background: '#C9CDD6'}} />
          <div style={{position: 'absolute', left: 30, top: 150, width: 60, height: 18, borderRadius: 9, background: color.ok}} />
          <div style={{position: 'absolute', left: 100, top: 150, width: 60, height: 18, borderRadius: 9, background: '#8E929C'}} />
        </div>
        <div style={{position: 'absolute', left: 920, width: 250, top: 466, height: 14, borderRadius: '50%', background: 'rgba(21,23,28,0.12)', filter: 'blur(3px)'}} />
        <div style={{position: 'absolute', left: 1170, top: 410, width: 220, height: 26, borderRadius: 13, background: '#C9C2B3'}} />
        {boxes.map((x, i) => (x < 210 ? <div key={i} style={{position: 'absolute', left: 1175 + x, top: 366, width: 54, height: 44, borderRadius: 5, background: color.wood}} /> : null))}
        {/* two two-armed mobile robots, on the floor, shuttling loads */}
        {robots.map((r, i) => (
          <React.Fragment key={i}>
            <div style={{position: 'absolute', left: r.x - half * 0.72, top: FLOOR - 9, width: size * 0.72, height: 18, borderRadius: '50%', background: 'rgba(21,23,28,0.20)', filter: 'blur(4px)'}} />
            <div style={{position: 'absolute', left: r.x - half, top}}>
              <Carrier size={size} carry={r.loaded ? r.carry : null} lift={r.lift} roll={(r.x / (size * 0.06)) * (180 / Math.PI)} />
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

const REQUESTS = ['new product', 'new box size', 'returns'];

export const Next: React.FC = () => {
  const frame = useCurrentFrame();
  const tRobots = useCue('robots', 0.02);
  const tPacking = useCue('packing', 0.15);
  const tEngineer = useCue('engineer', 0.34);
  const tQuestion = useCue('question', 0.56);
  const tLess = useCue('less', 0.64);
  const head = ramp(frame, tRobots, tRobots + 14);
  const cell = ramp(frame, tPacking - 6, tPacking + 16, Easing.out(Easing.cubic));
  const bar = ramp(frame, tQuestion - 4, tQuestion + 14, Easing.out(Easing.cubic));
  // the "with agents" bar shrinks and breathes: an open question, not a measured value
  const shrink = ramp(frame, tLess, tLess + 40, Easing.inOut(Easing.cubic));
  const breathe = shrink >= 1 ? 0.06 * Math.sin((frame - tLess - 40) / 9) : 0;
  const withAgents = 1 - 0.55 * shrink + breathe;

  return (
    <Stage act="next">
      <div style={{position: 'absolute', left: 120, top: 104, display: 'flex', alignItems: 'center', gap: 28, opacity: head}}>
        <Headline size="h2">Next: real robots</Headline>
        <span style={{padding: '8px 20px', borderRadius: 999, border: `3px dashed ${color.lead}`, color: color.lead, fontSize: 28, fontWeight: 780}}>the plan</span>
      </div>
      <div style={{position: 'absolute', left: 120, top: 196, fontSize: 32, fontWeight: 520, color: color.ink2, opacity: head}}>
        a complete packing operation on two-armed mobile robots
      </div>

      {/* the cell, as an illustration */}
      <div style={{position: 'absolute', left: 100, top: 300, opacity: cell, transform: `translateY(${(1 - cell) * 30}px)`}}>
        <div style={{borderRadius: 30, background: 'rgba(255,255,255,0.55)', border: `3px dashed ${color.line}`, padding: 20}}>
          <PackingFloor t={frame} width={1160} />
        </div>
        <div style={{fontSize: 26, fontWeight: 600, color: color.ink3, marginTop: 14}}>illustration</div>
      </div>

      {/* request cards arrive */}
      {REQUESTS.map((r, i) => {
        const p = ramp(frame, tEngineer + i * 14, tEngineer + i * 14 + 22, Easing.out(Easing.cubic));
        return (
          <div
            key={r}
            style={{
              position: 'absolute',
              left: 1390 + (1 - p) * 260,
              top: 300 + i * 92,
              width: 400,
              height: 76,
              boxSizing: 'border-box',
              borderRadius: 18,
              background: color.lead,
              color: '#fff',
              fontSize: 32,
              fontWeight: 760,
              display: 'flex',
              alignItems: 'center',
              padding: '0 26px',
              boxShadow: shadow.card,
              opacity: p,
            }}
          >
            {r}
          </div>
        );
      })}

      {/* the measure: engineer hours per new job */}
      <div
        style={{
          position: 'absolute',
          left: 1390,
          top: 600,
          width: 400,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: 24,
          boxShadow: shadow.card,
          padding: '24px 26px',
          opacity: bar,
          transform: `translateY(${(1 - bar) * 30}px)`,
        }}
      >
        <div style={{fontSize: 28, fontWeight: 800, color: color.ink, lineHeight: 1.15}}>engineer hours per new job</div>
        <div style={{marginTop: 22}}>
          <div style={{fontSize: 24, fontWeight: 650, color: color.ink2}}>today</div>
          <div style={{height: 34, borderRadius: 10, background: color.ink2, marginTop: 6, width: '100%'}} />
        </div>
        <div style={{marginTop: 18}}>
          <div style={{fontSize: 24, fontWeight: 650, color: color.agent}}>with agents</div>
          <div style={{display: 'flex', alignItems: 'center', gap: 14, marginTop: 6}}>
            <div style={{height: 34, borderRadius: 10, border: `3px dashed ${color.agent}`, boxSizing: 'border-box', background: 'rgba(75,92,240,0.10)', width: `${78 * withAgents}%`}} />
            <span style={{fontSize: 48, fontWeight: 850, color: color.agent, lineHeight: 1}}>?</span>
          </div>
        </div>
        <div style={{fontSize: 24, fontWeight: 650, color: color.lead, marginTop: 16}}>to be measured on the hardware</div>
      </div>
    </Stage>
  );
};
