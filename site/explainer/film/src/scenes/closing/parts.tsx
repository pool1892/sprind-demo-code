// Small drawn pieces shared by the closing scenes: flat icons (arm, data centre, hand,
// camera, clock), the lag badge, compact dot grids and the frame helpers. Everything here is
// schematic; footage comes only from recorded clips.
import React from 'react';
import {interpolate, useCurrentFrame, Easing} from 'remotion';
import {color, font, shadow} from '../../theme';

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

/** 0..1 between two frames, eased. */
export const ramp = (frame: number, a: number, b: number, ease: (t: number) => number = Easing.inOut(Easing.cubic)) =>
  interpolate(frame, [a, b], [0, 1], {...clamp, easing: ease});

/** A percentage from a recorded count, rounded as the records round it. */
export const pctOf = (k: number, n: number) => `${Math.round((100 * k) / Math.max(1, n))}%`;

/** A deterministic shuffle, so a grid of runs looks scattered but renders the same each frame. */
export const seeded = <T,>(items: T[], seed = 7): T[] => {
  const out = items.slice();
  let s = seed >>> 0;
  for (let i = out.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};

/** A flat robot arm on a base, for maps and installations. */
export const ArmIcon: React.FC<{size?: number; tint?: string; reach?: number}> = ({size = 90, tint = color.ink, reach = 0}) => {
  const a1 = -62 + reach * 10;
  const a2 = 70 - reach * 25;
  const r = (d: number) => (d * Math.PI) / 180;
  const x0 = 38;
  const y0 = 92;
  const x1 = x0 + 44 * Math.cos(r(a1));
  const y1 = y0 + 44 * Math.sin(r(a1));
  const x2 = x1 + 38 * Math.cos(r(a1 + a2));
  const y2 = y1 + 38 * Math.sin(r(a1 + a2));
  return (
    <svg width={size} height={size * 1.12} viewBox="0 0 100 112" style={{overflow: 'visible'}}>
      <rect x={16} y={94} width={44} height={14} rx={5} fill={tint} />
      <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={tint} strokeWidth={11} strokeLinecap="round" />
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={tint} strokeWidth={9} strokeLinecap="round" />
      <circle cx={x0} cy={y0} r={8} fill={tint} />
      <circle cx={x1} cy={y1} r={7} fill="#fff" stroke={tint} strokeWidth={4} />
      <path d={`M${x2 - 8},${y2 + 2} L${x2 - 8},${y2 + 14} M${x2 + 8},${y2 + 2} L${x2 + 8},${y2 + 14} M${x2 - 9},${y2 + 2} L${x2 + 9},${y2 + 2}`} stroke={tint} strokeWidth={5} strokeLinecap="round" fill="none" />
    </svg>
  );
};

/** A server rack: the customer's distant data centre. */
export const ServerIcon: React.FC<{size?: number; tint?: string; blink?: number}> = ({size = 150, tint = color.ink, blink = 0}) => (
  <svg width={size} height={size * 1.15} viewBox="0 0 100 115">
    {[0, 1, 2, 3].map((i) => (
      <g key={i}>
        <rect x={8} y={6 + i * 26} width={84} height={20} rx={5} fill={tint} />
        <circle cx={22} cy={16 + i * 26} r={3.6} fill={(i + Math.floor(blink)) % 2 === 0 ? color.ok : '#5A606E'} />
        <rect x={52} y={14 + i * 26} width={30} height={4} rx={2} fill="#5A606E" />
      </g>
    ))}
  </svg>
);

/** A small camera, for "after a look at the camera images". */
export const CameraIcon: React.FC<{size?: number; tint?: string}> = ({size = 44, tint = color.ink}) => (
  <svg width={size} height={size * 0.78} viewBox="0 0 50 39">
    <rect x={2} y={8} width={46} height={29} rx={6} fill={tint} />
    <rect x={15} y={2} width={18} height={9} rx={3} fill={tint} />
    <circle cx={25} cy={22} r={9} fill="#fff" />
    <circle cx={25} cy={22} r={5} fill={tint} />
  </svg>
);

/** A clock face, for the lag badge. */
export const ClockIcon: React.FC<{size?: number; tint?: string}> = ({size = 26, tint = '#fff'}) => (
  <svg width={size} height={size} viewBox="0 0 26 26">
    <circle cx={13} cy={13} r={10.5} fill="none" stroke={tint} strokeWidth={3} />
    <path d="M13 7 L13 13 L17.5 15.5" fill="none" stroke={tint} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

/** A padlock: the link has to stay. */
export const LockIcon: React.FC<{size?: number; tint?: string}> = ({size = 30, tint = color.ink}) => (
  <svg width={size} height={size * 1.2} viewBox="0 0 30 36">
    <path d="M8 16 V10 a7 7 0 0 1 14 0 V16" fill="none" stroke={tint} strokeWidth={4} />
    <rect x={3} y={15} width={24} height={19} rx={5} fill={tint} />
  </svg>
);

/** The same badge wherever the fleet lags. */
export const LagBadge: React.FC<{show: number; label?: string; tone?: string; style?: React.CSSProperties}> = ({show, label = 'lag', tone = color.night, style}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      padding: '8px 18px 8px 14px',
      borderRadius: 999,
      background: tone,
      color: '#fff',
      fontSize: 28,
      fontWeight: 760,
      boxShadow: shadow.pill,
      whiteSpace: 'nowrap',
      opacity: Math.min(1, show),
      transform: `scale(${0.7 + 0.3 * Math.min(1, show)})`,
      ...style,
    }}
  >
    <ClockIcon />
    {label}
  </div>
);

/** A gripper hand seen from above, for the probe diagram. */
export const HandIcon: React.FC<{size?: number; tint?: string}> = ({size = 110, tint = color.ink}) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <rect x={30} y={4} width={40} height={30} rx={8} fill={tint} />
    <rect x={18} y={30} width={64} height={16} rx={6} fill={tint} />
    <rect x={20} y={44} width={14} height={44} rx={6} fill={tint} />
    <rect x={66} y={44} width={14} height={44} rx={6} fill={tint} />
  </svg>
);

export type RunState = 'ok' | 'bad' | 'empty';

/** A compact grid of runs: one dot each, drawn plainly (no spring) so hundreds stay cheap. */
export const RunGrid: React.FC<{
  states: RunState[];
  perRow: number;
  size: number;
  gap: number;
  appear?: (i: number) => number; // 0..1 per dot
  pop?: (i: number) => number; // 0..1 extra scale pulse per dot
}> = ({states, perRow, size, gap, appear, pop}) => (
  <div style={{display: 'grid', gridTemplateColumns: `repeat(${perRow}, ${size}px)`, gap}}>
    {states.map((s, i) => {
      const a = appear ? appear(i) : 1;
      const p = pop ? pop(i) : 0;
      const fill = s === 'ok' ? color.ok : s === 'bad' ? color.bad : 'transparent';
      return (
        <div
          key={i}
          style={{
            width: size,
            height: size,
            borderRadius: '50%',
            background: fill,
            boxShadow: s === 'empty' ? `inset 0 0 0 ${Math.max(2, size * 0.14)}px ${color.line}` : 'none',
            opacity: a,
            transform: `scale(${(0.5 + 0.5 * a) * (1 + 0.35 * Math.sin(Math.PI * p))})`,
          }}
        />
      );
    })}
  </div>
);

/** k green then n-k red dots, filling from `start` (one row of runs with a recorded count). */
export const CountDots: React.FC<{k: number; n: number; start: number; perRow: number; size?: number; gap?: number; every?: number}> = ({
  k,
  n,
  start,
  perRow,
  size = 14,
  gap = 5,
  every = 0.5,
}) => {
  const frame = useCurrentFrame();
  const states: RunState[] = Array.from({length: n}, (_, i) => (frame >= start + i * every ? (i < k ? 'ok' : 'bad') : 'empty'));
  return <RunGrid states={states} perRow={perRow} size={size} gap={gap} appear={(i) => (frame >= start + i * every ? ramp(frame, start + i * every, start + i * every + 6) : 1)} />;
};

/** The big "a → b" of a result. */
export const FromTo: React.FC<{from: string; to: string; size?: number; tail?: string; show?: number}> = ({from, to, size = 96, tail, show = 1}) => (
  <div style={{display: 'flex', alignItems: 'baseline', gap: size * 0.22, fontFamily: font.sans, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap'}}>
    <span style={{fontSize: size * 0.72, fontWeight: 760, color: color.bad}}>{from}</span>
    <span style={{fontSize: size * 0.6, fontWeight: 600, color: color.ink3, opacity: show}}>→</span>
    <span style={{fontSize: size, fontWeight: 840, color: color.ok, letterSpacing: '-0.03em', opacity: show}}>{to}</span>
    {tail ? <span style={{fontSize: size * 0.36, fontWeight: 650, color: color.ink2, opacity: show}}>{tail}</span> : null}
  </div>
);

/** A small dark clock disc: this arm trails its commands. */
export const LagDot: React.FC<{show: number; size?: number}> = ({show, size = 40}) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: '50%',
      background: color.night,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      opacity: Math.min(1, show),
      transform: `scale(${0.6 + 0.4 * Math.min(1, show)})`,
      boxShadow: shadow.pill,
    }}
  >
    <ClockIcon size={size * 0.62} />
  </div>
);

/** A two-armed mobile robot, flat: wheeled base, torso, head, two arms. */
export const MobileRobot: React.FC<{size?: number; tint?: string; wave?: number}> = ({size = 200, tint = color.ink2, wave = 0}) => (
  <svg width={size} height={size * 1.2} viewBox="0 0 100 120" style={{overflow: 'visible'}}>
    <rect x={18} y={92} width={64} height={18} rx={6} fill={tint} />
    <circle cx={30} cy={112} r={6} fill={color.ink} />
    <circle cx={70} cy={112} r={6} fill={color.ink} />
    <rect x={40} y={46} width={20} height={48} rx={5} fill={tint} />
    <rect x={30} y={30} width={40} height={24} rx={8} fill={tint} />
    <rect x={39} y={14} width={22} height={16} rx={6} fill={tint} />
    <circle cx={46} cy={22} r={2.4} fill="#fff" />
    <circle cx={54} cy={22} r={2.4} fill="#fff" />
    <path d={`M30,36 L14,${56 + wave * 6} L10,${74 + wave * 4}`} stroke={tint} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" fill="none" />
    <path d={`M70,36 L86,${56 - wave * 6} L90,${74 - wave * 4}`} stroke={tint} strokeWidth={7} strokeLinecap="round" strokeLinejoin="round" fill="none" />
  </svg>
);

/**
 * A packing cell, flat and illustrative: a packaging machine, a table with
 * boxes, two two-armed mobile robots. Drawn in a 1400x560 box; `t` animates boxes and arms.
 */
export const PackingCell: React.FC<{t: number; width?: number}> = ({t, width = 1400}) => {
  const s = width / 1400;
  const boxes = Array.from({length: 5}, (_, i) => ((i * 150 + t * 1.6) % 750));
  return (
    <div style={{position: 'relative', width: 1400 * s, height: 560 * s}}>
      <div style={{position: 'absolute', left: 0, top: 0, width: 1400, height: 560, transform: `scale(${s})`, transformOrigin: 'top left'}}>
        {/* floor */}
        <div style={{position: 'absolute', left: 0, right: 0, top: 470, height: 90, borderRadius: 24, background: color.paper2}} />
        {/* shelving with totes */}
        <div style={{position: 'absolute', left: 30, top: 150, width: 210, height: 330, borderRadius: 14, border: `6px solid ${color.ink3}`, boxSizing: 'border-box'}}>
          {[0, 1, 2].map((r) => (
            <div key={r} style={{position: 'absolute', left: 14, right: 14, top: 22 + r * 100, height: 64, borderRadius: 8, background: r === 1 ? color.teal : '#B9C3C9', opacity: 0.85}} />
          ))}
        </div>
        {/* packing table with boxes */}
        <div style={{position: 'absolute', left: 300, top: 380, width: 420, height: 22, borderRadius: 6, background: color.wood}} />
        <div style={{position: 'absolute', left: 330, top: 402, width: 16, height: 70, background: color.wood}} />
        <div style={{position: 'absolute', left: 676, top: 402, width: 16, height: 70, background: color.wood}} />
        {[0, 1, 2].map((i) => (
          <div key={i} style={{position: 'absolute', left: 330 + i * 130, top: 320, width: 100, height: 60, borderRadius: 6, background: color.wood, boxShadow: 'inset 0 -10px 0 rgba(0,0,0,0.12)'}} />
        ))}
        {/* packaging machine with its outfeed belt */}
        <div style={{position: 'absolute', left: 920, top: 190, width: 250, height: 290, borderRadius: 22, background: color.ink2}}>
          <div style={{position: 'absolute', left: 30, top: 34, width: 190, height: 90, borderRadius: 12, background: '#C9CDD6'}} />
          <div style={{position: 'absolute', left: 30, top: 150, width: 60, height: 18, borderRadius: 9, background: color.ok}} />
          <div style={{position: 'absolute', left: 100, top: 150, width: 60, height: 18, borderRadius: 9, background: '#8E929C'}} />
        </div>
        <div style={{position: 'absolute', left: 1170, top: 410, width: 220, height: 26, borderRadius: 13, background: '#C9C2B3'}} />
        {boxes.map((x, i) => (x < 210 ? <div key={i} style={{position: 'absolute', left: 1175 + x, top: 366, width: 54, height: 44, borderRadius: 5, background: color.wood}} /> : null))}
        {/* two two-armed mobile robots */}
        <div style={{position: 'absolute', left: 400 + Math.sin(t / 40) * 30, top: 150}}>
          <MobileRobot size={240} tint="#3D4250" wave={Math.sin(t / 14)} />
        </div>
        <div style={{position: 'absolute', left: 670 - Math.sin(t / 50) * 20, top: 150}}>
          <MobileRobot size={240} tint="#3D4250" wave={Math.cos(t / 16)} />
        </div>
      </div>
    </div>
  );
};
