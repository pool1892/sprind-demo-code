// Shared building blocks: the stage, type, entrance motion, pills, cards, arrows.
import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing} from 'remotion';
import {color, font, shadow, space, spring as springs, type, durations} from '../theme';

/** Progress 0..1 of an element arriving at `delay` frames (a calm spring). */
export const useEnter = (delay = 0, config: keyof typeof springs = 'calm'): number => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: springs[config], durationInFrames: durations.enter * 2});
};

/** Progress 0..1 of a linear-eased move between two frames. */
export const useBetween = (from: number, to: number, ease: (t: number) => number = Easing.inOut(Easing.cubic)): number => {
  const frame = useCurrentFrame();
  return interpolate(frame, [from, to], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
};

/** Fades and rises into place at `delay`; optionally leaves at `exit`. */
export const Appear: React.FC<{
  delay?: number;
  exit?: number;
  rise?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
  pop?: boolean;
}> = ({delay = 0, exit, rise = 26, style, children, pop}) => {
  const frame = useCurrentFrame();
  const p = useEnter(delay, pop ? 'lively' : 'calm');
  const out = exit === undefined ? 0 : interpolate(frame, [exit, exit + 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const scale = pop ? interpolate(p, [0, 1], [0.85, 1]) : 1;
  return (
    <div style={{opacity: Math.min(p, 1) * (1 - out), transform: `translateY(${(1 - p) * rise}px) scale(${scale})`, ...style}}>
      {children}
    </div>
  );
};

/** The paper stage every scene sits on, with an optional act label top left. */
export const Stage: React.FC<{act?: string; dark?: boolean; children: React.ReactNode}> = ({act, dark, children}) => (
  <AbsoluteFill
    style={{
      background: dark ? color.night : `radial-gradient(120% 90% at 30% 20%, #FAF8F3 0%, ${color.paper} 55%, #EFEBE1 100%)`,
      fontFamily: font.sans,
      color: dark ? '#fff' : color.ink,
    }}
  >
    {act ? (
      <div style={{position: 'absolute', left: space.marginX, top: 52, ...type.kicker, fontSize: 22, color: dark ? '#7E8494' : color.ink3}}>{act}</div>
    ) : null}
    {children}
  </AbsoluteFill>
);

export const Kicker: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div style={{...type.kicker, ...style}}>{children}</div>
);

export const Headline: React.FC<{size?: 'display' | 'h1' | 'h2' | 'h3'; children: React.ReactNode; style?: React.CSSProperties}> = ({
  size = 'h1',
  children,
  style,
}) => <div style={{...type[size], ...style}}>{children}</div>;

export const Body: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <div style={{...type.body, ...style}}>{children}</div>
);

export type Tone = 'plain' | 'ok' | 'bad' | 'lead' | 'agent' | 'teal' | 'dark';
const toneColors: Record<Tone, {bg: string; fg: string}> = {
  plain: {bg: color.card, fg: color.ink},
  ok: {bg: color.ok, fg: '#fff'},
  bad: {bg: color.bad, fg: '#fff'},
  lead: {bg: color.lead, fg: '#fff'},
  agent: {bg: color.agent, fg: '#fff'},
  teal: {bg: color.teal, fg: '#fff'},
  dark: {bg: color.night, fg: '#fff'},
};

export const Pill: React.FC<{tone?: Tone; children: React.ReactNode; style?: React.CSSProperties; size?: number}> = ({
  tone = 'plain',
  children,
  style,
  size = 28,
}) => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 10,
      padding: `${size * 0.32}px ${size * 0.72}px`,
      borderRadius: 999,
      background: toneColors[tone].bg,
      color: toneColors[tone].fg,
      fontSize: size,
      fontWeight: 680,
      boxShadow: shadow.pill,
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {children}
  </div>
);

export const Card: React.FC<{children: React.ReactNode; style?: React.CSSProperties; dark?: boolean; accent?: string}> = ({
  children,
  style,
  dark,
  accent,
}) => (
  <div
    style={{
      background: dark ? color.night : color.card,
      color: dark ? '#fff' : color.ink,
      borderRadius: space.radius,
      boxShadow: shadow.card,
      padding: '28px 34px',
      borderTop: accent ? `14px solid ${accent}` : undefined,
      ...style,
    }}
  >
    {children}
  </div>
);

/** A hand-drawn-feeling arrow that draws itself from `delay` over `length` frames. */
export const Arrow: React.FC<{
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  bend?: number; // perpendicular offset of the control point, px
  delay?: number;
  length?: number;
  stroke?: string;
  width?: number;
  dashed?: boolean;
}> = ({x1, y1, x2, y2, bend = 0, delay = 0, length = 18, stroke = color.ink, width = 5, dashed}) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame, [delay, delay + length], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const cx = mx - (dy / len) * bend;
  const cy = my + (dx / len) * bend;
  const d = `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
  // tangent at the end for the head
  const tx = x2 - cx;
  const ty = y2 - cy;
  const tl = Math.hypot(tx, ty) || 1;
  const ux = tx / tl;
  const uy = ty / tl;
  const head = 18;
  const hx1 = x2 - ux * head - uy * head * 0.6;
  const hy1 = y2 - uy * head + ux * head * 0.6;
  const hx2 = x2 - ux * head + uy * head * 0.6;
  const hy2 = y2 - uy * head - ux * head * 0.6;
  const total = len * 1.2;
  return (
    <svg style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none'}}>
      <path
        d={d}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeLinecap="round"
        strokeDasharray={dashed ? '14 12' : `${total}`}
        strokeDashoffset={dashed ? 0 : total * (1 - p)}
        opacity={dashed ? p : 1}
      />
      {p > 0.95 ? <path d={`M${hx1},${hy1} L${x2},${y2} L${hx2},${hy2}`} fill="none" stroke={stroke} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round" /> : null}
    </svg>
  );
};

/** A number that counts up from `from` to `to` between two frames. */
export const CountUp: React.FC<{to: number; from?: number; start: number; frames?: number; style?: React.CSSProperties; suffix?: string}> = ({
  to,
  from = 0,
  start,
  frames = 20,
  style,
  suffix = '',
}) => {
  const frame = useCurrentFrame();
  const v = interpolate(frame, [start, start + frames], [from, to], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  return <span style={{fontVariantNumeric: 'tabular-nums', ...style}}>{Math.round(v)}{suffix}</span>;
};
