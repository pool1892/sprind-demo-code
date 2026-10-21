// Small helpers for the opening scenes: pure progress functions (safe inside loops), a
// tick mark, a padlock, a hand-placed pin for clips without a track, a schematic table
// layout (a diagram, never footage) and a loop that keeps every stage it has reached lit.
import React from 'react';
import {Easing, interpolate, spring} from 'remotion';
import {color, durations, font, shadow, spring as springs} from '../../theme';
import {AgentGlyph, LOOP_STAGES} from '../../components/agent';
import type {OverlayContext} from '../../components/footage';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

/** 0..1 arrival at `at` with the film's spring; a pure function, usable in loops. */
export const arrive = (frame: number, fps: number, at: number, cfg: keyof typeof springs = 'calm'): number =>
  spring({frame: frame - at, fps, config: springs[cfg], durationInFrames: durations.enter * 2});

/** 0..1 eased progress between two frames. */
export const span = (frame: number, a: number, b: number, ease: (t: number) => number = Easing.inOut(Easing.cubic)): number =>
  interpolate(frame, [a, Math.max(a + 1, b)], [0, 1], {...clamp, easing: ease});

export const mix = (t: number, a: number, b: number) => a + (b - a) * t;

/** A round tick (or cross) badge. */
export const Mark: React.FC<{ok?: boolean; size?: number; style?: React.CSSProperties; scale?: number}> = ({ok = true, size = 44, style, scale = 1}) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: '50%',
      flex: 'none',
      background: ok ? color.ok : color.bad,
      color: '#fff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: size * 0.58,
      fontWeight: 800,
      transform: `scale(${scale})`,
      opacity: Math.min(1, scale * 1.4),
      ...style,
    }}
  >
    {ok ? '✓' : '✕'}
  </div>
);

/** A flat padlock drawing. `closed` 0..1 lowers the shackle. */
export const Padlock: React.FC<{size?: number; tone?: string; closed?: number; style?: React.CSSProperties}> = ({size = 64, tone = color.ink, closed = 1, style}) => {
  const lift = (1 - closed) * size * 0.18;
  return (
    <svg width={size} height={size * 1.15} viewBox="0 0 100 115" style={{overflow: 'visible', ...style}}>
      <path
        d={`M28,${52 - lift / (size / 100)} V34 a22,22 0 0 1 44,0 V${52 - lift / (size / 100)}`}
        fill="none"
        stroke={tone}
        strokeWidth={12}
        strokeLinecap="round"
      />
      <rect x={12} y={48} width={76} height={62} rx={14} fill={tone} />
      <circle cx={50} cy={74} r={8} fill="#fff" opacity={0.9} />
      <rect x={46} y={76} width={8} height={18} rx={4} fill="#fff" opacity={0.9} />
    </svg>
  );
};

/**
 * A label pinned to a point of a clip, placed by hand in the clip's pixel space (for clips
 * without an overlay track): a dot on the object, a short stem, a chip.
 */
export const HandPin: React.FC<{
  ctx: OverlayContext;
  x: number;
  y: number;
  dx?: number;
  dy?: number;
  show: number;
  children: React.ReactNode;
}> = ({ctx, x, y, dx = 30, dy = -56, show, children}) => {
  if (show <= 0) return null;
  const p = ctx.toBox(x, y);
  const lx = p.x + dx;
  const ly = p.y + dy;
  return (
    <div style={{position: 'absolute', inset: 0, opacity: Math.min(1, show)}}>
      <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible'}}>
        <line x1={p.x} y1={p.y} x2={mix(Math.min(1, show), p.x, lx)} y2={mix(Math.min(1, show), p.y, ly)} stroke="#fff" strokeWidth={4} strokeLinecap="round" />
        <circle cx={p.x} cy={p.y} r={9 * Math.min(1.1, show)} fill={color.ink} stroke="#fff" strokeWidth={4} />
      </svg>
      <div
        style={{
          position: 'absolute',
          left: lx,
          top: ly,
          transform: `translate(${dx < 0 ? '-100%' : '0%'}, -50%) scale(${0.85 + 0.15 * Math.min(1, show)})`,
          transformOrigin: dx < 0 ? 'right center' : 'left center',
          background: '#fff',
          color: color.ink,
          padding: '9px 20px',
          borderRadius: 999,
          fontSize: 28,
          fontWeight: 720,
          whiteSpace: 'nowrap',
          boxShadow: shadow.pill,
        }}
      >
        {children}
      </div>
    </div>
  );
};

/** Deterministic pseudo-random in [0,1) from an integer seed. */
const rand = (seed: number) => {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * A schematic, top-down table layout: a flat diagram of a starting arrangement (stove,
 * bowl, plate, cheese, cabinet), jittered by `seed`. It explains "layout"; it is not footage.
 */
export const LayoutSketch: React.FC<{seed: number; size: number; style?: React.CSSProperties}> = ({seed, size, style}) => {
  const j = (k: number, amp: number) => (rand(seed * 7 + k) - 0.5) * amp;
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" style={{borderRadius: size * 0.12, background: '#E9DFCC', boxShadow: shadow.pill, ...style}}>
      <rect x={70} y={12} width={22} height={76} rx={3} fill="#3A3330" />
      <rect x={10 + j(1, 6)} y={12 + j(2, 6)} width={22} height={18} rx={3} fill="#B9BBC0" />
      <circle cx={21 + j(1, 6)} cy={21 + j(2, 6)} r={6} fill="none" stroke="#6F737C" strokeWidth={2.5} />
      <circle cx={44 + j(3, 18)} cy={44 + j(4, 16)} r={8.5} fill="#7C8796" />
      <circle cx={44 + j(5, 18)} cy={70 + j(6, 14)} r={10} fill="#fff" stroke="#D9696C" strokeWidth={2.5} />
      <rect x={18 + j(7, 10)} y={56 + j(8, 18)} width={8} height={12} rx={1.5} fill="#3C5BA8" />
    </svg>
  );
};

const stageColor = [color.bad, color.agent, color.agent, color.ok, color.lead];

/**
 * The improvement loop, every stage reached so far lit (`lit` = how many, 0..5), the newest
 * one (`current`) enlarged. Draws itself from `start`. A local variant of the shared `Loop`,
 * which lights one stage at a time.
 */
export const LoopLit: React.FC<{
  size: number;
  frame: number;
  start: number;
  lit: number[]; // per stage, 0..1 how lit
  current: number | null;
  labelOpacity?: number;
}> = ({size, frame, start, lit, current, labelOpacity = 1}) => {
  const r = size * 0.36;
  const c = size / 2;
  const angles = LOOP_STAGES.map((_, i) => -Math.PI / 2 + (i * 2 * Math.PI) / LOOP_STAGES.length);
  const draw = interpolate(frame, [start, start + 40], [0, 1], {...clamp, easing: Easing.inOut(Easing.cubic)});
  const circ = 2 * Math.PI * r;
  // the progress arc follows the lit stages around the circle
  const reached = lit.reduce((s, v) => s + v, 0);
  const arc = Math.min(1, Math.max(0, (reached - 1) / LOOP_STAGES.length + (reached >= 5 ? 0.2 : 0)));
  return (
    <div style={{position: 'relative', width: size, height: size}}>
      <svg width={size} height={size} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke={color.line}
          strokeWidth={size * 0.014}
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - draw)}
          transform={`rotate(-90 ${c} ${c})`}
          strokeLinecap="round"
        />
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke={color.agent}
          strokeWidth={size * 0.014}
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - arc)}
          transform={`rotate(-90 ${c} ${c})`}
          strokeLinecap="round"
        />
        {angles.map((a, i) => {
          const appear = interpolate(frame, [start + i * 7, start + i * 7 + 10], [0, 1], clamp);
          const big = current === i ? 1 : 0;
          const rr = size * (0.03 + 0.016 * big + 0.006 * lit[i]) * appear;
          return (
            <g key={i}>
              {current === i ? <circle cx={c + r * Math.cos(a)} cy={c + r * Math.sin(a)} r={rr * 1.9} fill={stageColor[i]} opacity={0.16} /> : null}
              <circle cx={c + r * Math.cos(a)} cy={c + r * Math.sin(a)} r={rr} fill={lit[i] > 0.5 ? stageColor[i] : '#fff'} stroke={lit[i] > 0.5 ? stageColor[i] : color.line} strokeWidth={size * 0.008} />
            </g>
          );
        })}
      </svg>
      {angles.map((a, i) => {
        const appear = interpolate(frame, [start + i * 7 + 4, start + i * 7 + 14], [0, 1], clamp);
        const x = c + (r + size * 0.085) * Math.cos(a);
        const y = c + (r + size * 0.085) * Math.sin(a);
        const on = lit[i];
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x,
              top: y,
              transform: `translate(${Math.cos(a) < -0.2 ? '-100%' : Math.cos(a) > 0.2 ? '0%' : '-50%'}, ${Math.sin(a) < -0.9 ? '-100%' : '-50%'})`,
              fontSize: size * 0.054,
              fontWeight: 780,
              letterSpacing: '-0.01em',
              color: on > 0.5 ? color.ink : color.ink3,
              whiteSpace: 'nowrap',
              opacity: appear * labelOpacity * (0.4 + 0.6 * on),
            }}
          >
            {LOOP_STAGES[i]}
          </div>
        );
      })}
      <div style={{position: 'absolute', left: c, top: c, transform: 'translate(-50%,-50%)'}}>
        <AgentGlyph size={size * 0.2} />
      </div>
    </div>
  );
};

export const mono = font.mono;
