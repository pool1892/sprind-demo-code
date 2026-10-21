// The agent's own words, the loop, the agent glyph.
import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {color, font, shadow, space} from '../theme';

/** A dark card with the agent's words, verbatim, typed on from `start` at `cps` characters a second. */
export const AgentNote: React.FC<{
  text: string;
  start: number;
  label?: string;
  cps?: number;
  width?: number;
  fontSize?: number;
  style?: React.CSSProperties;
}> = ({text, start, label = "agent's note", cps = 55, width = 640, fontSize = 30, style}) => {
  const frame = useCurrentFrame();
  const n = Math.max(0, Math.floor(((frame - start) / 30) * cps));
  const shown = text.slice(0, n);
  const visible = interpolate(frame, [start - 8, start], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <div style={{width, background: color.night, color: '#fff', borderRadius: space.radius, boxShadow: shadow.card, padding: '26px 32px', opacity: visible, ...style}}>
      <div style={{fontSize: 20, fontWeight: 750, letterSpacing: '0.14em', textTransform: 'uppercase', color: '#8A90A0', marginBottom: 14}}>{label}</div>
      <div style={{fontFamily: font.mono, fontSize, lineHeight: 1.45}}>
        “{shown}
        {n < text.length ? <span style={{opacity: frame % 16 < 8 ? 1 : 0}}>▍</span> : '”'}
      </div>
    </div>
  );
};

/** The agent glyph: an indigo disc with a spark. */
export const AgentGlyph: React.FC<{size?: number; style?: React.CSSProperties}> = ({size = 160, style}) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: '50%',
      background: color.agent,
      color: '#fff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: size * 0.46,
      boxShadow: `0 0 0 ${size * 0.11}px rgba(75,92,240,0.15), ${shadow.card}`,
      ...style,
    }}
  >
    ✦
  </div>
);

export const LOOP_STAGES = ['watch it fail', 'measure why', 'change one thing', 'prove it', 'keep it'] as const;
export type LoopStage = 0 | 1 | 2 | 3 | 4;
const stageColor = [color.bad, color.agent, color.agent, color.ok, color.lead];

/**
 * The improvement loop. Large: the full diagram, drawing itself from `start`. Small: a corner
 * badge with the `active` stage lit.
 */
export const Loop: React.FC<{size?: number; active?: LoopStage | null; start?: number; labels?: boolean; style?: React.CSSProperties}> = ({
  size = 560,
  active = null,
  start = 0,
  labels = true,
  style,
}) => {
  const frame = useCurrentFrame();
  const r = size * 0.36;
  const c = size / 2;
  const angles = LOOP_STAGES.map((_, i) => -Math.PI / 2 + (i * 2 * Math.PI) / LOOP_STAGES.length);
  const draw = interpolate(frame, [start, start + 40], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const circ = 2 * Math.PI * r;
  return (
    <div style={{position: 'relative', width: size, height: size, ...style}}>
      <svg width={size} height={size} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
        <circle cx={c} cy={c} r={r} fill="none" stroke={color.line} strokeWidth={size * 0.012} />
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke={color.agent}
          strokeWidth={size * 0.012}
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - draw)}
          transform={`rotate(-90 ${c} ${c})`}
          strokeLinecap="round"
        />
        {angles.map((a, i) => {
          const appear = interpolate(frame, [start + i * 8, start + i * 8 + 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
          const on = active === null || active === i;
          return (
            <circle
              key={i}
              cx={c + r * Math.cos(a)}
              cy={c + r * Math.sin(a)}
              r={size * (active === i ? 0.042 : 0.03) * appear}
              fill={on ? stageColor[i] : color.line}
            />
          );
        })}
      </svg>
      {labels
        ? angles.map((a, i) => {
            const appear = interpolate(frame, [start + i * 8 + 4, start + i * 8 + 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
            const x = c + (r + size * 0.1) * Math.cos(a);
            const y = c + (r + size * 0.1) * Math.sin(a);
            const on = active === null || active === i;
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: x,
                  top: y,
                  transform: `translate(${Math.cos(a) < -0.2 ? '-100%' : Math.cos(a) > 0.2 ? '0%' : '-50%'}, -50%)`,
                  fontSize: size * 0.05,
                  fontWeight: 750,
                  color: on ? color.ink : color.ink3,
                  whiteSpace: 'nowrap',
                  opacity: appear,
                }}
              >
                {LOOP_STAGES[i]}
              </div>
            );
          })
        : null}
      <div style={{position: 'absolute', left: c, top: c, transform: 'translate(-50%,-50%)'}}>
        <AgentGlyph size={size * 0.2} />
      </div>
    </div>
  );
};

/** The corner badge: a small loop with the current stage named beside it. */
export const LoopBadge: React.FC<{active: LoopStage}> = ({active}) => (
  <div style={{position: 'absolute', right: 70, top: 40, display: 'flex', alignItems: 'center', gap: 14}}>
    <div style={{fontSize: 24, fontWeight: 750, color: color.ink2, textTransform: 'uppercase', letterSpacing: '0.1em'}}>{LOOP_STAGES[active]}</div>
    <Loop size={92} active={active} labels={false} />
  </div>
);
