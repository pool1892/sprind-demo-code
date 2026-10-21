// The two experiment cards, introduced in `question` and reused as chapter cards: copy or
// import this component for the chapter openers of the other lanes.
//
//   <ExperimentCard which={1} />                    a full card (560 x 360 by default)
//   <ExperimentCard which={2} width={420} />        smaller, everything scales with width
//   <ExperimentTag which={1} />                     a one-line chapter tag for a corner
//
// The fleet icon draws the fleet's robots in its sites from film-data (b_robots, b_sites).
import React from 'react';
import {useNumber} from '../../data';
import {color, shadow} from '../../theme';

export const EXPERIMENTS = {
  1: {kicker: 'Experiment 1', title: 'A new job from existing skills', accent: color.ok},
  2: {kicker: 'Experiment 2', title: 'Faults planted in a fleet', accent: color.lead},
} as const;

/** Flat icon: a bowl going onto a cabinet with an open drawer. */
const JobIcon: React.FC<{size: number}> = ({size}) => (
  <svg width={size * 1.6} height={size} viewBox="0 -4 160 104">
    <rect x={62} y={30} width={70} height={66} rx={6} fill={color.ink} />
    <rect x={62} y={52} width={84} height={18} rx={4} fill="#3A3D46" />
    <rect x={92} y={58} width={24} height={5} rx={2.5} fill="#C9CBD3" />
    <rect x={72} y={80} width={24} height={5} rx={2.5} fill="#C9CBD3" />
    <path d="M72,6 Q97,44 122,6 Z" fill="#7C8796" />
    <ellipse cx={97} cy={6} rx={25} ry={4} fill="#9AA4B2" />
    <path d="M14,64 C26,40 40,26 58,18" fill="none" stroke={color.ok} strokeWidth={5} strokeLinecap="round" strokeDasharray="2 10" />
    <circle cx={14} cy={70} r={8} fill={color.ok} />
  </svg>
);

/** Flat icon: the fleet, small arms grouped by site. */
const FleetIcon: React.FC<{size: number}> = ({size}) => {
  const robots = Number(useNumber('b_robots')) || 0;
  const sites = Math.max(1, Number(useNumber('b_sites')) || 1);
  const per = Math.ceil(robots / sites);
  const arm = size * 0.34;
  return (
    <div style={{display: 'flex', gap: size * 0.18, alignItems: 'flex-end'}}>
      {Array.from({length: sites}, (_, s) => (
        <div key={s} style={{display: 'grid', gridTemplateColumns: `repeat(2, ${arm}px)`, gap: size * 0.05, padding: size * 0.05, borderRadius: size * 0.08, background: color.paper2}}>
          {Array.from({length: Math.min(per, robots - s * per)}, (_, i) => (
            <svg key={i} width={arm} height={arm} viewBox="0 0 40 40">
              <rect x={8} y={32} width={24} height={6} rx={2} fill={color.ink2} />
              <path d="M20,32 L16,18 L28,10" fill="none" stroke={color.ink} strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />
              <circle cx={28} cy={10} r={4} fill={color.lead} />
            </svg>
          ))}
        </div>
      ))}
    </div>
  );
};

export const ExperimentCard: React.FC<{
  which: 1 | 2;
  width?: number;
  style?: React.CSSProperties;
  /** Why this experiment, one short line under the title (the question beat), and its arrival. */
  reason?: string;
  reasonShow?: number;
}> = ({which, width = 560, style, reason, reasonShow = 1}) => {
  const e = EXPERIMENTS[which];
  const k = width / 560;
  return (
    <div
      style={{
        width,
        boxSizing: 'border-box',
        background: color.card,
        borderRadius: 26 * k,
        boxShadow: shadow.card,
        borderTop: `${12 * k}px solid ${e.accent}`,
        padding: `${30 * k}px ${36 * k}px ${34 * k}px`,
        ...style,
      }}
    >
      <div style={{fontSize: Math.max(24, 26 * k), fontWeight: 780, letterSpacing: '0.12em', textTransform: 'uppercase', color: e.accent}}>{e.kicker}</div>
      <div style={{fontSize: 46 * k, fontWeight: 790, letterSpacing: '-0.015em', lineHeight: 1.12, color: color.ink, marginTop: 12 * k, minHeight: 104 * k}}>{e.title}</div>
      {reason ? (
        <div style={{fontSize: 31 * k, fontWeight: 640, lineHeight: 1.25, color: color.ink2, marginTop: 12 * k, minHeight: 78 * k, opacity: Math.min(1, reasonShow * 1.3), transform: `translateY(${(1 - Math.min(1, reasonShow)) * 10}px)`}}>
          {reason}
        </div>
      ) : null}
      <div style={{marginTop: 22 * k, height: (reason ? 96 : 150) * k, display: 'flex', alignItems: 'flex-end'}}>
        {which === 1 ? <JobIcon size={(reason ? 90 : 140) * k} /> : <FleetIcon size={(reason ? 96 : 150) * k} />}
      </div>
    </div>
  );
};

/** A one-line chapter tag: "EXPERIMENT 1 · A new job from existing skills". */
export const ExperimentTag: React.FC<{which: 1 | 2; style?: React.CSSProperties}> = ({which, style}) => {
  const e = EXPERIMENTS[which];
  return (
    <div style={{display: 'inline-flex', alignItems: 'center', gap: 14, padding: '10px 22px', borderRadius: 999, background: color.card, boxShadow: shadow.pill, whiteSpace: 'nowrap', ...style}}>
      <div style={{width: 14, height: 14, borderRadius: '50%', background: e.accent}} />
      <div style={{fontSize: 24, fontWeight: 780, letterSpacing: '0.1em', textTransform: 'uppercase', color: e.accent}}>{e.kicker}</div>
      <div style={{fontSize: 26, fontWeight: 650, color: color.ink}}>{e.title}</div>
    </div>
  );
};
