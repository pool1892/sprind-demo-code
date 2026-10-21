// The dot motif: one dot is one recorded run. Dots fill in as the runs "land", green for
// delivered and red for not, then become the number.
import React from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {color, durations, font} from '../theme';
import type {Program} from '../data';

export const Dot: React.FC<{state: '1' | '0' | 'empty' | 'outline-ok' | 'outline'; size: number; appear: number; darkEmpty?: boolean}> = ({
  state,
  size,
  appear,
  darkEmpty,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = spring({frame: frame - appear, fps, config: {damping: 13, stiffness: 160, mass: 0.6}});
  const filled = state === '1' || state === '0';
  const fill = state === '1' ? color.ok : state === '0' ? color.bad : 'transparent';
  const ring = state === 'outline-ok' ? color.ok : darkEmpty ? '#5A606E' : color.line;
  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        flex: 'none',
        background: filled ? fill : 'transparent',
        boxShadow: filled ? 'none' : `inset 0 0 0 ${Math.max(2, size * 0.14)}px ${ring}`,
        transform: `scale(${filled ? Math.min(p, 1.15) : 1})`,
        opacity: filled ? Math.min(1, p * 1.5) : 1,
      }}
    />
  );
};

/** A row of outcome dots ("0100111001") that fill one by one from `start`. */
export const DotRow: React.FC<{outcomes: string; start: number; size?: number; gap?: number; stagger?: number}> = ({
  outcomes,
  start,
  size = 30,
  gap = 10,
  stagger = durations.dotStagger,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: 'flex', gap}}>
      {[...outcomes].map((c, i) => {
        const at = start + i * stagger;
        return <Dot key={i} state={frame >= at ? (c as '1' | '0') : 'empty'} size={size} appear={at} />;
      })}
    </div>
  );
};

/** Outlined dots (a bar or a pool of starts not yet run), optionally some marked as required. */
export const OutlineDots: React.FC<{count: number; marked?: number; size?: number; gap?: number; perRow?: number; dark?: boolean; start?: number}> = ({
  count,
  marked = 0,
  size = 30,
  gap = 10,
  perRow = 10,
  dark,
  start = 0,
}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: 'grid', gridTemplateColumns: `repeat(${perRow}, ${size}px)`, gap}}>
      {Array.from({length: count}, (_, i) => {
        const p = interpolate(frame, [start + i, start + i + 8], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
        return (
          <div key={i} style={{opacity: p, transform: `scale(${0.6 + 0.4 * p})`}}>
            <Dot state={i < marked ? 'outline-ok' : 'outline'} size={size} appear={0} darkEmpty={dark} />
          </div>
        );
      })}
    </div>
  );
};

/** The lab notebook: one row per program, its label, its dots, its count. */
export const ProgramRows: React.FC<{
  programs: Program[];
  start: number; // frame the first row starts filling
  rowEvery?: number; // frames between rows starting
  size?: number;
  labelWidth?: number;
  fontSize?: number;
  highlight?: (p: Program) => 'win' | 'warn' | 'dim' | null;
  rowGap?: number;
}> = ({programs, start, rowEvery = 18, size = 28, labelWidth = 560, fontSize = 28, highlight, rowGap = 12}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: rowGap}}>
      {programs.map((p, r) => {
        const rowStart = start + r * rowEvery;
        const shown = interpolate(frame, [rowStart - 6, rowStart], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
        const done = frame >= rowStart + p.outcomes.length * durations.dotStagger;
        const hl = highlight ? highlight(p) : null;
        const bg = hl === 'win' ? 'rgba(18,162,108,0.12)' : hl === 'warn' ? 'rgba(229,72,77,0.10)' : 'transparent';
        return (
          <div
            key={p.job}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 22,
              padding: `${size * 0.22}px 18px`,
              borderRadius: 14,
              background: bg,
              opacity: shown * (hl === 'dim' ? 0.45 : 1),
              transform: `translateX(${(1 - shown) * -20}px)`,
            }}
          >
            <div
              style={{
                width: labelWidth,
                fontSize,
                fontWeight: hl === 'win' ? 760 : 520,
                color: color.ink,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {p.label}
            </div>
            <DotRow outcomes={p.outcomes} start={rowStart} size={size} gap={size * 0.32} />
            <div style={{width: size * 2.4, textAlign: 'right', fontSize: fontSize * 1.05, fontWeight: 760, fontFamily: font.sans, fontVariantNumeric: 'tabular-nums', color: done ? color.ink : 'transparent'}}>
              {p.k}
            </div>
          </div>
        );
      })}
    </div>
  );
};
