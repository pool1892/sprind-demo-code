// Charts drawn from recorded tables, and the exam mosaic.
import React from 'react';
import {interpolate, useCurrentFrame, Easing} from 'remotion';
import {color, font, shadow} from '../theme';
import {Footage, type Segment} from './footage';
import {useFilm} from '../data';

export type Series = {
  label: string;
  color: string;
  step0: number; // control step of values[0]
  every: number; // control steps between values
  values: number[];
  width?: number;
};

export type Band = {from: number; to: number; label?: string; fill?: string};
export type Strip = {label: string; color: string; intervals: [number, number][]};

/**
 * A line chart of recorded values over control steps, with background bands (program steps),
 * on/off strips (contact, goal holds) and a playhead. Lines draw on up to `reveal` (a control
 * step), so the chart can grow in sync with footage.
 */
export const TraceChart: React.FC<{
  width: number;
  height: number;
  steps: [number, number]; // x range in control steps
  y: [number, number]; // value range
  series: Series[];
  bands?: Band[];
  strips?: Strip[];
  marks?: {step: number; label: string}[];
  yLabels?: {value: number; label: string}[];
  reveal?: number; // control step up to which lines are drawn
  playhead?: number | null;
  title?: string;
}> = ({width, height, steps, y, series, bands = [], strips = [], marks = [], yLabels = [], reveal, playhead = null, title}) => {
  const padL = 150;
  const padR = 30;
  const padT = title ? 64 : 24;
  const stripH = 30;
  const stripsH = strips.length ? strips.length * (stripH + 14) + 40 : 0;
  const plotH = height - padT - 60 - stripsH;
  const X = (s: number) => padL + ((s - steps[0]) / (steps[1] - steps[0])) * (width - padL - padR);
  const Y = (v: number) => padT + plotH - ((v - y[0]) / (y[1] - y[0])) * plotH;
  const upTo = reveal ?? steps[1];
  return (
    <svg width={width} height={height} style={{overflow: 'visible', fontFamily: font.sans}}>
      {title ? (
        <text x={padL} y={34} fontSize={28} fontWeight={800} fill={color.ink}>
          {title}
        </text>
      ) : null}
      {bands.map((b, i) => (
        <g key={i}>
          <rect x={X(b.from)} y={padT} width={Math.max(0, X(b.to) - X(b.from))} height={plotH} fill={b.fill ?? color.paper2} />
          {b.label ? (
            <text x={(X(b.from) + X(b.to)) / 2} y={padT + 34} textAnchor="middle" fontSize={24} fontWeight={700} fill={color.ink3}>
              {b.label}
            </text>
          ) : null}
        </g>
      ))}
      <line x1={padL} x2={width - padR} y1={padT + plotH} y2={padT + plotH} stroke={color.line} strokeWidth={2} />
      {yLabels.map((l, i) => (
        <text key={i} x={padL - 16} y={Y(l.value) + 8} textAnchor="end" fontSize={24} fill={color.ink2}>
          {l.label}
        </text>
      ))}
      {marks.map((m, i) => (
        <g key={i}>
          <line x1={X(m.step)} x2={X(m.step)} y1={padT} y2={padT + plotH + stripsH} stroke={color.ink} strokeDasharray="6 6" strokeWidth={2} />
          <text x={X(m.step) + 8} y={padT + 64} fontSize={22} fill={color.ink}>
            {m.label}
          </text>
        </g>
      ))}
      {series.map((s, i) => {
        const pts: string[] = [];
        s.values.forEach((v, k) => {
          const st = s.step0 + k * s.every;
          if (st <= upTo) pts.push(`${pts.length ? 'L' : 'M'}${X(st).toFixed(1)},${Y(v).toFixed(1)}`);
        });
        return <path key={i} d={pts.join(' ')} fill="none" stroke={s.color} strokeWidth={s.width ?? 6} strokeLinejoin="round" strokeLinecap="round" />;
      })}
      {strips.map((st, i) => {
        const yy = padT + plotH + 40 + i * (stripH + 14);
        return (
          <g key={i}>
            <text x={padL - 16} y={yy + 22} textAnchor="end" fontSize={20} fill={color.ink2}>
              {st.label}
            </text>
            <rect x={padL} y={yy} width={width - padL - padR} height={stripH} rx={6} fill="#EFEBE2" />
            {st.intervals
              .filter(([a]) => a <= upTo)
              .map(([a, b], k) => (
                <rect key={k} x={X(a)} y={yy} width={Math.max(3, X(Math.min(b, upTo)) - X(a))} height={stripH} rx={4} fill={st.color} />
              ))}
          </g>
        );
      })}
      {playhead !== null ? <line x1={X(playhead)} x2={X(playhead)} y1={padT} y2={padT + plotH + stripsH} stroke={color.agent} strokeWidth={4} /> : null}
    </svg>
  );
};

/**
 * Twenty recorded runs at once: each tile plays its clip, then its border turns green or red
 * by the run's recorded verdict at `verdictAt`.
 */
export const Mosaic: React.FC<{
  clips: string[];
  columns?: number;
  tile: number;
  gap?: number;
  startAt: number; // frame playback starts
  playFrames: number; // frames the whole run takes to play (each clip is fitted to this)
  verdictAt: number;
  appearAt?: number;
}> = ({clips, columns = 5, tile, gap = 14, startAt, playFrames, verdictAt, appearAt = 0}) => {
  const frame = useCurrentFrame();
  const film = useFilm();
  return (
    <div style={{display: 'grid', gridTemplateColumns: `repeat(${columns}, ${tile}px)`, gap}}>
      {clips.map((id, i) => {
        const clip = film.clips[id];
        const appear = interpolate(frame, [appearAt + i, appearAt + i + 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
        const verdict = frame >= verdictAt + i * 1.5;
        const ok = clip?.delivered;
        const border = verdict ? (ok ? color.ok : color.bad) : 'rgba(0,0,0,0)';
        let segments: Segment[] | undefined;
        if (clip) {
          const steps = (clip.frames - 1) * clip.frame_every;
          const rate = ((steps / clip.fps) * 30) / Math.max(1, playFrames);
          segments = [{from: clip.first_step, to: clip.first_step + steps, rate}];
        }
        return (
          <div key={id} style={{opacity: appear, transform: `scale(${0.92 + 0.08 * appear})`, borderRadius: 18, boxShadow: `0 0 0 7px ${border}, ${shadow.card}`}}>
            <Footage clip={id} segments={segments} startAt={startAt} width={tile} height={tile} radius={14} />
          </div>
        );
      })}
    </div>
  );
};
