// The final exam: the twenty locked-away layouts, the plain chain against the agent's job, every
// tile a recorded run with its recorded verdict; the counts from the bound record. Shared by
// r1-prove, r1-keep (which opens on the finished panel) and r2-fix.
import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {useFilm, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {ExamMosaic, Verdict, ramp, useArrive} from './common';
import {TileLegend} from './DotLegend';

export const EXAM = {x: 120, top: 290, tile: 150, gap: 12, between: 84};
export const mosaicW = (tile = EXAM.tile, gap = EXAM.gap) => 5 * tile + 4 * gap;
export const mosaicH = (tile = EXAM.tile, gap = EXAM.gap) => 4 * tile + 3 * gap;

/** A padlock, drawn: `open` 0..1 lifts and swings the shackle. */
export const Padlock: React.FC<{size?: number; open?: number; tone?: string}> = ({size = 120, open = 0, tone = '#fff'}) => (
  <svg width={size} height={size * 1.25} viewBox="0 0 100 125" style={{overflow: 'visible'}}>
    <g transform={`translate(${70 - 0} ${52}) rotate(${-38 * open}) translate(${-70} ${-52}) translate(0 ${-14 * open})`}>
      <path d="M30 56 V36 a20 20 0 0 1 40 0 V56" fill="none" stroke={tone} strokeWidth={11} strokeLinecap="round" />
    </g>
    <rect x={14} y={52} width={72} height={60} rx={12} fill={tone} />
    <circle cx={50} cy={78} r={8} fill={color.night} />
    <rect x={46} y={80} width={8} height={18} rx={4} fill={color.night} />
  </svg>
);

/** The dark vault over the exam area; `open` 0..1 swings its doors apart. */
export const Vault: React.FC<{open: number; show: number; label: string; width: number; height: number}> = ({open, show, label, width, height}) => {
  if (show <= 0 || open >= 1) return null;
  const doors = interpolate(open, [0.35, 1], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const lock = interpolate(open, [0, 0.35], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const half = width / 2;
  const door = (side: -1 | 1): React.CSSProperties => ({
    position: 'absolute',
    top: 0,
    left: side < 0 ? 0 : half,
    width: half,
    height,
    background: color.night,
    borderRadius: side < 0 ? '28px 0 0 28px' : '0 28px 28px 0',
    transform: `translateX(${side * doors * half}px)`,
    boxShadow: shadow.frame,
  });
  return (
    <div style={{position: 'absolute', inset: 0, opacity: show, overflow: 'hidden', borderRadius: 28}}>
      <div style={door(-1)} />
      <div style={door(1)} />
      <div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 26, opacity: 1 - doors}}>
        <Padlock size={110} open={lock} />
        <div style={{fontSize: 34, fontWeight: 700, color: color.nightText, letterSpacing: '0.01em'}}>{label}</div>
      </div>
    </div>
  );
};

/** One side's header: its name, then its count and verdict glyph when the verdicts land. */
const Header: React.FC<{name: string; count: string; of: string; ok: boolean; at: number; hold?: boolean; width: number}> = ({name, count, of, ok, at, hold, width}) => {
  const pop = useArrive(at, 'lively');
  const p = hold ? 1 : pop;
  return (
    <div style={{width, height: 72, display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between'}}>
      <div style={{fontSize: 36, fontWeight: 780, color: color.ink, paddingBottom: 8}}>{name}</div>
      <div style={{display: 'flex', alignItems: 'center', gap: 14, opacity: Math.min(1, p), transform: `scale(${0.7 + 0.3 * Math.min(1.1, p)})`, transformOrigin: 'right bottom'}}>
        <span style={{fontSize: 62, fontWeight: 840, letterSpacing: '-0.02em', color: color.ink, fontVariantNumeric: 'tabular-nums', lineHeight: 1}}>
          {count} <span style={{fontSize: 40, fontWeight: 700, color: color.ink2}}>of {of}</span>
        </span>
        <Verdict ok={ok} size={44} />
      </div>
    </div>
  );
};

/** Frames from a mosaic's first verdict to its last (the verdicts ripple tile by tile). */
export const VERDICT_RIPPLE = 26;

/**
 * The two mosaics side by side with their headers and the rescued/broken line. Times are frames
 * of the calling scene; each count pops once its tiles' verdicts have landed (the runs first,
 * then the number); `hold` shows the finished panel (every tile on its last frame).
 */
export const ExamPanel: React.FC<{
  job: 'r1' | 'r2';
  appearAt: number;
  chainPlay: [number, number]; // start frame, frames to play
  jobPlay: [number, number];
  chainVerdictAt: number;
  jobVerdictAt: number;
  footerAt: number;
  hold?: boolean;
  top?: number;
  /** Dim the side the voice has not reached yet, until its verdicts. */
  dim?: 'chain' | 'job';
}> = ({job, appearAt, chainPlay, jobPlay, chainVerdictAt, jobVerdictAt, footerAt, hold, top = EXAM.top, dim}) => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const chainIds = film.mosaics[`${job}-chain`] ?? [];
  const jobIds = film.mosaics[`${job}-job`] ?? [];
  const naive = useNumber(`${job}_naive_delivered`);
  const delivered = useNumber(`${job}_delivered`);
  const n = useNumber(`${job}_release_n`);
  const rescued = useNumber(`${job}_rescued`);
  const broken = useNumber(`${job}_broken`);
  const w = mosaicW();
  const footer = hold ? 1 : ramp(frame, footerAt, footerAt + 14);
  return (
    <>
      <div style={{position: 'absolute', left: EXAM.x, top: top - 84}}>
        <Header name="the plain chain" count={naive} of={n} ok={false} at={chainVerdictAt + VERDICT_RIPPLE} hold={hold} width={w} />
      </div>
      <div style={{position: 'absolute', left: EXAM.x + w + EXAM.between, top: top - 84}}>
        <Header name="the agent's job" count={delivered} of={n} ok at={jobVerdictAt + VERDICT_RIPPLE} hold={hold} width={w} />
      </div>
      <div style={{position: 'absolute', left: EXAM.x, top}}>
        <ExamMosaic clips={chainIds} tile={EXAM.tile} gap={EXAM.gap} startAt={chainPlay[0]} playFrames={chainPlay[1]} verdictAt={chainVerdictAt} appearAt={appearAt} hold={hold} dimUntil={dim === 'chain' ? chainVerdictAt : undefined} />
      </div>
      <div style={{position: 'absolute', left: EXAM.x + w + EXAM.between, top}}>
        <ExamMosaic clips={jobIds} tile={EXAM.tile} gap={EXAM.gap} startAt={jobPlay[0]} playFrames={jobPlay[1]} verdictAt={jobVerdictAt} appearAt={appearAt + 6} hold={hold} dimUntil={dim === 'job' ? jobVerdictAt : undefined} />
      </div>
      <div style={{position: 'absolute', left: EXAM.x, top: top + mosaicH() + 26, opacity: hold ? 1 : ramp(frame, appearAt + 10, appearAt + 24)}}>
        <TileLegend />
      </div>
      <div
        style={{
          position: 'absolute',
          left: EXAM.x + w + EXAM.between,
          width: w,
          top: top + mosaicH() + 26,
          display: 'flex',
          justifyContent: 'flex-end',
          gap: 14,
          opacity: footer,
          fontSize: 26,
          fontWeight: 600,
          color: color.ink2,
          whiteSpace: 'nowrap',
        }}
      >
        <span>
          <b style={{color: color.ink}}>{rescued}</b> rescued
        </span>
        <span style={{color: color.ink3}}>·</span>
        <span>
          <b style={{color: color.ink}}>{broken}</b> broken
        </span>
      </div>
    </>
  );
};
