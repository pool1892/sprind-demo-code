// lessons: the engineering loop the agent worked through, as five cards, each with a thumbnail
// from earlier in the film (a recorded clip, a drawn icon, recorded program dots, the recorded
// probe, the final exams' dots), then "first, compact tests".
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {Footage} from '../../components/footage';
import {useCue, useFilm, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {RunGrid, ramp, type RunState} from './parts';
import {DotLegend} from '../opening/DotLegend';

const ICON_W = 450;
const ICON_H = 250;
const TW = 284; // thumbnail box
const TH = 200;


/** The skill says done while the hand still holds the bowl. */
const DoneIcon: React.FC<{at: number}> = ({at}) => {
  const frame = useCurrentFrame();
  const done = ramp(frame, at + 14, at + 26, Easing.out(Easing.back(1.8)));
  const hold = ramp(frame, at + 34, at + 46);
  const pulse = hold * (0.6 + 0.4 * Math.sin((frame - at) / 5));
  return (
    <svg width={ICON_W} height={ICON_H} viewBox={`0 0 ${ICON_W} ${ICON_H}`}>
      {/* cabinet top */}
      <rect x={70} y={196} width={300} height={22} rx={6} fill={color.wood} />
      {/* arm coming down */}
      <rect x={208} y={0} width={34} height={92} rx={10} fill={color.ink} />
      <rect x={170} y={86} width={110} height={22} rx={8} fill={color.ink} />
      {/* bowl, just above the cabinet */}
      <path d="M160,150 Q225,214 290,150 Z" fill="#3D4250" />
      <rect x={156} y={144} width={138} height={10} rx={5} fill="#3D4250" />
      {/* fingers still closed on the bowl */}
      <rect x={160} y={104} width={16} height={60} rx={6} fill={color.ink} />
      <rect x={274} y={104} width={16} height={60} rx={6} fill={color.ink} />
      <circle cx={225} cy={140} r={92} fill="none" stroke={color.bad} strokeWidth={6} opacity={pulse} />
      <g opacity={done} transform={`translate(330, 40) scale(${0.6 + 0.4 * done})`}>
        <rect x={-10} y={-26} width={120} height={52} rx={26} fill={color.ok} />
        <text x={50} y={10} textAnchor="middle" fontSize={30} fontWeight={800} fill="#fff">
          done ✓
        </text>
      </g>
      <text x={20} y={44} fontSize={26} fontWeight={760} fill={color.bad} opacity={hold}>
        still holding
      </text>
    </svg>
  );
};

/** The agent's own first run, the plain chain on practice start 2: the recorded run around the bowl's fall from the cabinet. */
const EvidenceThumb: React.FC<{at: number}> = ({at}) => {
  const film = useFilm();
  const h: number[] = film.traces['r1-practice2-chain']?.height_m ?? [];
  const top = Math.max(...h);
  const mid = (h[0] + top) / 2;
  const peakAt = h.indexOf(top);
  let fall = peakAt;
  while (fall < h.length - 1 && h[fall] > mid) fall++;
  return <Footage clip="r1-chain-practice2-hero" segments={[{from: Math.max(0, fall - 40), to: fall + 10}]} startAt={at + 6} width={TW} height={TH} radius={14} focus={{x: 0.55, y: 0.5}} />;
};

/** Programs on the same practice layouts, one row each: recorded outcomes. */
const ChangesThumb: React.FC<{at: number}> = ({at}) => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const rows = film.programs.r1.filter((p) => p.role !== 'confirm').slice(0, 6);
  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: 10, padding: '18px 22px'}}>
      {rows.map((p, r) => (
        <RunGrid
          key={p.job}
          perRow={p.outcomes.length}
          size={14}
          gap={8}
          states={[...p.outcomes].map((c, i) => (frame >= at + 8 + r * 5 + i ? (c === '1' ? 'ok' : 'bad') : 'empty') as RunState)}
        />
      ))}
    </div>
  );
};

/** The probe: commanded against the hand, the linked site's arm and the lagging robot. */
const InstrumentThumb: React.FC<{at: number}> = ({at}) => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const robots = film.fleet.probe.robots as Record<string, {commanded_cm: number[]; measured_cm: number[]}>;
  const r01 = robots.r01;
  const r06 = robots.r06;
  const n = 200;
  const peak = Math.max(...r01.commanded_cm);
  const reveal = ramp(frame, at + 6, at + 40) * n;
  const path = (v: number[], y0: number, hgt: number) => {
    const pts: string[] = [];
    for (let s = 0; s <= Math.min(n, reveal); s += 2) pts.push(`${pts.length ? 'L' : 'M'}${(12 + (s / n) * (TW - 24)).toFixed(1)},${(y0 + hgt - (v[s] / peak) * hgt).toFixed(1)}`);
    return pts.join(' ');
  };
  return (
    <svg width={TW} height={TH}>
      <path d={path(r01.commanded_cm, 16, 64)} fill="none" stroke={color.ink} strokeWidth={3} />
      <path d={path(r01.measured_cm, 16, 64)} fill="none" stroke={color.lead} strokeWidth={5} />
      <path d={path(r06.commanded_cm, 112, 64)} fill="none" stroke={color.ink} strokeWidth={3} />
      <path d={path(r06.measured_cm, 112, 64)} fill="none" stroke={color.violet} strokeWidth={5} />
    </svg>
  );
};

/** The final exams on cases never seen: every reserved layout of both jobs. */
const ProofThumb: React.FC<{at: number}> = ({at}) => {
  const frame = useCurrentFrame();
  const k1 = Number(useNumber('r1_delivered'));
  const n1 = Number(useNumber('r1_release_n'));
  const k2 = Number(useNumber('r2_delivered'));
  const n2 = Number(useNumber('r2_release_n'));
  const row = (k: number, n: number, d: number): RunState[] => Array.from({length: n}, (_, i) => (frame >= at + d + i ? (i < k ? 'ok' : 'bad') : 'empty'));
  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: 16, padding: '26px 22px'}}>
      <RunGrid perRow={10} size={18} gap={7} states={row(k1, n1, 8)} />
      <RunGrid perRow={10} size={18} gap={7} states={row(k2, n2, 20)} />
    </div>
  );
};

const CauseThumb: React.FC<{at: number}> = ({at}) => (
  <div style={{transform: `scale(${TW / ICON_W})`, transformOrigin: 'top left', width: ICON_W, height: ICON_H, marginTop: 10}}>
    <DoneIcon at={at} />
  </div>
);

const STEPS = [
  {cue: 'evidence', label: 'looked at the evidence', Thumb: EvidenceThumb},
  {cue: 'cause', label: 'found the physical cause', Thumb: CauseThumb},
  {cue: 'change', label: 'one change at a time', Thumb: ChangesThumb},
  {cue: 'instrument', label: 'built a measurement', Thumb: InstrumentThumb},
  {cue: 'proof', label: 'proof on cases never seen', Thumb: ProofThumb},
] as const;

export const Lessons: React.FC = () => {
  const frame = useCurrentFrame();
  const tEngineer = useCue('engineer', 0.1);
  const at = [useCue('evidence', 0.17), useCue('cause', 0.22), useCue('change', 0.29), useCue('instrument', 0.37), useCue('proof', 0.49)];
  const tCompact = useCue('compact', 0.6);
  const tThrough = useCue('through', 0.75);
  const line1 = ramp(frame, tCompact - 4, tCompact + 12);
  const line2 = ramp(frame, tThrough - 4, tThrough + 12);
  return (
    <Stage act="what we learned">
      <div style={{position: 'absolute', left: 120, top: 104, opacity: ramp(frame, tEngineer - 4, tEngineer + 12)}}>
        <Headline size="h2">The agent worked like a good engineer</Headline>
      </div>
      {STEPS.map((s, i) => {
        const p = ramp(frame, at[i] - 6, at[i] + 12, Easing.out(Easing.cubic));
        const current = frame >= at[i] - 6 && (i === STEPS.length - 1 || frame < at[i + 1] - 6) && frame < tCompact;
        return (
          <div
            key={s.cue}
            style={{
              position: 'absolute',
              left: 120 + i * 340,
              top: 270,
              width: 316,
              height: 370,
              boxSizing: 'border-box',
              background: color.card,
              borderRadius: 24,
              boxShadow: shadow.card,
              borderTop: `12px solid ${current ? color.agent : color.line}`,
              padding: 16,
              opacity: p,
              transform: `translateY(${(1 - p) * 36}px)`,
            }}
          >
            <div style={{width: TW, height: TH, borderRadius: 14, background: color.paper, overflow: 'hidden'}}>
              <s.Thumb at={at[i]} />
            </div>
            <div style={{fontSize: 36, fontWeight: 780, lineHeight: 1.15, color: color.ink, marginTop: 22}}>{s.label}</div>
          </div>
        );
      })}
      <div style={{position: 'absolute', left: 120, top: 666, opacity: ramp(frame, at[2], at[2] + 12)}}>
        <DotLegend small />
      </div>
      <div style={{position: 'absolute', left: 120, right: 120, top: 770, textAlign: 'center'}}>
        <span style={{fontSize: 50, fontWeight: 800, color: color.ink, letterSpacing: '-0.015em', opacity: line1}}>First, compact tests</span>
        <div style={{fontSize: 46, fontWeight: 720, color: color.ok, marginTop: 14, opacity: line2, transform: `translateY(${(1 - line2) * 14}px)`}}>each investigation carried through to a working result</div>
      </div>
    </Stage>
  );
};
