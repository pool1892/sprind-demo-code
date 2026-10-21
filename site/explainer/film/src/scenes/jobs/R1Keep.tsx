// r1-keep: what the installation keeps. The job and its check land on the installation's
// library shelf, ready for the next request; the time from request to proven job counts up.
// Names and counts come from the bound record.
import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {CountUp} from '../../components/basics';
import {useCue, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {JobStage, Label, MonoName, ramp, useArrive} from './common';

const SHELF_Y = 846;
const CARD = {w: 620, h: 370, gap: 70};

export const ShelfCard: React.FC<{x: number; arrive: number; accent: string; kicker: string; children: React.ReactNode}> = ({x, arrive, accent, kicker, children}) => {
  const p = Math.min(1.06, arrive);
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: SHELF_Y - CARD.h,
        width: CARD.w,
        height: CARD.h,
        background: color.card,
        borderRadius: 24,
        boxShadow: shadow.card,
        borderTop: `14px solid ${accent}`,
        padding: '26px 32px',
        boxSizing: 'border-box',
        opacity: Math.min(1, arrive * 1.4),
        transform: `translateY(${(1 - p) * -90}px)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      <Label tone={accent}>{kicker}</Label>
      {children}
    </div>
  );
};

/** A clock face whose arc sweeps to the minutes taken (of an hour). */
const Clock: React.FC<{minutes: number; progress: number; size?: number}> = ({minutes, progress, size = 128}) => {
  const r = size * 0.4;
  const c = size / 2;
  const frac = Math.min(1, (minutes / 60) * progress);
  const a = -Math.PI / 2 + frac * 2 * Math.PI;
  const large = frac > 0.5 ? 1 : 0;
  const x = c + r * Math.cos(a);
  const y = c + r * Math.sin(a);
  return (
    <svg width={size} height={size}>
      <circle cx={c} cy={c} r={r + size * 0.06} fill={color.card} stroke={color.line} strokeWidth={4} />
      {frac > 0.001 ? <path d={`M${c},${c} L${c},${c - r} A${r},${r} 0 ${large} 1 ${x},${y} Z`} fill={color.lead} opacity={0.9} /> : null}
      <circle cx={c} cy={c} r={size * 0.05} fill={color.ink} />
    </svg>
  );
};

export const R1Keep: React.FC = () => {
  const frame = useCurrentFrame();
  const jobName = useNumber('r1_job_name');
  const checkName = useNumber('r1_check_name');
  const delivered = useNumber('r1_delivered');
  const n = useNumber('r1_release_n');
  const minutes = useNumber('r1_wall_minutes');

  const libraryF = useCue('library', 0.04);
  const shelfF = useCue('shelf', 0.15);
  const readyF = useCue('ready', 0.34);
  const timeF = useCue('time', 0.53);

  const shelfIn = useArrive(0);
  const jobIn = useArrive(libraryF + 6, 'lively');
  const checkIn = useArrive(libraryF + 22, 'lively');
  const titleIn = useArrive(shelfF - 4);
  const timerIn = useArrive(timeF - 6);
  const sweep = ramp(frame, timeF, timeF + 36);
  const x0 = (1920 - (2 * CARD.w + CARD.gap)) / 2;

  return (
    <JobStage job={1} step={null}>
      <div style={{position: 'absolute', left: 120, top: 136, opacity: titleIn, transform: `translateY(${(1 - titleIn) * 16}px)`}}>
        <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>The installation's library</div>
        <div style={{fontSize: 32, fontWeight: 480, color: color.ink2, marginTop: 8, opacity: ramp(frame, readyF - 4, readyF + 10)}}>ready for the next request</div>
      </div>

      {/* request to proven job */}
      <div style={{position: 'absolute', right: 120, top: 150, display: 'flex', alignItems: 'center', gap: 26, opacity: timerIn, transform: `translateY(${(1 - Math.min(1, timerIn)) * 16}px)`}}>
        <Clock minutes={Number(minutes) || 0} progress={sweep} />
        <div>
          <div style={{fontSize: 28, fontWeight: 650, color: color.ink2}}>request → proven job</div>
          <div style={{fontSize: 84, fontWeight: 840, letterSpacing: '-0.03em', color: color.ink, lineHeight: 1}}>
            <CountUp to={Number(minutes) || 0} start={timeF} frames={36} /> <span style={{fontSize: 44, fontWeight: 700, color: color.ink2}}>min</span>
          </div>
        </div>
      </div>

      {/* the shelf */}
      <div style={{position: 'absolute', left: 120, right: 120, top: SHELF_Y, height: 30, borderRadius: 8, opacity: shelfIn, background: `linear-gradient(${color.wood}, #A07848)`, boxShadow: '0 18px 30px rgba(80,55,25,0.28)'}} />
      <div style={{position: 'absolute', left: 120, right: 120, top: SHELF_Y, height: 6, borderRadius: 8, opacity: shelfIn, background: 'rgba(255,255,255,0.35)'}} />
      {[260, 1660].map((x) => (
        <div key={x} style={{position: 'absolute', left: x, top: SHELF_Y + 30, width: 22, height: 70, opacity: shelfIn, background: '#8C6A43', borderRadius: '0 0 6px 6px'}} />
      ))}

      <ShelfCard x={x0} arrive={jobIn} accent={color.ok} kicker="job · proven">
        <MonoName size={40}>{jobName}</MonoName>
        <div style={{fontSize: 30, fontWeight: 560, color: color.ink2, lineHeight: 1.3, marginTop: 'auto'}}>
          delivered <b style={{color: color.ink}}>{delivered} of {n}</b> on the final exam
        </div>
      </ShelfCard>
      <ShelfCard x={x0 + CARD.w + CARD.gap} arrive={checkIn} accent={color.teal} kicker="check · by the agent">
        <MonoName size={40}>{checkName}</MonoName>
        <div style={{fontSize: 30, fontWeight: 560, color: color.ink2, lineHeight: 1.3, marginTop: 'auto'}}>Does the bowl stay put while the next skill runs?</div>
      </ShelfCard>
      <div style={{position: 'absolute', left: 0, right: 0, top: SHELF_Y + 64, textAlign: 'center', opacity: interpolate(frame, [timeF + 40, timeF + 56], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
        <span style={{fontSize: 28, fontWeight: 600, color: color.ink2}}>the second job starts from this library</span>
      </div>
    </JobStage>
  );
};
