// results: three result cards (job 1, job 2, the fleet), each first as dots, then the number;
// under them: the same robot model and design throughout (the voice names the robots'
// vision-language-action model), and the time from request to proven job.
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {RunGrid, ramp, type RunState} from './parts';
import {DotLegend} from '../opening/DotLegend';

const ResultCard: React.FC<{
  x: number;
  at: number;
  label: string;
  sub: string;
  from: string;
  to: string;
  tail?: string;
  dots?: {before: number; after: number; n: number};
  bar?: {before: number; after: number};
  compact?: boolean;
}> = ({x, at, label, sub, from, to, tail, dots, bar, compact}) => {
  const k = compact ? 0.8 : 1;
  const frame = useCurrentFrame();
  const p = ramp(frame, at - 4, at + 14, Easing.out(Easing.cubic));
  const flip = ramp(frame, at + 22, at + 40);
  const big = ramp(frame, at + 34, at + 46, Easing.out(Easing.back(1.6)));
  let states: RunState[] = [];
  if (dots) {
    states = Array.from({length: dots.n}, (_, i) => {
      if (frame < at + 6 + i * 0.6) return 'empty';
      const after = i < dots.after;
      const before = i < dots.before;
      return frame >= at + 22 + i * 0.9 ? (after ? 'ok' : 'bad') : before ? 'ok' : 'bad';
    });
  }
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: 290,
        width: 530,
        height: 470,
        boxSizing: 'border-box',
        background: color.card,
        borderRadius: 28,
        boxShadow: shadow.card,
        padding: '34px 40px',
        opacity: p,
        transform: `translateY(${(1 - p) * 40}px)`,
      }}
    >
      <div style={{fontSize: 30, fontWeight: 780, letterSpacing: '0.12em', textTransform: 'uppercase', color: color.ink3}}>{label}</div>
      <div style={{fontSize: 30, fontWeight: 560, color: color.ink2, marginTop: 6}}>{sub}</div>
      <div style={{height: 110, marginTop: 30, display: 'flex', alignItems: 'center'}}>
        {dots ? <RunGrid states={states} perRow={10} size={30} gap={14} /> : null}
        {bar ? (
          <div style={{width: '100%'}}>
            <div style={{height: 34, borderRadius: 17, background: color.paper2, overflow: 'hidden', position: 'relative'}}>
              <div style={{position: 'absolute', left: 0, top: 0, bottom: 0, width: `${100 * (bar.before + (bar.after - bar.before) * flip) * Math.min(1, ramp(frame, at + 4, at + 20) * 1.0)}%`, background: flip > 0.5 ? color.ok : color.bad, borderRadius: 17}} />
            </div>
            <div style={{fontSize: 26, color: color.ink3, fontWeight: 600, marginTop: 12}}>share of attempts that succeeded</div>
          </div>
        ) : null}
      </div>
      <div style={{position: 'absolute', left: 40, bottom: 34, display: 'flex', alignItems: 'baseline', gap: 18, whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums'}}>
        <span style={{fontSize: 76 * k, fontWeight: 800, color: color.bad, opacity: ramp(frame, at + 8, at + 20)}}>{from}</span>
        <span style={{fontSize: 60 * k, fontWeight: 600, color: color.ink3, opacity: big}}>→</span>
        <span style={{fontSize: 110 * k, fontWeight: 850, color: color.ok, letterSpacing: '-0.04em', opacity: big, display: 'inline-block', transform: `scale(${0.8 + 0.2 * big})`, transformOrigin: 'left bottom'}}>{to}</span>
        {tail ? <span style={{fontSize: 34, fontWeight: 650, color: color.ink2, opacity: big}}>{tail}</span> : null}
      </div>
    </div>
  );
};

export const Results: React.FC = () => {
  const frame = useCurrentFrame();
  const r1From = useNumber('r1_naive_delivered');
  const r1To = useNumber('r1_delivered');
  const r1N = useNumber('r1_release_n');
  const r2From = useNumber('r2_naive_delivered');
  const r2To = useNumber('r2_delivered');
  const r2N = useNumber('r2_release_n');
  const bFrom = useNumber('b_eval_original_pct');
  const bTo = useNumber('b_eval_final_pct');
  const bK0 = useNumber('b_eval_original_k');
  const bK1 = useNumber('b_eval_final_k');
  const bN = useNumber('b_eval_final_n');
  const robots = useNumber('b_robots');
  const m1 = useNumber('r1_wall_minutes');
  const m2 = useNumber('r2_wall_minutes');
  const tBoth = useCue('both', 0.02);
  const tJobs = useCue('jobs', 0.22);
  const tJob2 = useCue('job2', 0.32);
  const tFleet = useCue('fleet', 0.45);
  const tModel = useCue('model', 0.72);
  const foot = ramp(frame, tModel - 4, tModel + 12);
  const foot2 = ramp(frame, tModel + 26, tModel + 42);
  return (
    <Stage act="the results">
      <div style={{position: 'absolute', left: 120, top: 104, opacity: ramp(frame, tBoth, tBoth + 14)}}>
        <Headline size="h2">In both experiments, the agent did the engineering</Headline>
        <div style={{display: 'flex', alignItems: 'center', gap: 36, marginTop: 12}}>
          <span style={{fontSize: 30, fontWeight: 520, color: color.ink2}}>final tests on cases it never saw</span>
          <DotLegend small />
        </div>
      </div>
      <ResultCard x={120} at={tJobs} label="Experiment 1 · job 1" sub="bowl and drawer" from={r1From} to={r1To} tail={`of ${r1N}`} dots={{before: Number(r1From), after: Number(r1To), n: Number(r1N)}} />
      <ResultCard x={695} at={tJob2} label="Experiment 1 · job 2" sub="plate and bowl" from={r2From} to={r2To} tail={`of ${r2N}`} dots={{before: Number(r2From), after: Number(r2To), n: Number(r2N)}} />
      <ResultCard
        x={1270}
        at={tFleet}
        label="Experiment 2 · fleet"
        sub={`${robots} arms · ${bK0} → ${bK1} of ${bN}`}
        from={bFrom}
        to={bTo}
        bar={{before: Number(bK0) / Number(bN), after: Number(bK1) / Number(bN)}}
        compact
      />
      <div style={{position: 'absolute', left: 120, right: 120, top: 820, textAlign: 'center'}}>
        <div style={{fontSize: 52, fontWeight: 800, color: color.ink, letterSpacing: '-0.015em', opacity: foot, transform: `translateY(${(1 - foot) * 16}px)`}}>
          Same VLA and robot design throughout
        </div>
        <div style={{fontSize: 36, fontWeight: 560, color: color.ink2, marginTop: 18, opacity: foot2, transform: `translateY(${(1 - foot2) * 12}px)`}}>
          Request to proven job: <b style={{color: color.ink}}>{m1}</b> and <b style={{color: color.ink}}>{m2}</b> minutes
        </div>
      </div>
    </Stage>
  );
};
