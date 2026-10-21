// b-result: the reserved panel of attempts the agent never saw, as dots flipping from the fleet
// as it was to the fleet with the agent's changes; the agent's forecast beside the measured
// gain; then the check against what we had planted: the same four items, in the same words,
// as b-planted's cards, the calibration (our false lead) last and in grey.
import React from 'react';
import {interpolate, useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue, useFilm, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {RunGrid, clamp, ramp, seeded, type RunState} from './parts';
import {ExperimentTag} from '../opening/ExperimentCard';
import {DotLegend} from '../opening/DotLegend';
import {PLANTED, castOf} from './planted';

export const BResult: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const total = useNumber('b_eval_final_n');
  const before = useNumber('b_eval_original_k');
  const after = useNumber('b_eval_final_k');
  const pctBefore = useNumber('b_eval_original_pct');
  const pctAfter = useNumber('b_eval_final_pct');
  const rescued = useNumber('b_eval_rescued');
  const broken = useNumber('b_eval_broken');
  const tPanel = useCue('panel', 0.02);
  const tBefore = useCue('before', 0.21);
  const tAfter = useCue('after', 0.35);
  const tPct = useCue('pct', 0.49);
  const tChecked = useCue('checked', 0.61);
  const tFound = useCue('found', 0.69);
  const tFalse = useCue('false', 0.83);

  const cast = castOf(film.fleet);
  const ev = film.fleet.evaluation as {before: number; after: number; attempts: number; pairs: {rescued: number; broken: number}};
  const both = ev.before - ev.pairs.broken;
  const neither = ev.attempts - both - ev.pairs.rescued - ev.pairs.broken;
  const kinds = seeded(
    [
      ...Array<string>(both).fill('both'),
      ...Array<string>(ev.pairs.rescued).fill('rescued'),
      ...Array<string>(ev.pairs.broken).fill('broken'),
      ...Array<string>(Math.max(0, neither)).fill('neither'),
    ],
    11,
  );
  const perRow = 32;
  const fillAt = (i: number) => tBefore + (i % perRow) * 0.4 + Math.floor(i / perRow) * 0.8;
  const flipAt = (i: number) => tAfter + ((i % perRow) / perRow) * 22 + Math.floor(i / perRow) * 0.4;
  const states: RunState[] = kinds.map((k, i) => {
    if (frame < fillAt(i)) return 'empty';
    const flipped = frame >= flipAt(i);
    if (k === 'both') return 'ok';
    if (k === 'neither') return 'bad';
    if (k === 'rescued') return flipped ? 'ok' : 'bad';
    return flipped ? 'bad' : 'ok';
  });
  const head = ramp(frame, tPanel, tPanel + 14);
  const shownBefore = ramp(frame, tBefore, tBefore + 12);
  const count = Math.round(interpolate(frame, [tAfter, tAfter + 30], [ev.before, ev.after], {...clamp, easing: Easing.out(Easing.cubic)}));
  const flipped = frame >= tAfter;
  const pct = ramp(frame, tPct - 4, tPct + 12);
  const chips = ramp(frame, tPct + 20, tPct + 34);
  const fc = ramp(frame, tPct + 44, tPct + 58);
  const checkHead = ramp(frame, tChecked - 4, tChecked + 12);
  const forecast = film.fleet.forecast_gain_pts as {mid: number};
  const measured = film.fleet.measured_gain_pts as number;

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={2} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 124, opacity: head, transform: `translateY(${(1 - head) * 16}px)`}}>
        <Headline size="h2">{total} attempts the agent never saw</Headline>
        <div style={{display: 'flex', alignItems: 'center', gap: 36, marginTop: 12}}>
          <span style={{fontSize: 30, fontWeight: 520, color: color.ink2}}>the fleet as it was, then with the agent's changes</span>
          <DotLegend small wording="attempts" />
        </div>
      </div>
      <div style={{position: 'absolute', left: 120, top: 262, padding: 26, background: color.card, borderRadius: 26, boxShadow: shadow.card, opacity: head}}>
        <RunGrid
          states={states}
          perRow={perRow}
          size={17}
          gap={6}
          pop={(i) => (kinds[i] === 'rescued' || kinds[i] === 'broken' ? ramp(frame, flipAt(i), flipAt(i) + 10) : 0)}
        />
      </div>
      <div style={{position: 'absolute', left: 1030, top: 262, width: 780}}>
        <div style={{display: 'flex', alignItems: 'baseline', gap: 18, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', opacity: shownBefore}}>
          <span style={{fontSize: 140, fontWeight: 850, letterSpacing: '-0.04em', color: flipped ? color.ok : color.ink, lineHeight: 1}}>{flipped ? (frame >= tAfter + 30 ? after : count) : before}</span>
          <span style={{fontSize: 38, fontWeight: 650, color: color.ink2}}>of {total} succeed</span>
        </div>
        <div style={{fontSize: 32, fontWeight: 600, color: color.ink2, marginTop: 8, whiteSpace: 'nowrap', opacity: ramp(frame, tAfter + 20, tAfter + 34)}}>
          as the fleet was: <span style={{color: color.ink, fontWeight: 800}}>{before}</span>
        </div>
        <div style={{fontSize: 60, fontWeight: 820, marginTop: 16, whiteSpace: 'nowrap', letterSpacing: '-0.02em', opacity: pct}}>
          <span style={{color: color.bad}}>{pctBefore}</span>
          <span style={{color: color.ink3, fontWeight: 600}}> → </span>
          <span style={{color: color.ok}}>{pctAfter}</span>
        </div>
        <div style={{display: 'flex', gap: 14, marginTop: 16, opacity: chips}}>
          <span style={{padding: '6px 16px', borderRadius: 999, background: 'rgba(18,162,108,0.12)', color: color.ok, fontSize: 26, fontWeight: 760}}>{rescued} rescued</span>
          <span style={{padding: '6px 16px', borderRadius: 999, background: 'rgba(229,72,77,0.10)', color: color.bad, fontSize: 26, fontWeight: 760}}>{broken} broken</span>
        </div>
        <div style={{marginTop: 22, opacity: fc, fontSize: 30, fontWeight: 650, whiteSpace: 'nowrap'}}>
          <span style={{color: color.agent, fontWeight: 780}}>agent's forecast +{forecast.mid.toFixed(1)} points</span>
          <span style={{color: color.ink3}}> · </span>
          <span style={{color: color.ink, fontWeight: 780}}>measured +{measured.toFixed(1)}</span>
        </div>
      </div>

      {/* checked against what we had planted */}
      <div style={{position: 'absolute', left: 120, top: 790, right: 120, opacity: checkHead}}>
        <div style={{fontSize: 26, fontWeight: 760, letterSpacing: '0.12em', textTransform: 'uppercase', color: color.ink3}}>against what we planted</div>
        <div style={{display: 'flex', gap: 22, marginTop: 14}}>
          {PLANTED.map((c, i) => {
            const isLead = c.key === 'calibration';
            const at = isLead ? tFalse : tFound + i * 12;
            const p = ramp(frame, at, at + 14, Easing.out(Easing.cubic));
            const who = c.key === 'controller' ? `robot ${cast.lagRobot}, site ${cast.siteOf[c.key]}` : `site ${cast.siteOf[c.key]}`;
            return (
              <div
                key={c.key}
                style={{
                  flex: 1,
                  background: isLead ? color.paper2 : color.card,
                  borderRadius: 20,
                  boxShadow: isLead ? 'none' : shadow.card,
                  padding: '14px 20px 16px',
                  minWidth: 0,
                  opacity: p,
                  transform: `translateY(${(1 - p) * 20}px)`,
                  borderTop: `8px solid ${c.tone}`,
                }}
              >
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', whiteSpace: 'nowrap'}}>
                  <span style={{fontSize: 23, fontWeight: 700, color: c.tone === color.ink3 ? color.ink2 : c.tone}}>{who}</span>
                  {isLead ? <span style={{fontSize: 23, fontWeight: 800, color: color.lead}}>the false lead</span> : null}
                </div>
                <div style={{fontSize: 24, fontWeight: 560, color: color.ink2, marginTop: 4, lineHeight: 1.22, height: 60}}>{c.title}</div>
                <div style={{fontSize: 30, fontWeight: 780, color: color.ink, marginTop: 6, whiteSpace: 'nowrap'}}>
                  <span style={{color: color.ok}}>✓</span> {c.check}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Stage>
  );
};
