// b-symptom: what the agent sees, symptoms only. Success by robot in the regular work: the
// first site's four robots and one robot at the second site low, the third site's robots in
// between, the healthy robots high; the same lag mark on the first site's robots and on the
// lagging robot (its label coloured, nothing drawn around it); the change log with the third
// site's two entries on one day; then the two questions.
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue, useFilm} from '../../data';
import {color, shadow} from '../../theme';
import {LagDot, ramp} from './parts';
import {ExperimentTag} from '../opening/ExperimentCard';
import {castOf} from './planted';

type ChangeRow = {day: number; scope: string; change: string};

const GROUP_W = 540;
const GROUP_GAP = 30;
const BASE = 640; // the bars' baseline
const BAR_H = 230; // 100%
const BAR_W = 72;

const pct = (k: number, n: number) => `${Math.round((100 * k) / Math.max(1, n))}%`;

export const BSymptom: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const cast = castOf(film.fleet);
  const log = film.fleet.console.change_log as ChangeRow[];
  const siteIds = Object.keys(cast.work);

  const tNone = useCue('none', 0.02);
  const tLinked = useCue('linked', 0.2);
  const tTrail = useCue('trail', 0.26);
  const tOther = useCue('other', 0.37);
  const tThird = useCue('third', 0.48);
  const tLog = useCue('log', 0.63);
  const tSame = useCue('same', 0.75);
  const tWhich = useCue('which', 0.83);

  const headIn = ramp(frame, tNone, tNone + 14);
  const head = headIn * (1 - ramp(frame, tSame - 10, tSame));
  const fill = ramp(frame, tNone + 50, tNone + 100, Easing.out(Easing.cubic));
  const chart = ramp(frame, tNone + 36, tNone + 54);
  const s1 = ramp(frame, tLinked, tLinked + 14);
  const lag1 = ramp(frame, tTrail, tTrail + 14);
  const lag2 = ramp(frame, tOther + 6, tOther + 20);
  const third = ramp(frame, tThird, tThird + 14);
  const logIn = ramp(frame, tLog - 6, tLog + 12, Easing.out(Easing.cubic));
  const same = ramp(frame, tSame - 4, tSame + 12);
  const which = ramp(frame, tWhich - 4, tWhich + 12);

  const toneOf = (sid: string, rid: string) => {
    if (sid === cast.linked) return {c: color.lead, on: s1};
    if (rid === cast.lagRobot) return {c: color.violet, on: lag2};
    if (sid === cast.third) return {c: color.teal, on: third};
    return {c: color.ok, on: third};
  };

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={2} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 128, opacity: head, transform: `translateY(${(1 - headIn) * 16}px)`}}>
        <Headline size="h2">What the agent sees</Headline>
        <div style={{fontSize: 32, fontWeight: 520, color: color.ink2, marginTop: 10}}>the operating records: each robot's share of attempts that succeeded</div>
      </div>
      <div style={{position: 'absolute', left: 120, top: 116}}>
        <Headline size="h2" style={{opacity: same, transform: `translateY(${(1 - same) * 14}px)`}}>
          Same symptom, same cause?
        </Headline>
        <Headline size="h2" style={{color: color.teal, marginTop: 8, opacity: which, transform: `translateY(${(1 - which) * 14}px)`}}>
          Which of the two changes is to blame?
        </Headline>
      </div>

      {/* success by robot */}
      <div style={{position: 'absolute', inset: 0, opacity: chart}}>
        <div style={{position: 'absolute', left: 120, width: 1680, top: BASE, height: 3, background: color.line}} />
        {siteIds.map((sid, gi) => {
          const gx = 120 + gi * (GROUP_W + GROUP_GAP);
          const robots = Object.entries(cast.work[sid].robots);
          return (
            <React.Fragment key={sid}>
              {robots.map(([rid, c], j) => {
                const cx = gx + 75 + j * 130;
                const h = BAR_H * (c.k / c.n) * fill;
                const t = toneOf(sid, rid);
                const lit = sid === cast.linked ? lag1 : rid === cast.lagRobot ? lag2 : 0;
                const named = rid === cast.lagRobot && lag2 > 0.5;
                return (
                  <React.Fragment key={rid}>
                    <div style={{position: 'absolute', left: cx - BAR_W / 2, top: BASE - h, width: BAR_W, height: h, borderRadius: '10px 10px 0 0', background: t.on > 0.5 ? t.c : color.ink3}} />
                    <div style={{position: 'absolute', left: cx - 60, width: 120, top: BASE - h - 42, textAlign: 'center', fontSize: 28, fontWeight: 760, color: t.on > 0.5 ? t.c : color.ink2, fontVariantNumeric: 'tabular-nums', opacity: fill}}>
                      {pct(c.k, c.n)}
                    </div>
                    {lit > 0 ? (
                      <div style={{position: 'absolute', left: cx - 19, top: BASE - BAR_H * (c.k / c.n) - 96}}>
                        <LagDot show={lit} size={38} />
                      </div>
                    ) : null}
                    <div style={{position: 'absolute', left: cx - 60, width: 120, top: BASE + 12, textAlign: 'center', fontSize: 26, fontWeight: named ? 820 : 620, color: named ? color.violet : color.ink2}}>{rid}</div>
                  </React.Fragment>
                );
              })}
              <div style={{position: 'absolute', left: gx, width: GROUP_W, top: BASE + 52, textAlign: 'center', fontSize: 30, fontWeight: 780, color: color.ink}}>site {sid}</div>
            </React.Fragment>
          );
        })}
        <div style={{position: 'absolute', right: 120, top: 296, display: 'flex', alignItems: 'center', gap: 28, fontSize: 24, fontWeight: 600, color: color.ink3, opacity: lag1}}>
          <span style={{display: 'flex', alignItems: 'center', gap: 12}}>
            <LagDot show={1} size={28} /> trails its commands
          </span>
          <span style={{display: 'flex', alignItems: 'center', gap: 10, opacity: third}}>
            <span style={{width: 22, height: 22, borderRadius: 5, background: color.ok}} /> healthy robots
          </span>
        </div>
      </div>

      {/* the change log, the third site's two same-day entries in colour */}
      <div
        style={{
          position: 'absolute',
          left: 120,
          top: 790,
          width: 1680,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: 22,
          boxShadow: shadow.card,
          padding: '16px 36px',
          opacity: logIn,
          transform: `translateY(${(1 - logIn) * 24}px)`,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          columnGap: 60,
        }}
      >
        <div style={{gridColumn: '1 / span 2', fontSize: 24, fontWeight: 750, letterSpacing: '0.12em', textTransform: 'uppercase', color: color.ink3, marginBottom: 6}}>change log</div>
        {log.map((r, i) => {
          const hot = r.scope === cast.third && r.day === cast.day;
          return (
            <div key={i} style={{display: 'flex', gap: 24, fontSize: 28, padding: '7px 0', whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums', color: hot ? color.teal : color.ink3, fontWeight: hot ? 780 : 560}}>
              <span style={{width: 90}}>day {r.day}</span>
              <span style={{width: 80}}>{r.scope}</span>
              <span>{r.change}</span>
            </div>
          );
        })}
      </div>
    </Stage>
  );
};
