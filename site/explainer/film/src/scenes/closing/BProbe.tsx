// b-probe: the agent's instrument. A hand steps out and back with the model switched off; then
// the recorded probe: commanded against measured hand position for the linked site's arm,
// the lagging robot elsewhere and a healthy arm, aligned in time. Delays are read from the
// recorded steps, never drawn by hand.
import React from 'react';
import {interpolate, useCurrentFrame, Easing} from 'remotion';
import {Stage, Appear, Headline, Pill} from '../../components/basics';
import {useCue, useFilm} from '../../data';
import {color, font, shadow} from '../../theme';
import {HandIcon, clamp, ramp} from './parts';
import {ExperimentTag} from '../opening/ExperimentCard';

type ProbeRobot = {site: string; commanded_cm: number[]; measured_cm: number[]};

/** Each step edge of the command and how many control steps the hand took to move 1 mm after it. */
const edgesOf = (r: ProbeRobot) => {
  const out: {edge: number; delay: number; from: number; to: number}[] = [];
  const c = r.commanded_cm;
  for (let i = 1; i < c.length; i++) {
    if (c[i] !== c[i - 1]) {
      let j = i;
      while (j < r.measured_cm.length && Math.abs(r.measured_cm[j] - c[i - 1]) < 0.1) j++;
      out.push({edge: i, delay: j - i, from: c[i - 1], to: c[i]});
    }
  }
  return out;
};

const ROWS = [
  {id: 'r01', tone: color.lead, title: 'site', caption: 'starts late, by the same amount', meaning: 'a delay on the line'},
  {id: 'r06', tone: color.violet, title: 'site', caption: 'falls further behind', meaning: "its controller can't keep up"},
  {id: 'r08', tone: color.ok, title: 'healthy, site', caption: 'responds at once', meaning: ''},
] as const;

const secs = (steps: number, stepS: number) => `${parseFloat((steps * stepS).toFixed(2))} s`;

const PLOT = {x: 930, w: 860, rowTop: 300, rowH: 228, padT: 34, h: 150};

export const BProbe: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const probe = film.fleet.probe as {robots: Record<string, ProbeRobot>; step_s: number; origin_cm: number; axis_s: number[]};
  const stepS = probe.step_s;
  const tProbe = useCue('probe', 0.12);
  const tMotion = useCue('motion', 0.26);
  const tLinked = useCue('linked', 0.47);
  const tBacklog = useCue('backlog', 0.73);
  const tTell = useCue('tell', 0.02);
  const tDelay = useCue('delay', 0.55);
  const tBehind = useCue('behind', 0.7);
  const tHealthy = tBehind + 56;

  // x range: from the start to a little after the slowest return, read from the records
  const all = ROWS.map((r) => probe.robots[r.id]).filter(Boolean);
  const lastMove = Math.max(...all.map((r) => {
    const e = edgesOf(r);
    const back = e[e.length - 1];
    return back ? back.edge + back.delay : r.commanded_cm.length;
  }));
  const maxStep = Math.min(all[0]?.commanded_cm.length ?? 200, lastMove + 34);
  const peak = Math.max(...all.flatMap((r) => r.commanded_cm));
  const X = (s: number) => PLOT.x + (s / maxStep) * PLOT.w;
  const reveal = interpolate(frame, [tMotion + 6, tLinked - 8], [0, maxStep], {...clamp, easing: Easing.inOut(Easing.quad)});
  const chartIn = ramp(frame, tMotion - 6, tMotion + 10);

  const focusOf = (id: string) => {
    if (frame < tLinked) return 1;
    if (frame < tBacklog) return id === 'r01' ? 1 : 0.3;
    if (frame < tHealthy) return id === 'r06' ? 1 : id === 'r01' ? 0.55 : 0.3;
    return 1;
  };
  const captionAt: Record<string, number> = {r01: tLinked + 6, r06: tBacklog + 6, r08: tHealthy};
  const meaningAt: Record<string, number> = {r01: tDelay, r06: tBehind, r08: 1e9};

  // the instrument diagram: hold, step out, hold, step back, looping
  const cyc = 72;
  const t = ((frame - tProbe) % cyc + cyc) % cyc;
  const pos = t < 12 ? 0 : t < 24 ? Easing.inOut(Easing.cubic)((t - 12) / 12) : t < 48 ? 1 : t < 60 ? 1 - Easing.inOut(Easing.cubic)((t - 48) / 12) : 0;
  const card = ramp(frame, tProbe, tProbe + 16, Easing.out(Easing.cubic));

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={2} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 128}}>
        <Appear delay={tTell}>
          <Headline size="h2">The task records can't tell them apart</Headline>
        </Appear>
        <Appear delay={tProbe}>
          <div style={{fontSize: 34, fontWeight: 560, color: color.agent, marginTop: 12}}>So the agent builds an instrument</div>
        </Appear>
      </div>

      {/* the instrument */}
      <div
        style={{
          position: 'absolute',
          left: 120,
          top: 330,
          width: 400,
          background: color.card,
          borderRadius: 26,
          boxShadow: shadow.card,
          borderTop: `14px solid ${color.agent}`,
          padding: '30px 34px 34px',
          boxSizing: 'border-box',
          opacity: card,
          transform: `translateY(${(1 - card) * 30}px)`,
        }}
      >
        <div style={{fontSize: 36, fontWeight: 800, color: color.ink, lineHeight: 1.15}}>A test motion</div>
        <div style={{fontSize: 28, fontWeight: 520, color: color.ink2, marginTop: 8}}>hold · step out · step back</div>
        <div style={{position: 'relative', height: 200, marginTop: 26}}>
          {/* rail with its two marks */}
          <div style={{position: 'absolute', left: 40, right: 40, top: 128, height: 8, borderRadius: 4, background: color.line}} />
          {[0, 1].map((m) => (
            <div key={m} style={{position: 'absolute', left: 40 + m * 240, top: 146, transform: 'translateX(-50%)', fontSize: 26, fontWeight: 700, color: color.ink2, whiteSpace: 'nowrap'}}>
              {m === 0 ? `${probe.origin_cm}` : `${peak} cm`}
            </div>
          ))}
          <div style={{position: 'absolute', left: 40 + pos * 240 - 50, top: 10}}>
            <HandIcon size={100} tint={color.ink} />
          </div>
        </div>
        <Pill tone="teal" size={28} style={{marginTop: 4}}>
          model switched off
        </Pill>
      </div>

      {/* legend */}
      <div style={{position: 'absolute', left: PLOT.x, top: 238, display: 'flex', gap: 40, alignItems: 'center', opacity: chartIn, fontSize: 26, fontWeight: 650, color: color.ink2}}>
        <span style={{display: 'flex', alignItems: 'center', gap: 12}}>
          <span style={{width: 40, height: 5, background: color.ink, borderRadius: 3}} /> commanded
        </span>
        <span style={{display: 'flex', alignItems: 'center', gap: 12}}>
          <span style={{width: 40, height: 7, background: color.ink3, borderRadius: 3}} /> where the hand is
        </span>
      </div>

      {/* rows */}
      {ROWS.map((row, ri) => {
        const r = probe.robots[row.id];
        if (!r) return null;
        const top = PLOT.rowTop + ri * PLOT.rowH;
        const Y = (v: number) => top + PLOT.padT + PLOT.h - (v / peak) * PLOT.h;
        const path = (vals: number[]) => {
          const pts: string[] = [];
          for (let s = 0; s <= Math.min(reveal, maxStep); s++) pts.push(`${pts.length ? 'L' : 'M'}${X(s).toFixed(1)},${Y(vals[s]).toFixed(1)}`);
          return pts.join(' ');
        };
        const f = focusOf(row.id);
        const cap = ramp(frame, captionAt[row.id], captionAt[row.id] + 14);
        const edges = edgesOf(r);
        const rowIn = ramp(frame, tMotion - 4 + ri * 5, tMotion + 12 + ri * 5);
        return (
          <div key={row.id} style={{position: 'absolute', inset: 0, opacity: rowIn * f}}>
            <div style={{position: 'absolute', left: 560, top: top + 12, width: 350}}>
              <div style={{display: 'flex', alignItems: 'baseline', gap: 12, whiteSpace: 'nowrap'}}>
                <span style={{fontSize: 40, fontWeight: 820, color: row.tone, lineHeight: 1}}>{row.id}</span>
                <span style={{fontSize: 26, fontWeight: 600, color: color.ink2}}>
                  {row.title} {r.site}
                </span>
              </div>
              <div style={{opacity: cap, transform: `translateY(${(1 - cap) * 10}px)`}}>
                <div style={{fontSize: 28, fontWeight: 780, color: color.ink, marginTop: 10, lineHeight: 1.2}}>{row.caption}</div>

                {row.id !== 'r08' && edges.length >= 2 ? (
                  <div style={{fontSize: 26, fontWeight: 650, color: row.tone, marginTop: 6, whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums'}}>
                    late {secs(edges[0].delay, stepS)} out · {secs(edges[edges.length - 1].delay, stepS)} back
                  </div>
                ) : null}
              </div>
            </div>
            <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible'}}>
              <rect x={PLOT.x - 16} y={top + 8} width={PLOT.w + 32} height={PLOT.h + 54} rx={18} fill={color.card} opacity={0.75} />
              <line x1={PLOT.x} x2={PLOT.x + PLOT.w} y1={Y(0)} y2={Y(0)} stroke={color.line} strokeWidth={2} />
              <path d={path(r.commanded_cm)} fill="none" stroke={color.ink} strokeWidth={4} strokeLinejoin="round" />
              <path d={path(r.measured_cm)} fill="none" stroke={row.tone} strokeWidth={7} strokeLinejoin="round" strokeLinecap="round" />
              {/* the command, shifted by the first move's delay: it lands on the hand's path for a
                  fixed delay, and falls short at the step back for a growing backlog */}
              {row.id !== 'r08' && edges.length > 0 && frame >= meaningAt[row.id] - 2 ? (
                <path
                  d={path(r.commanded_cm.map((_, s) => r.commanded_cm[Math.max(0, Math.round(s - edges[0].delay * ramp(frame, meaningAt[row.id], meaningAt[row.id] + 30, Easing.inOut(Easing.cubic))))]))}
                  fill="none"
                  stroke={row.tone}
                  strokeWidth={4}
                  strokeDasharray="12 9"
                  opacity={0.95}
                />
              ) : null}
              {row.meaning && edges.length >= 2 ? (
                <text
                  x={row.id === 'r06' ? (X(edges[0].edge + edges[0].delay) + X(edges[edges.length - 1].edge)) / 2 + 10 : X((edges[0].edge + edges[edges.length - 1].edge) / 2)}
                  y={Y(peak * (row.id === 'r06' ? 0.16 : 0.38))}
                  textAnchor="middle"
                  fontSize={row.id === 'r06' ? 30 : 34}
                  fontWeight={820}
                  fill={row.tone}
                  fontFamily={font.sans}
                  opacity={ramp(frame, meaningAt[row.id] + 10, meaningAt[row.id] + 24)}
                >
                  {row.meaning}
                </text>
              ) : null}
              {/* the delay after each command edge, read from the recorded steps */}
              {row.id !== 'r08'
                ? edges.map((e, k) => {
                    const growing = row.id === 'r06' && k === edges.length - 1;
                    const at = growing ? meaningAt[row.id] + 24 : captionAt[row.id] + k * 10;
                    const p = ramp(frame, at, at + (growing ? 45 : 12), Easing.out(Easing.cubic));
                    if (p <= 0 || e.edge + e.delay > reveal) return null;
                    const xa = X(e.edge);
                    const xb = X(e.edge + e.delay);
                    const y = Y(peak / 2);
                    return (
                      <g key={k} opacity={p}>
                        <line x1={xa} x2={xa + (xb - xa) * p} y1={y} y2={y} stroke={row.tone} strokeWidth={5} strokeLinecap="round" />
                        <line x1={xa} x2={xa} y1={y - 14} y2={y + 14} stroke={color.ink} strokeWidth={4} strokeLinecap="round" />
                        <line x1={xb} x2={xb} y1={y - 14} y2={y + 14} stroke={color.ink} strokeWidth={4} strokeLinecap="round" opacity={p > 0.9 ? 1 : 0} />
                      </g>
                    );
                  })
                : null}
            </svg>
          </div>
        );
      })}
      {/* time axis under the last row */}
      <div style={{opacity: chartIn}}>
        {probe.axis_s.map((secs, k) => {
          const s = secs / stepS;
          if (s > maxStep) return null;
          return (
            <div key={k} style={{position: 'absolute', left: X(s), top: PLOT.rowTop + 3 * PLOT.rowH - 6, transform: 'translateX(-50%)', fontSize: 24, fontWeight: 600, color: color.ink3}}>
              {secs} s
            </div>
          );
        })}
      </div>
    </Stage>
  );
};
