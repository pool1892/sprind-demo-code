// b-fix: every repair tested on its own before it is adopted. One experiment band per site, in
// the order of the voice: the first site's three compensation settings side
// by side, the kept one in colour; the second site's controller service; the third site's two
// rollbacks side by side, the software rollback adopted and the calibration rollback not. Each
// arm's attempts as dots, every count and setting from fleet.experiments; what was adopted
// from the round's recorded decision. "False lead" is said only in b-planted and b-result.
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Appear, Headline} from '../../components/basics';
import {useCue, useFilm} from '../../data';
import {color, shadow} from '../../theme';
import {CountDots, LockIcon, ramp} from './parts';
import {ExperimentTag} from '../opening/ExperimentCard';
import {DotLegend} from '../opening/DotLegend';
import {castOf} from './planted';

type KN = {k: number; n: number};
type Experiment = {experiment: string; change: string; without: KN; with: KN; rescued: number; broken: number};

const BAND_X = 120;
const BAND_W = 1680;
const BAND_H = 215;
const BAND_TOP = [290, 525, 760];
const BLOCK_X0 = 600; // first dot block, from the stage's left
const BLOCK_STEP = 300;
const PER_ROW = 15;

type Block = {label: string; kn: KN; at: number; status?: {text: string; ok: boolean; at: number}; extra?: {text: string; at: number}; shownAt?: number};

export const BFix: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const cast = castOf(film.fleet);
  const tTested = useCue('tested', 0.02);
  const tDelay = useCue('delay', 0.19);
  const tSettings = useCue('settings', 0.31);
  const tService = useCue('service', 0.4);
  const tThird = useCue('third', 0.49);
  const tBoth = useCue('both', 0.52);
  const tSoftware = useCue('software', 0.61);
  const tCal = useCue('calibration', 0.7);
  const tAdopts = useCue('adopts', 0.79);
  const tAlone = useCue('alone', 0.87);

  const adopted: string[] = String(film.fleet.b_round1?.value ?? '').split('; ');
  const all = film.fleet.experiments as Experiment[];
  const isAdopted = (e: Experiment) => adopted.includes(e.change);
  const stepsOf = (e: Experiment) => e.change.match(/of (\d+) control steps/)?.[1] ?? '';

  const comp = all.filter((e) => e.change.includes('latency compensation')).sort((a, b) => Number(stepsOf(a)) - Number(stepsOf(b)));
  const service = all.find((e) => e.change.includes('controller'));
  const software = all.find((e) => e.change.includes('software update'));
  const calib = all.find((e) => e.change.includes('calibration'));

  const bands: {key: string; tone: string; at: number; title: string; sub: React.ReactNode; note?: string; blocks: Block[]}[] = [
    {
      key: 'delay',
      tone: color.lead,
      at: tDelay,
      title: 'Compensate the delay',
      sub: (
        <span style={{display: 'inline-flex', alignItems: 'center', gap: 10}}>
          site {cast.linked} · <LockIcon size={20} tint={color.lead} /> the link stays
        </span>
      ),
      note: 'skips commands that are out of date on arrival',
      blocks: comp.length
        ? [
            {label: 'without', kn: comp[0].without, at: tDelay + 16},
            ...comp.map((e, i) => ({
              label: `${stepsOf(e)} control steps`,
              kn: e.with,
              at: tSettings + 6 + i * 16,
              status: isAdopted(e) ? {text: 'kept', ok: true, at: tSettings + 70} : undefined,
            })),
          ]
        : [],
    },
    {
      key: 'controller',
      tone: color.violet,
      at: tService,
      title: 'Service the controller',
      sub: <span>robot <span style={{color: color.violet, fontWeight: 800}}>{cast.lagRobot}</span>, site {cast.lagSite}</span>,
      blocks: service
        ? [
            {label: 'without', kn: service.without, at: tService + 14},
            {label: 'serviced', kn: service.with, at: tService + 34, status: isAdopted(service) ? {text: 'adopted', ok: true, at: tService + 70} : undefined},
          ]
        : [],
    },
    {
      key: 'third',
      tone: color.teal,
      at: tThird,
      title: 'Undo each change in its own test',
      sub: <span>site {cast.third} · two changes on day {cast.day}</span>,
      blocks:
        software && calib
          ? [
              {label: 'without', kn: software.without, at: tThird + 14},
              {label: 'software rolled back', kn: software.with, at: tSoftware + 4, shownAt: tBoth, status: {text: isAdopted(software) ? 'adopted' : 'not adopted', ok: isAdopted(software), at: tAdopts}},
              {
                label: 'calibration rolled back',
                kn: calib.with,
                at: tCal + 4,
                shownAt: tBoth + 8,
                extra: {text: `${calib.rescued} rescued · ${calib.broken} broken`, at: tCal + 44},
                status: {text: isAdopted(calib) ? 'adopted' : 'not adopted', ok: isAdopted(calib), at: tAlone},
              },
            ]
          : [],
    },
  ];

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={2} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 128}}>
        <Appear delay={tTested}>
          <Headline size="h2">Each repair is tested on its own</Headline>
        </Appear>
        <Appear delay={tTested + 10}>
          <div style={{display: 'flex', alignItems: 'center', gap: 36, marginTop: 12}}>
            <span style={{fontSize: 30, fontWeight: 520, color: color.ink2}}>attempts that succeeded, without the change and with it</span>
            <DotLegend small wording="attempts" />
          </div>
        </Appear>
      </div>

      {bands.map((b, bi) => {
        const p = ramp(frame, b.at - 6, b.at + 14, Easing.out(Easing.cubic));
        return (
          <div
            key={b.key}
            style={{
              position: 'absolute',
              left: BAND_X,
              top: BAND_TOP[bi],
              width: BAND_W,
              height: BAND_H,
              boxSizing: 'border-box',
              background: color.card,
              borderRadius: 24,
              borderLeft: `14px solid ${b.tone}`,
              boxShadow: shadow.card,
              opacity: p,
              transform: `translateY(${(1 - p) * 30}px)`,
            }}
          >
            <div style={{position: 'absolute', left: 30, top: 26, width: 420}}>
              <div style={{fontSize: 36, fontWeight: 800, color: color.ink, lineHeight: 1.1}}>{b.title}</div>
              <div style={{fontSize: 27, fontWeight: 650, color: color.ink2, marginTop: 10}}>{b.sub}</div>
              {b.note ? <div style={{fontSize: 25, fontWeight: 560, color: color.ink2, marginTop: 10, lineHeight: 1.25}}>{b.note}</div> : null}
            </div>
            {b.blocks.map((k, i) => {
              const show = ramp(frame, (k.shownAt ?? k.at) - 6, (k.shownAt ?? k.at) + 10);
              const done = frame >= k.at + k.kn.n * 0.5;
              const st = k.status ? ramp(frame, k.status.at, k.status.at + 12, Easing.out(Easing.cubic)) : 0;
              const keptTone = k.status?.ok ? color.ok : color.ink2;
              const dim = k.status && !k.status.ok ? 1 - 0.35 * st : 1;
              const isBase = k.label === 'without';
              return (
                <div key={i} style={{position: 'absolute', left: BLOCK_X0 - BAND_X + i * BLOCK_STEP, top: 16, width: 270, opacity: show * dim}}>
                  <div style={{fontSize: 25, fontWeight: 700, whiteSpace: 'nowrap', color: st > 0.5 && k.status?.ok ? color.ok : isBase ? color.ink3 : color.ink2}}>{k.label}</div>
                  <div style={{marginTop: 10, height: 76}}>
                    <CountDots k={k.kn.k} n={k.kn.n} start={k.at} perRow={PER_ROW} size={11} gap={4} every={0.5} />
                  </div>
                  <div style={{display: 'flex', alignItems: 'baseline', gap: 14, marginTop: 8, whiteSpace: 'nowrap'}}>
                    <span style={{fontSize: 32, fontWeight: 820, fontVariantNumeric: 'tabular-nums', color: st > 0.5 && k.status?.ok ? color.ok : color.ink, opacity: done ? 1 : 0}}>
                      {k.kn.k} <span style={{fontSize: 23, fontWeight: 600, color: color.ink2}}>of {k.kn.n}</span>
                    </span>
                    {k.status ? <span style={{fontSize: 27, fontWeight: 800, color: keptTone, opacity: st}}>{k.status.ok ? `✓ ${k.status.text}` : k.status.text}</span> : null}
                  </div>
                  {k.extra ? (
                    <div style={{fontSize: 24, fontWeight: 650, color: color.ink2, marginTop: 2, whiteSpace: 'nowrap', opacity: ramp(frame, k.extra.at, k.extra.at + 12)}}>{k.extra.text}</div>
                  ) : null}
                </div>
              );
            })}
          </div>
        );
      })}
    </Stage>
  );
};
