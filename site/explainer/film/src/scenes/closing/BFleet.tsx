// b-fleet: experiment two's premise. Chapter card; the fleet of arms at three sites; a sealed
// envelope of planted problems goes into a locked drawer; the operator console the agent works
// from (results, telemetry, change log, maintenance log, the action catalogue with its
// operator-hour costs, read from the fleet's public description).
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue, useFilm, useNumber} from '../../data';
import {color, font, shadow} from '../../theme';
import {ArmIcon, LockIcon, ramp} from './parts';
import {ExperimentCard, ExperimentTag} from '../opening/ExperimentCard';

type CatalogueRow = {verb: string; target: string; hours: number};
type ChangeRow = {day: number; scope: string; change: string};

const SOURCES = ['results', 'telemetry', 'change log', 'maintenance log', 'actions'] as const;

const SourceIcon: React.FC<{kind: (typeof SOURCES)[number]}> = ({kind}) => {
  const c = color.ink2;
  if (kind === 'results')
    return (
      <svg width={48} height={36} viewBox="0 0 48 36">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <circle key={i} cx={8 + (i % 3) * 16} cy={i < 3 ? 9 : 27} r={6} fill={i === 4 ? color.bad : color.ok} />
        ))}
      </svg>
    );
  if (kind === 'telemetry')
    return (
      <svg width={48} height={36} viewBox="0 0 48 36">
        <path d="M2,26 L10,26 L16,8 L22,30 L28,14 L34,22 L46,22" fill="none" stroke={color.teal} strokeWidth={4} strokeLinejoin="round" strokeLinecap="round" />
      </svg>
    );
  if (kind === 'actions')
    return (
      <svg width={48} height={36} viewBox="0 0 48 36">
        <path d="M14,30 L30,14 M26,6 a8,8 0 1 0 10,10 l-4,-4 l-2,-6 z" fill="none" stroke={c} strokeWidth={4} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  return (
    <svg width={48} height={36} viewBox="0 0 48 36">
      <rect x={8} y={2} width={32} height={32} rx={5} fill="none" stroke={c} strokeWidth={4} />
      <path d="M15,12 L33,12 M15,19 L33,19 M15,26 L27,26" stroke={c} strokeWidth={3.5} strokeLinecap="round" />
    </svg>
  );
};

export const BFleet: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const robots = useNumber('b_robots');
  const sites = useNumber('b_sites');
  const work = film.fleet.regular_work.sites as Record<string, {robots: Record<string, unknown>}>;
  const console_ = film.fleet.console as {catalogue: CatalogueRow[]; change_log: ChangeRow[]};
  const tFleet = useCue('fleet', 0.27);
  const tPlanted = useCue('planted', 0.47);
  const tNothing = useCue('nothing', 0.53);
  const tOperator = useCue('operator', 0.62);
  const tActions = useCue('actions', 0.86);

  // one row per kind of action, the controller's where there is one
  const verbs: string[] = [];
  console_.catalogue.forEach((r) => (verbs.includes(r.verb) ? null : verbs.push(r.verb)));
  const rows = verbs.map((v) => console_.catalogue.find((r) => r.verb === v && r.target === 'controller') ?? console_.catalogue.find((r) => r.verb === v)!);

  const chapter = 1 - ramp(frame, tFleet - 16, tFleet - 2);
  const settle = ramp(frame, tFleet - 22, tFleet - 4);
  const tag = ramp(frame, tFleet - 8, tFleet + 6);
  const chapterIn = ramp(frame, 0, 16);
  const map = ramp(frame, tFleet - 4, tFleet + 14) * (1 - ramp(frame, tOperator - 14, tOperator));
  const env = ramp(frame, tPlanted - 4, tPlanted + 10, Easing.out(Easing.cubic));
  const slide = ramp(frame, tPlanted + 14, tPlanted + 40, Easing.inOut(Easing.cubic));
  const shut = ramp(frame, tNothing, tNothing + 12);
  const cons = ramp(frame, tOperator - 4, tOperator + 14, Easing.out(Easing.cubic));
  const cat = ramp(frame, tActions - 6, tActions + 12, Easing.out(Easing.cubic));
  const DR = {x: 1400, y: 500, w: 380, h: 300};

  return (
    <Stage>
      {/* the chapter card, as experiment one's, then its tag in the corner */}
      {settle < 0.99 ? (
        <div style={{position: 'absolute', left: 960 + (400 - 960) * settle, top: 470 + (120 - 470) * settle, transform: `translate(-50%, -50%) scale(${(1 - 0.6 * settle) * (0.9 + 0.1 * chapterIn)})`, opacity: chapterIn * (1 - settle)}}>
          <ExperimentCard which={2} width={820} />
        </div>
      ) : null}
      <div style={{position: 'absolute', left: 0, right: 0, top: 800, textAlign: 'center', fontSize: 44, fontWeight: 560, color: color.ink2, opacity: chapterIn * chapter}}>
        When working robots start to fail, can the agent find out why?
      </div>
      <div style={{position: 'absolute', left: 120, top: 46, opacity: tag}}>
        <ExperimentTag which={2} />
      </div>

      {/* the fleet and the sealed drawer */}
      <div style={{position: 'absolute', inset: 0, opacity: map}}>
        <div style={{position: 'absolute', left: 120, top: 128}}>
          <Headline size="h2">
            {robots} simulated arms at {sites} sites
          </Headline>
          <div style={{fontSize: 34, fontWeight: 520, color: color.ink2, marginTop: 14}}>doing pick-and-place work</div>
        </div>
        {Object.entries(work).map(([sid, s], i) => {
          const e = ramp(frame, tFleet + i * 6, tFleet + 20 + i * 6, Easing.out(Easing.cubic));
          return (
            <div
              key={sid}
              style={{position: 'absolute', left: 120 + i * 410, top: 380, width: 380, height: 300, boxSizing: 'border-box', background: color.card, borderRadius: 26, boxShadow: shadow.card, padding: '26px 28px', opacity: e, transform: `translateY(${(1 - e) * 30}px)`}}
            >
              <div style={{fontSize: 36, fontWeight: 800}}>site {sid}</div>
              <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 40}}>
                {Object.keys(s.robots).map((rid, j) => (
                  <div key={rid} style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                    <ArmIcon size={66} tint={color.ink} reach={Math.sin((frame + j * 17 + i * 31) / 22) * 0.5} />
                    <div style={{fontSize: 24, fontWeight: 600, color: color.ink3, marginTop: 8}}>{rid}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* drawer: it stays beside the console, for b-planted to open */}
      <div style={{position: 'absolute', left: DR.x, top: DR.y, width: DR.w, height: DR.h, opacity: env}}>
        <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: 200, borderRadius: 22, background: color.ink2}} />
        <div
          style={{
            position: 'absolute',
            left: 20,
            right: 80,
            bottom: 26,
            height: 120,
            borderRadius: 14,
            background: color.ink,
            transform: `translateX(${(1 - shut) * 90}px)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div style={{width: 80, height: 14, borderRadius: 7, background: '#5A606E'}} />
        </div>
        {/* the envelope slides down into the drawer */}
        <div
          style={{
            position: 'absolute',
            left: 70 + slide * 60,
            top: -160 + slide * 250,
            width: 240,
            height: 150,
            borderRadius: 12,
            background: '#FFF8EC',
            border: `4px solid ${color.lead}`,
            boxShadow: shadow.card,
            opacity: 1 - ramp(frame, tPlanted + 36, tPlanted + 44),
          }}
        >
          <svg width={232} height={142} style={{position: 'absolute', inset: 0}}>
            <path d="M2,2 L116,80 L230,2" fill="none" stroke={color.lead} strokeWidth={4} />
            <circle cx={116} cy={80} r={16} fill={color.bad} />
          </svg>
        </div>
        <div style={{position: 'absolute', left: 0, right: 0, top: -210, textAlign: 'center', fontSize: 30, fontWeight: 780, color: color.lead, opacity: 1 - ramp(frame, tPlanted + 36, tPlanted + 44)}}>planted problems</div>
        <div style={{position: 'absolute', left: 0, right: 0, top: DR.h + 20, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, opacity: shut}}>
          <LockIcon size={28} tint={color.ink} />
          <span style={{fontSize: 30, fontWeight: 760, color: color.ink}}>kept from the agent</span>
        </div>
      </div>

      {/* the operator console */}
      <div style={{position: 'absolute', left: 120, top: 128, opacity: cons}}>
        <Headline size="h2">The agent sees what an operator sees</Headline>
      </div>
      <div
        style={{
          position: 'absolute',
          left: 120,
          top: 270,
          width: 1240,
          height: 700,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: 28,
          boxShadow: shadow.frame,
          overflow: 'hidden',
          opacity: cons,
          transform: `translateY(${(1 - cons) * 40}px)`,
        }}
      >
        <div style={{height: 64, background: color.night, display: 'flex', alignItems: 'center', gap: 12, padding: '0 28px'}}>
          {[color.bad, color.lead, color.ok].map((c) => (
            <div key={c} style={{width: 16, height: 16, borderRadius: 8, background: c}} />
          ))}
          <div style={{fontFamily: font.mono, fontSize: 26, color: color.nightText, marginLeft: 18}}>operator console</div>
        </div>
        <div style={{display: 'flex', height: 636}}>
          <div style={{width: 400, background: color.paper2, padding: '30px 22px', boxSizing: 'border-box', display: 'flex', flexDirection: 'column', gap: 18}}>
            {SOURCES.map((s, i) => {
              const a = ramp(frame, tOperator + 10 + i * 14, tOperator + 24 + i * 14);
              const on = s === 'actions' ? cat : s === 'change log' ? 1 - cat : 0;
              return (
                <div
                  key={s}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 20,
                    padding: '16px 20px',
                    borderRadius: 16,
                    background: on > 0.5 ? color.card : 'transparent',
                    boxShadow: on > 0.5 ? shadow.pill : 'none',
                    opacity: a,
                    transform: `translateX(${(1 - a) * -20}px)`,
                  }}
                >
                  <SourceIcon kind={s} />
                  <span style={{fontSize: 30, fontWeight: 720, color: color.ink, whiteSpace: 'nowrap'}}>{s}</span>
                </div>
              );
            })}
          </div>
          <div style={{position: 'relative', flex: 1, padding: '34px 44px'}}>
            {/* change log (public entries, in plain words) */}
            <div style={{position: 'absolute', left: 44, right: 44, top: 34, opacity: ramp(frame, tOperator + 70, tOperator + 86) * (1 - cat)}}>
              <div style={{fontSize: 26, fontWeight: 750, color: color.ink3, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 18}}>change log</div>
              {console_.change_log.map((r, i) => (
                <div key={i} style={{display: 'flex', gap: 24, fontSize: 28, padding: '14px 0', borderBottom: `2px solid ${color.paper2}`, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap'}}>
                  <span style={{width: 100, color: color.ink3, fontWeight: 650}}>day {r.day}</span>
                  <span style={{width: 90, color: color.ink2, fontWeight: 650}}>{r.scope}</span>
                  <span style={{color: color.ink, fontWeight: 600}}>{r.change}</span>
                </div>
              ))}
            </div>
            {/* action catalogue with operator-hour costs */}
            <div style={{position: 'absolute', left: 44, right: 44, top: 34, opacity: cat, transform: `translateY(${(1 - cat) * 20}px)`}}>
              <div style={{display: 'flex', justifyContent: 'space-between', fontSize: 26, fontWeight: 750, color: color.ink3, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 18}}>
                <span>actions an operator can take</span>
                <span>hours</span>
              </div>
              {rows.map((r, i) => (
                <div key={i} style={{display: 'flex', justifyContent: 'space-between', fontSize: 34, padding: '14px 0', borderBottom: `2px solid ${color.paper2}`, opacity: ramp(frame, tActions + i * 4, tActions + 10 + i * 4)}}>
                  <span style={{color: color.ink, fontWeight: 650}}>
                    {r.verb} <span style={{color: color.ink2, fontWeight: 500}}>{r.target}</span>
                  </span>
                  <span style={{color: color.ink, fontWeight: 760, fontVariantNumeric: 'tabular-nums'}}>{r.hours} h</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Stage>
  );
};
