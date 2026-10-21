// b-planted: everything we planted, the false lead included, shown to the viewer before the
// agent's work begins. The drawer from b-fleet opens towards the viewer;
// the agent's console stays dark beneath. One plain card per site as the voice reaches it;
// beside the software update, in grey, the calibration record that changes nothing.
import React from 'react';
import {useCurrentFrame, Easing} from 'remotion';
import {Stage, Headline} from '../../components/basics';
import {useCue, useFilm} from '../../data';
import {color, font, shadow} from '../../theme';
import {ArmIcon, CameraIcon, LockIcon, ServerIcon, ramp} from './parts';
import {ExperimentTag} from '../opening/ExperimentCard';
import {PLANTED, castOf} from './planted';

const COL_W = 540;
const COL_GAP = 30;
const COL_X = [0, 1, 2].map((i) => 120 + i * (COL_W + COL_GAP));
const CARD_TOP = 300;
const CARD_H = 385;
// the drawer where b-fleet left it
const DR = {x: 1400, y: 500, w: 380, h: 300};

export const BPlanted: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const cast = castOf(film.fleet);
  const tPlanted = useCue('planted', 0.02);
  const tSite1 = useCue('site1', 0.06);
  const tStays = useCue('stays', 0.28);
  const tSite2 = useCue('site2', 0.38);
  const tSite3 = useCue('site3', 0.52);
  const tCrop = useCue('crop', 0.58);
  const tCal = useCue('calibration', 0.68);
  const tNothing = useCue('nothing', 0.79);
  const tFalse = useCue('false', 0.86);

  // the drawer opens towards the viewer, the envelope lifts out, then both give way to the cards
  const open = ramp(frame, 2, 18, Easing.out(Easing.cubic));
  const lift = ramp(frame, 12, 36, Easing.inOut(Easing.cubic));
  const drawerOut = 1 - ramp(frame, tSite1 - 14, tSite1 + 2);
  const head = ramp(frame, tPlanted, tPlanted + 14);
  const dark = ramp(frame, tSite1 - 6, tSite1 + 14);

  const plantedOf = (k: string) => PLANTED.find((p) => p.key === k)!;
  const cards = [
    {key: 'delay', at: tSite1, site: cast.siteOf.delay},
    {key: 'controller', at: tSite2, site: cast.siteOf.controller},
    {key: 'software', at: tSite3, site: cast.siteOf.software},
  ] as const;

  const siteArms = (sid: string, lit: number) => (
    <div style={{display: 'flex', gap: 22, marginTop: 18}}>
      {Object.keys(cast.work[sid]?.robots ?? {}).map((rid, j) => {
        const hot = rid === cast.lagRobot;
        return (
          <div key={rid} style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
            <ArmIcon size={50} tint={color.ink} reach={Math.sin((frame + j * 17) / 22) * 0.5} />
            <div style={{fontSize: 24, fontWeight: hot && lit > 0.5 ? 820 : 600, color: hot && lit > 0.5 ? color.violet : color.ink3, marginTop: 4}}>{rid}</div>
          </div>
        );
      })}
    </div>
  );

  const cal = plantedOf('calibration');
  const calIn = ramp(frame, tCal - 4, tCal + 14, Easing.out(Easing.cubic));
  const noEffect = ramp(frame, tNothing, tNothing + 12);
  const lead = ramp(frame, tFalse, tFalse + 12);

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={2} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 128, opacity: head, transform: `translateY(${(1 - head) * 16}px)`}}>
        <Headline size="h2">What we planted</Headline>
        <div style={{fontSize: 32, fontWeight: 520, color: color.ink2, marginTop: 10}}>shown to you, not to the agent</div>
      </div>

      {/* the drawer opens towards the viewer and the envelope comes out */}
      <div style={{position: 'absolute', left: DR.x, top: DR.y, width: DR.w, height: DR.h, opacity: drawerOut}}>
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
            transform: `translateY(${open * 70}px) scale(${1 + open * 0.12})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div style={{width: 80, height: 14, borderRadius: 7, background: '#5A606E'}} />
        </div>
        <div
          style={{
            position: 'absolute',
            left: 70,
            top: 90 - lift * 250,
            width: 240,
            height: 150,
            borderRadius: 12,
            background: '#FFF8EC',
            border: `4px solid ${color.lead}`,
            boxShadow: shadow.card,
            opacity: open,
            transform: `scale(${1 + lift * 0.15})`,
          }}
        >
          <svg width={232} height={142} style={{position: 'absolute', inset: 0}}>
            <path d={`M2,2 L116,${80 - lift * 70} L230,2`} fill="none" stroke={color.lead} strokeWidth={4} />
          </svg>
        </div>
      </div>

      {/* one card per site, as the voice reaches it */}
      {cards.map((c, i) => {
        const p = plantedOf(c.key);
        const e = ramp(frame, c.at - 6, c.at + 14, Easing.out(Easing.cubic));
        return (
          <div
            key={c.key}
            style={{
              position: 'absolute',
              left: COL_X[i],
              top: CARD_TOP,
              width: COL_W,
              height: CARD_H,
              boxSizing: 'border-box',
              background: color.card,
              borderRadius: 26,
              borderTop: `14px solid ${p.tone}`,
              boxShadow: shadow.card,
              padding: '24px 30px',
              opacity: e,
              transform: `translateY(${(1 - e) * 30}px)`,
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 52}}>
              <span style={{fontSize: 36, fontWeight: 800, color: color.ink}}>site {c.site}</span>
              {c.key === 'delay' ? (
                <span style={{display: 'flex', alignItems: 'center', gap: 10}}>
                  <ServerIcon size={44} tint={color.ink2} blink={frame / 12} />
                  <span style={{fontSize: 22, fontWeight: 650, color: color.ink2, lineHeight: 1.1, width: 130}}>customer's data centre</span>
                </span>
              ) : c.key === 'software' ? (
                <CameraIcon size={46} tint={color.ink2} />
              ) : null}
            </div>
            {siteArms(c.site, c.key === 'controller' ? e : 0)}
            <div style={{fontSize: 36, fontWeight: 760, color: color.ink, marginTop: 22, lineHeight: 1.2}}>
              {c.key === 'controller' ? (
                <>
                  <span style={{color: color.violet}}>robot {cast.lagRobot}</span>: {p.title}
                </>
              ) : (
                p.title.charAt(0).toUpperCase() + p.title.slice(1)
              )}
            </div>
            {c.key === 'delay' ? (
              <div style={{display: 'flex', alignItems: 'center', gap: 12, marginTop: 14, opacity: ramp(frame, tStays - 2, tStays + 12)}}>
                <LockIcon size={24} tint={color.lead} />
                <span style={{fontSize: 30, fontWeight: 700, color: color.lead}}>the link stays</span>
              </div>
            ) : null}
            {c.key === 'software' ? (
              <div style={{fontSize: 28, fontWeight: 600, color: color.ink2, marginTop: 12, opacity: ramp(frame, tCrop, tCrop + 12)}}>
                before the model sees them
              </div>
            ) : null}
          </div>
        );
      })}

      {/* beside the software update, in grey: the calibration record */}
      <div
        style={{
          position: 'absolute',
          left: COL_X[2],
          top: CARD_TOP + CARD_H + 28,
          width: COL_W,
          height: 230,
          boxSizing: 'border-box',
          background: color.paper2,
          borderRadius: 26,
          padding: '22px 30px',
          opacity: calIn,
          transform: `translateY(${(1 - calIn) * 30}px)`,
        }}
      >
        <div style={{fontSize: 26, fontWeight: 650, color: color.ink3}}>site {cast.siteOf.calibration}, the same day</div>
        <div style={{fontSize: 34, fontWeight: 720, color: color.ink2, marginTop: 8, lineHeight: 1.2}}>{cal.title.charAt(0).toUpperCase() + cal.title.slice(1)}</div>
        <div style={{display: 'flex', alignItems: 'baseline', gap: 22, marginTop: 14}}>
          <span style={{fontSize: 30, fontWeight: 700, color: color.ink2, opacity: noEffect}}>no effect</span>
          <span style={{fontSize: 30, fontWeight: 800, color: color.lead, opacity: lead}}>the false lead</span>
        </div>
      </div>

      {/* the agent's console stays dark */}
      <div
        style={{
          position: 'absolute',
          left: COL_X[0],
          top: CARD_TOP + CARD_H + 28,
          width: 2 * COL_W + COL_GAP,
          height: 230,
          boxSizing: 'border-box',
          background: color.night,
          borderRadius: 26,
          boxShadow: shadow.card,
          padding: '28px 36px',
          opacity: dark,
          display: 'flex',
          alignItems: 'center',
          gap: 30,
        }}
      >
        <LockIcon size={54} tint={color.nightText} />
        <div>
          <div style={{fontFamily: font.mono, fontSize: 26, color: '#7E8494'}}>the agent's console</div>
          {/* The console does show the site note and the change log; what it never shows is the cause. */}
          <div style={{fontSize: 40, fontWeight: 760, color: '#fff', marginTop: 6}}>the causes are not shown here</div>
        </div>
      </div>
    </Stage>
  );
};
