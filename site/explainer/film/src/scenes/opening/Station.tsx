// station: the test bench, real. The recorded station (reserved start 31) held on its first
// frame with a slow push-in, labels pinned from its recorded track (the stove by hand), a
// "physics simulator" tag;
// then the diagram: camera images + a written instruction → X-VLA, "open robot
// vision-language-action model" → arm motion, and the
// padlock: the model is fixed for these experiments.
// At "into motion" the recorded run plays, in the main frame and the camera thumbnails alike.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage, Arrow} from '../../components/basics';
import {Footage, type Segment} from '../../components/footage';
import {useCue, useNumber} from '../../data';
import {color, font, shadow, type} from '../../theme';
import {arrive, HandPin, Padlock, span} from './kit';

const MAIN = 'r1-job-31-establishing';
const CAM = 'r1-chain-31-scene';
const WRIST = 'r1-chain-31-wrist';
const BOX = 820;
const BX = 110;
const BY = 150;
const RX = 1010; // the right column
const RW = 800;

// labels pinned to the clip's recorded track (object name), or by hand in clip pixels where
// the track has no such object (the stove)
const PINS: {label: string; object?: string; x?: number; y?: number; dx: number; dy: number}[] = [
  {label: 'bowl', object: 'akita_black_bowl_1', dx: -60, dy: -100},
  {label: 'plate', object: 'plate_1', dx: -70, dy: 76},
  {label: 'cream cheese', object: 'cream_cheese_1', dx: 40, dy: 80},
  {label: 'stove', x: 1104, y: 660, dx: -30, dy: -120},
  {label: 'cabinet', object: 'wooden_cabinet_1_top_side', dx: -60, dy: -80},
  {label: 'drawers', object: 'wooden_cabinet_1_middle_region', dx: -70, dy: 60},
];

const PLAY: Segment[] = [{from: 0, to: 70, rate: 1}];

export const Station: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cSim = useCue('sim', 0.03);
  const cBench = useCue('bench', 0.18);
  const cModel = useCue('model', 0.38);
  const cSees = useCue('sees', 0.64);
  const cMotion = useCue('motion', 0.8);
  const cLock = useCue('lock', 0.87);
  const cWhy = [useCue('repeat', 0.25), useCue('measured', 0.33), useCue('kitchen', 0.39)];
  const instruction = useNumber('r1_step0_instruction');

  const boxIn = arrive(frame, fps, 0);
  const simP = arrive(frame, fps, cSim + 4, 'lively');
  const pinsOut = span(frame, cMotion - 6, cMotion + 6);
  const phase1 = arrive(frame, fps, cBench - 4) * (1 - span(frame, cModel - 10, cModel + 4));
  const vla = arrive(frame, fps, cModel);
  const sees = arrive(frame, fps, cSees);
  const motion = arrive(frame, fps, cMotion);
  const lock = arrive(frame, fps, cLock, 'lively');
  const lockDrop = Math.min(1.1, lock);

  return (
    <Stage act="the test bench">
      {/* the station, real */}
      <div style={{position: 'absolute', left: BX, top: BY, opacity: boxIn, transform: `scale(${0.97 + 0.03 * boxIn})`}}>
        <Footage
          clip={MAIN}
          segments={PLAY}
          startAt={cMotion}
          width={BOX}
          height={BOX}
          zoom={{from: 1.2, to: 1.5, at: [0, Math.max(cMotion, 60)], focus: {x: 0.45, y: 0.6}}}
          label={
            <div style={{opacity: Math.min(1, simP), transform: `scale(${0.85 + 0.15 * Math.min(1, simP)})`, transformOrigin: 'left top', background: color.night, color: '#fff', padding: '10px 22px', borderRadius: 999, fontSize: 28, fontWeight: 720, boxShadow: shadow.pill, whiteSpace: 'nowrap'}}>
              physics simulator
            </div>
          }
        >
          {(ctx) => (
            <>
              {PINS.map((p, i) => {
                const o = p.object ? ctx.at?.objects[p.object] : null;
                if (p.object && (!o || !o.visible)) return null;
                const x = o ? o.x : (p.x ?? 0);
                const y = o ? o.y : (p.y ?? 0);
                return (
                  <HandPin key={p.label} ctx={ctx} x={x} y={y} dx={p.dx} dy={p.dy} show={arrive(frame, fps, cBench + 2 + i * 9) * (1 - pinsOut)}>
                    {p.label}
                  </HandPin>
                );
              })}
            </>
          )}
        </Footage>
      </div>

      {/* phase 1: what this is */}
      <div style={{position: 'absolute', left: RX, top: 200, width: RW, opacity: phase1, transform: `translateY(${(1 - phase1) * 24}px)`}}>
        <div style={{...type.kicker, marginBottom: 22}}>the test bench</div>
        <div style={{...type.h1, fontSize: 80}}>A kitchen scene</div>
        <div style={{...type.body, fontSize: 36, marginTop: 18, maxWidth: 760}}>from a public robotics benchmark</div>
        <div style={{display: 'flex', flexDirection: 'column', gap: 20, marginTop: 46}}>
          {['every start repeatable, exactly', 'every motion measured', 'failures easy to follow'].map((t, i) => {
            const p = arrive(frame, fps, cWhy[i]);
            return (
              <div key={t} style={{display: 'flex', alignItems: 'center', gap: 20, opacity: p, transform: `translateX(${(1 - p) * -20}px)`}}>
                <div style={{width: 44, height: 44, borderRadius: '50%', background: color.teal, color: '#fff', fontSize: 26, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>✓</div>
                <div style={{fontSize: 38, fontWeight: 650, color: color.ink}}>{t}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* phase 2: the model, as a diagram */}
      <div style={{position: 'absolute', left: RX, top: 150, width: RW, opacity: sees, transform: `translateY(${(1 - sees) * 24}px)`}}>
        <div style={{...type.kicker, fontSize: 24, marginBottom: 14}}>camera images + a written instruction</div>
        <div style={{display: 'flex', gap: 18, alignItems: 'stretch'}}>
          <Footage clip={CAM} segments={PLAY} startAt={cMotion} width={190} height={190} radius={18} />
          <Footage clip={WRIST} segments={PLAY} startAt={cMotion} width={190} height={190} radius={18} />
          <div
            style={{
              flex: 1,
              background: color.card,
              borderRadius: 18,
              boxShadow: shadow.card,
              padding: '22px 26px',
              display: 'flex',
              alignItems: 'center',
              fontFamily: font.mono,
              fontSize: 28,
              lineHeight: 1.4,
              color: color.ink,
            }}
          >
            “{instruction}”
          </div>
        </div>
      </div>
      <div style={{position: 'absolute', inset: 0, opacity: sees}}>
        {sees > 0.02 ? <Arrow x1={RX + RW / 2} y1={398} x2={RX + RW / 2} y2={460} delay={cSees + 6} stroke={color.ink3} width={5} /> : null}
      </div>

      <div
        style={{
          position: 'absolute',
          left: RX,
          top: 478,
          width: RW,
          boxSizing: 'border-box',
          background: color.night,
          borderRadius: 24,
          boxShadow: shadow.card,
          padding: '30px 38px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          opacity: vla,
          transform: `translateY(${(1 - vla) * 30}px) scale(${0.96 + 0.04 * vla})`,
        }}
      >
        <div>
          <div style={{fontSize: 72, fontWeight: 820, letterSpacing: '-0.03em', color: '#fff', lineHeight: 1}}>X-VLA</div>
          <div style={{fontSize: 30, fontWeight: 520, color: color.nightText, marginTop: 12, lineHeight: 1.25, whiteSpace: 'nowrap'}}>
            open robot
            <br />
            vision-language-action model
          </div>
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 18, opacity: Math.min(1, lock * 1.5)}}>
          <div style={{fontSize: 26, fontWeight: 720, color: '#fff', lineHeight: 1.2, textAlign: 'right', opacity: span(frame, cLock + 8, cLock + 18)}}>fixed for these<br />experiments</div>
          <div style={{transform: `translateY(${(1 - lockDrop) * -90}px) rotate(${(1 - Math.min(1, lock)) * -14}deg)`}}>
            <Padlock size={70} tone={color.lead} closed={span(frame, cLock + 6, cLock + 14)} />
          </div>
        </div>
      </div>

      <div style={{position: 'absolute', inset: 0, opacity: motion}}>
        {motion > 0.02 ? <Arrow x1={RX + RW / 2} y1={694} x2={RX + RW / 2} y2={754} delay={cMotion} stroke={color.ink3} width={5} /> : null}
      </div>
      <div
        style={{
          position: 'absolute',
          left: RX,
          top: 770,
          width: RW,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: 24,
          boxShadow: shadow.card,
          padding: '26px 38px',
          display: 'flex',
          alignItems: 'center',
          gap: 22,
          opacity: motion,
          transform: `translateY(${(1 - motion) * 30}px)`,
        }}
      >
        <div style={{fontSize: 48, color: color.ink, fontWeight: 760}}>←</div>
        <div style={{fontSize: 44, fontWeight: 780, color: color.ink}}>arm motion</div>
      </div>
    </Stage>
  );
};
