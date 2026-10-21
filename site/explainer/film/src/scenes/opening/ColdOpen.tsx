// cold-open: each skill works, the job doesn't. Three recorded clips (the cabinet skill alone,
// the drawer skill alone, both chained on reserved start 31), each starting on its cue; the
// chained run ends on the bowl fallen (fall at half speed). Then the plain chain's 20 reserved
// runs as dots, in start order, and the count. No ring on the bowl.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage, Pill} from '../../components/basics';
import {Dot} from '../../components/dots';
import {Footage, segmentsFrames, useClip, type Segment} from '../../components/footage';
import {useCue, useFilm, useNumber} from '../../data';
import {color, durations, type} from '../../theme';
import {arrive, span} from './kit';

// two smaller skill clips stacked on the left, the chained job large on the right (16:9 frames)
const SW = 580;
const SH = Math.round((SW * 9) / 16);
const BW = 1050;
const BH = Math.round((BW * 9) / 16);
const TOP = 150;
const SLOTS = [
  {x: 110, y: TOP, w: SW, h: SH},
  {x: 110, y: TOP + SH + 80, w: SW, h: SH},
  {x: 1920 - 110 - BW, y: TOP, w: BW, h: BH},
];

const fullSegs = (frames: number, every: number, from: number, rate: number): Segment[] => [{from, to: (frames - 1) * every, rate}];

const Slot: React.FC<{
  i: number;
  at: number;
  kicker: string;
  pillAt: number;
  pill: React.ReactNode;
  tone: 'ok' | 'bad';
  children: React.ReactNode;
}> = ({i, at, kicker, pillAt, pill, tone, children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = arrive(frame, fps, at);
  const q = arrive(frame, fps, pillAt, 'lively');
  const {x, y, w, h} = SLOTS[i];
  return (
    <>
      <div style={{position: 'absolute', left: x, top: y - 50, ...type.kicker, fontSize: 24, opacity: p}}>{kicker}</div>
      <div style={{position: 'absolute', left: x, top: y, opacity: Math.min(1, p * 1.3), transform: `translateY(${(1 - p) * 40}px) scale(${0.96 + 0.04 * p})`}}>
        {children}
      </div>
      <div style={{position: 'absolute', left: x + 24, top: i === 2 ? y + 24 : y + h - 82, opacity: Math.min(1, q * 1.5), transform: `translateY(${(1 - Math.min(q, 1)) * 18}px) scale(${0.85 + 0.15 * q})`, transformOrigin: 'left center'}}>
        <Pill tone={tone} size={i === 2 ? 34 : 28}>
          {pill}
        </Pill>
      </div>
    </>
  );
};

export const ColdOpen: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const film = useFilm();
  const c1 = useCue('skill1', 0.04);
  const c2 = useCue('skill2', 0.29);
  const c3 = useCue('both', 0.44);
  const cCount = useCue('count', 0.54);
  const cNum = useCue('number', 0.79);
  const naive = useNumber('r1_naive_delivered');
  const n = useNumber('r1_release_n');

  const cab = useClip('single-cabinet-hero');
  const drw = useClip('single-drawer-hero');
  const chain = useClip('r1-chain-31-hero');

  // the single skills play fast enough to finish before the next cue
  const cabSegs = cab ? fullSegs(cab.frames, cab.frame_every, 0, 2) : [];
  const drwSegs = drw ? fullSegs(drw.frames, drw.frame_every, 20, 2.5) : [];
  const cabEnd = c1 + (cab ? segmentsFrames(cab, cabSegs) : 60);
  const drwEnd = c2 + (drw ? segmentsFrames(drw, drwSegs) : 60);
  // the chain: placement and turn sped up so the fall lands on "it gets the job right",
  // the fall itself at half speed, then the settle in real time
  const preFrames = Math.max(30, cCount - c3 + 6);
  const preRate = ((128 - 50) / 20) * 30 / preFrames;
  const chainSegs: Segment[] = [
    {from: 50, to: 128, rate: preRate},
    {from: 128, to: 150, rate: 0.5},
    {from: 150, to: 175, rate: 1},
  ];
  const chainEnd = c3 + (chain ? segmentsFrames(chain, chainSegs) : 120);

  // the dots: the plain chain on the 20 reserved starts, in start order
  const ids = film.mosaics['r1-chain'] ?? [];
  const outcomes = ids.map((id) => (film.clips[id]?.delivered ? '1' : '0'));
  const DOT = 44;
  const DGAP = 16;
  const stagger = 3;
  const fillStart = cCount + 18;
  const fillEnd = fillStart + outcomes.length * stagger;
  const numAt = Math.max(fillEnd + 4, Math.min(cNum, fillEnd + 30));
  const numP = arrive(frame, fps, numAt, 'lively');

  return (
    <Stage>
      <div style={{position: 'absolute', inset: 0}}>
        <Slot i={0} at={c1 - 6} kicker="one skill" pillAt={cabEnd - 4} pill="✓ bowl on the cabinet" tone="ok">
          <Footage clip="single-cabinet-hero" segments={cabSegs} startAt={c1} width={SW} height={SH} />
        </Slot>
        <Slot i={1} at={c2 - 6} kicker="another skill" pillAt={drwEnd - 4} pill="✓ drawer open" tone="ok">
          <Footage clip="single-drawer-hero" segments={drwSegs} startAt={c2} width={SW} height={SH} />
        </Slot>
        <Slot i={2} at={c3 - 6} kicker="the job: both skills" pillAt={chainEnd - 10} pill="✗ both, one after the other" tone="bad">
          <Footage clip="r1-chain-31-hero" segments={chainSegs} startAt={c3} width={BW} height={BH} />
        </Slot>
      </div>

      {/* the count: 20 recorded runs as dots, then the number */}
      <div style={{position: 'absolute', left: SLOTS[2].x, top: TOP + BH + 104, display: 'flex', alignItems: 'center', gap: 64}}>
        <div style={{display: 'grid', gridTemplateColumns: `repeat(10, ${DOT}px)`, gap: DGAP}}>
          {outcomes.map((o, i) => {
            const show = span(frame, cCount + i, cCount + i + 8);
            const at = fillStart + i * stagger;
            return (
              <div key={i} style={{opacity: show, transform: `scale(${0.6 + 0.4 * show})`}}>
                <Dot state={frame >= at ? (o as '1' | '0') : 'outline'} size={DOT} appear={at} />
              </div>
            );
          })}
        </div>
        <div style={{opacity: Math.min(1, numP * 1.4), transform: `translateY(${(1 - Math.min(numP, 1)) * 20}px) scale(${0.85 + 0.15 * numP})`, transformOrigin: 'left center', minWidth: 330}}>
          <div style={{fontSize: 112, fontWeight: 830, letterSpacing: '-0.035em', lineHeight: 1, color: color.ink, whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums'}}>
            <span style={{color: color.bad}}>{naive}</span>
            <span style={{color: color.ink3, fontWeight: 700}}> of </span>
            {n}
          </div>
          <div style={{...type.small, fontSize: 28, marginTop: 8, whiteSpace: 'nowrap'}}>fresh tries, job done right</div>
        </div>
      </div>
      {/* a hairline that says "these dots are runs" */}
      <div
        style={{
          position: 'absolute',
          left: SLOTS[2].x,
          top: TOP + BH + 48,
          ...type.kicker,
          fontSize: 24,
          opacity: span(frame, cCount, cCount + durations.enter),
        }}
      >
        both skills chained · one dot per recorded try
      </div>
    </Stage>
  );
};
