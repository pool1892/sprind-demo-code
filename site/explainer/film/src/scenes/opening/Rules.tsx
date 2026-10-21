// rules: a fair test. A flick through stored practice layouts (recorded first frames, no
// rings); the practice tiles; the final exam's 20 layouts sliding into a
// locked vault; the plain chain's two skill tiles clipping together beside it as the
// baseline; then the dot legend for the film. On the last sentence the baseline steps back
// and two small tags say how the split carries over to real robots:
// practice on simulated copies of the real station, the exam in the real world.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {Footage} from '../../components/footage';
import {useCue, useFilm, useNumber} from '../../data';
import {color, shadow, type} from '../../theme';
import {DotLegend} from './DotLegend';
import {arrive, mix, Padlock, span} from './kit';
import {JobTiles} from './parts';

// practice layouts with the same camera (job 2's development and session 5 runs)
const PRACTICE = ['r2-chain-dev1-hero', 'r2-chain-dev5-hero', 'r2-chain-dev7-hero', 'r2-relocate-0-hero'];

const BIG_W = 1120;
const BIG_H = 630;
const BIG_X = (1920 - BIG_W) / 2;
const BIG_Y = 190;

const VX = 1020; // vault
const VY = 150;
const VW = 780;
const TILE = 120;
const TGAP = 12;

/** A small tag: how this part of the split carries over to real robots. */
const RealTag: React.FC<{x: number; y: number; show: number; text: string}> = ({x, y, show, text}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      opacity: Math.min(1, show * 1.4),
      transform: `translateY(${(1 - Math.min(1, show)) * 16}px)`,
      display: 'inline-flex',
      alignItems: 'baseline',
      gap: 12,
      padding: '12px 26px',
      borderRadius: 999,
      background: color.card,
      boxShadow: shadow.pill,
      borderLeft: `8px solid ${color.teal}`,
      whiteSpace: 'nowrap',
    }}
  >
    <span style={{fontSize: 24, fontWeight: 760, letterSpacing: '0.08em', textTransform: 'uppercase', color: color.teal}}>on real robots:</span>
    <span style={{fontSize: 30, fontWeight: 700, color: color.ink}}>{text}</span>
  </div>
);

export const Rules: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const film = useFilm();
  const n = useNumber('r1_release_n');
  const cLayout = useCue('layout', 0.02);
  const cPractice = useCue('practice', 0.23);
  const cVault = useCue('vault', 0.43);
  const cExam = useCue('exam', 0.49);
  const cBaseline = useCue('baseline', 0.66);
  const cSkills = useCue('skills', 0.74);
  const cLegend = useCue('baseline2', 0.85);
  const cReal = useCue('real', 0.7);
  const cSimCopy = useCue('simcopy', 0.8);
  const cRealWorld = useCue('realworld', 0.92);
  const tiles = film.mosaics['r1-chain'] ?? [];

  // phase 1: the flick
  const every = Math.max(18, Math.floor((cPractice - cLayout - 10) / PRACTICE.length));
  const idx = Math.max(0, Math.min(PRACTICE.length - 1, Math.floor((frame - cLayout) / every)));
  const into = frame - cLayout - idx * every; // frames into the current layout
  const big = arrive(frame, fps, cLayout - 8) * (1 - span(frame, cPractice - 8, cPractice + 6));

  const practice = arrive(frame, fps, cPractice);
  const vault = arrive(frame, fps, cVault - 12);
  const lockAt = Math.max(cExam + 10, cVault + tiles.length * 2 + 16);
  const locked = span(frame, lockAt, lockAt + 10);
  const lockDrop = arrive(frame, fps, lockAt - 8, 'lively');
  const base = arrive(frame, fps, cBaseline - 6);
  const arrow = span(frame, cBaseline + 30, cBaseline + 50);
  const legend = arrive(frame, fps, cLegend);
  const baseOut = 1 - span(frame, cReal - 4, cReal + 10);
  const tagSim = arrive(frame, fps, cSimCopy);
  const tagReal = arrive(frame, fps, cRealWorld);

  return (
    <Stage act="a fair test">
      {/* phase 1: stored layouts, one after another */}
      {big > 0.01 ? (
        <div style={{position: 'absolute', inset: 0, opacity: big}}>
          <div style={{position: 'absolute', left: 0, right: 0, top: 108, textAlign: 'center'}}>
            <span style={{...type.h3, fontSize: 44}}>a layout: </span>
            <span style={{...type.body, fontSize: 40, color: color.ink2}}>the table's stored starting arrangement</span>
          </div>
          <div style={{position: 'absolute', left: BIG_X, top: BIG_Y, transform: `scale(${mix(1 - big, 1, 0.92)})`}}>
            <Footage key={PRACTICE[idx]} clip={PRACTICE[idx]} segments={[{from: 0, to: 0}]} width={BIG_W} height={BIG_H} />
            <div style={{position: 'absolute', left: 24, top: 22, padding: '10px 22px', borderRadius: 999, background: color.night, color: '#fff', fontSize: 28, fontWeight: 720, boxShadow: shadow.pill, opacity: into < 4 && idx > 0 ? 0.6 : 1}}>practice layout</div>
          </div>
        </div>
      ) : null}

      {/* practice layouts, open to the agent */}
      <div style={{position: 'absolute', left: 110, top: VY, opacity: practice, transform: `translateY(${(1 - practice) * 24}px)`}}>
        <div style={{...type.kicker, marginBottom: 18}}>practice layouts · open</div>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(2, 340px)', gap: 16}}>
          {PRACTICE.map((id, i) => {
            const p = span(frame, cPractice + i * 3, cPractice + 10 + i * 3);
            return (
              <div key={id} style={{opacity: p, transform: `scale(${0.85 + 0.15 * p})`}}>
                <Footage clip={id} segments={[{from: 0, to: 0}]} width={340} height={191} radius={16} style={{boxShadow: shadow.card}} />
              </div>
            );
          })}
        </div>
        <div style={{...type.small, fontSize: 28, marginTop: 14}}>the agent compares its programs here</div>
      </div>

      {/* the final exam: 20 layouts, locked away */}
      <div
        style={{
          position: 'absolute',
          left: VX,
          top: VY,
          width: VW,
          boxSizing: 'border-box',
          background: color.night,
          borderRadius: 28,
          boxShadow: shadow.frame,
          padding: '28px 36px 30px',
          opacity: vault,
          transform: `translateY(${(1 - vault) * 30}px)`,
        }}
      >
        <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
          <div style={{transform: `translateY(${(1 - Math.min(1, lockDrop)) * -40}px)`, opacity: Math.min(1, lockDrop * 1.5)}}>
            <Padlock size={40} tone={color.lead} closed={locked} />
          </div>
          <div style={{fontSize: 30, fontWeight: 760, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#fff'}}>final exam · {n} layouts</div>
        </div>
        <div style={{position: 'relative', marginTop: 22, display: 'grid', gridTemplateColumns: `repeat(5, ${TILE}px)`, gap: TGAP}}>
          {tiles.map((id, i) => {
            const p = arrive(frame, fps, cVault + i * 2);
            return (
              <div key={id} style={{position: 'relative', width: TILE, height: TILE}}>
                <div style={{position: 'absolute', inset: 0, borderRadius: 14, border: '3px dashed #3A3E4A', boxSizing: 'border-box'}} />
                <div style={{position: 'absolute', inset: 0, opacity: Math.min(1, p * 1.4), transform: `translateX(${(1 - p) * -260}px)`}}>
                  <Footage clip={id} segments={[{from: 0, to: 0}]} width={TILE} height={TILE} radius={14} style={{boxShadow: 'none'}} />
                </div>
              </div>
            );
          })}
          <div style={{position: 'absolute', inset: -6, borderRadius: 12, background: color.night, opacity: locked * 0.45}} />
        </div>
        <div style={{fontSize: 30, fontWeight: 560, color: color.nightText, marginTop: 24, opacity: locked}}>locked until the agent's solution is final</div>
      </div>

      {/* the baseline: the plain chain, on the same layouts; it steps back on "Later, on real robots" */}
      <div style={{position: 'absolute', inset: 0, opacity: baseOut}}>
      <div style={{position: 'absolute', left: 110, top: 666, ...type.kicker, opacity: base}}>the baseline</div>
      <JobTiles x={110} y={712} at={cSkills - 18} crack={0} w={340} h={100} fontSize={32} />
      <div style={{position: 'absolute', left: 110, top: 828, ...type.body, fontSize: 30, opacity: arrive(frame, fps, cSkills + 10)}}>the plain chain: the skills one after the other</div>
      <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', opacity: arrow}}>
        <path d={`M${110 + 680 + 30},${762} C${930},${762} ${960},${700} ${VX - 14},${660}`} fill="none" stroke={color.ink3} strokeWidth={5} strokeDasharray="10 10" />
        <path d={`M${VX - 34},${652} L${VX - 12},${660} L${VX - 30},${676}`} fill="none" stroke={color.ink3} strokeWidth={5} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      </div>

      {/* later, on real robots: two small tags */}
      <RealTag x={110} y={700} show={tagSim} text="simulated copies of the real station" />
      <RealTag x={VX} y={872} show={tagReal} text="the real world" />

      {/* the legend that stays for the film */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 952, display: 'flex', justifyContent: 'center', opacity: Math.min(1, legend * 1.3), transform: `translateY(${(1 - Math.min(1, legend)) * 20}px)`}}>
        <div style={{padding: '16px 34px', borderRadius: 999, background: color.card, boxShadow: shadow.card}}>
          <DotLegend />
        </div>
      </div>
    </Stage>
  );
};
