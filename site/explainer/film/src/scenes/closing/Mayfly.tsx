// mayfly: zoom out from the packing cell (an illustration); request cards flow through the
// loop onto a growing library shelf; then the Mayfly wordmark and the line.
import React from 'react';
import {interpolate, useCurrentFrame, Easing} from 'remotion';
import {Stage} from '../../components/basics';
import {Loop} from '../../components/agent';
import {useCue} from '../../data';
import {color, shadow} from '../../theme';
import {PackingCell, ramp} from './parts';

const REQUESTS = ['new product', 'new box size', 'returns'];
const LOOP = {x: 830, y: 360, size: 260};
const SHELF = {x: 1300, top: 380, w: 480, row: 92};

export const Mayfly: React.FC = () => {
  const frame = useCurrentFrame();
  const tInstall = useCue('installations', 0.22);
  const tAgents = useCue('agents', 0.57);
  const z = ramp(frame, 4, 54, Easing.inOut(Easing.cubic));
  const w = interpolate(z, [0, 1], [1400, 560]);
  const x = interpolate(z, [0, 1], [260, 120]);
  const y = interpolate(z, [0, 1], [230, 380]);
  const outro = ramp(frame, tAgents - 8, tAgents + 6);
  const flowStart = Math.min(tInstall, 60);
  const word = ramp(frame, tAgents, tAgents + 16);
  const line = ramp(frame, tAgents + 12, tAgents + 28);

  return (
    <Stage>
      <div style={{position: 'absolute', inset: 0, opacity: 1 - outro}}>
        <div style={{position: 'absolute', left: x, top: y}}>
          <PackingCell t={frame} width={w} />
        </div>
        <div style={{position: 'absolute', left: LOOP.x, top: LOOP.y, opacity: ramp(frame, flowStart - 10, flowStart + 6)}}>
          <Loop size={LOOP.size} labels={false} start={flowStart - 10} />
        </div>
        <div style={{position: 'absolute', left: SHELF.x, top: SHELF.top + 3 * SHELF.row + 4, width: SHELF.w, height: 18, borderRadius: 6, background: color.wood, opacity: ramp(frame, flowStart, flowStart + 12)}} />
        <div style={{position: 'absolute', left: SHELF.x, top: SHELF.top + 3 * SHELF.row + 36, fontSize: 28, fontWeight: 700, color: color.ink2, opacity: ramp(frame, flowStart + 50, flowStart + 64)}}>kept for the next one</div>
        {REQUESTS.map((label, i) => {
          const s = flowStart + i * 22;
          const a = ramp(frame, s - 6, s + 4);
          const toLoop = ramp(frame, s, s + 18, Easing.inOut(Easing.cubic));
          const toShelf = ramp(frame, s + 26, s + 44, Easing.inOut(Easing.cubic));
          const from = {x: 160, y: 760};
          const mid = {x: LOOP.x + LOOP.size / 2 - 150, y: LOOP.y + LOOP.size / 2 - 38};
          const slot = {x: SHELF.x, y: SHELF.top + (2 - i) * SHELF.row};
          const px = toShelf > 0 ? interpolate(toShelf, [0, 1], [mid.x, slot.x]) : interpolate(toLoop, [0, 1], [from.x, mid.x]);
          const py = toShelf > 0 ? interpolate(toShelf, [0, 1], [mid.y, slot.y]) : interpolate(toLoop, [0, 1], [from.y, mid.y]);
          const kept = toShelf > 0.5;
          const inside = toLoop >= 1 && toShelf <= 0;
          return (
            <div
              key={label}
              style={{
                position: 'absolute',
                left: px,
                top: py,
                width: kept ? SHELF.w : 300,
                height: 76,
                boxSizing: 'border-box',
                borderRadius: 18,
                background: kept ? color.card : color.lead,
                color: kept ? color.ink : '#fff',
                borderLeft: kept ? `14px solid ${color.ok}` : 'none',
                boxShadow: shadow.card,
                display: 'flex',
                alignItems: 'center',
                gap: 14,
                padding: '0 24px',
                fontSize: 32,
                fontWeight: 760,
                whiteSpace: 'nowrap',
                opacity: a * (inside ? 0 : 1),
                transform: `scale(${toShelf > 0 ? interpolate(toShelf, [0, 1], [0.5, 1]) : interpolate(toLoop, [0, 1], [1, 0.5])})`,
              }}
            >
              {kept ? <span style={{color: color.ok}}>✓</span> : null}
              {label}
              {kept ? <span style={{fontSize: 24, fontWeight: 650, color: color.ink3, marginLeft: 'auto'}}>job + check</span> : null}
            </div>
          );
        })}
      </div>

      <div style={{position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: word}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 30, transform: `translateY(${(1 - word) * 24}px)`}}>
          <div style={{width: 34, height: 34, borderRadius: '50%', background: color.agent, boxShadow: `0 0 0 10px rgba(75,92,240,0.15)`}} />
          <div style={{fontSize: 168, fontWeight: 850, letterSpacing: '-0.045em', color: color.ink, lineHeight: 1}}>Mayfly</div>
        </div>
        <div style={{fontSize: 54, fontWeight: 720, color: color.ink2, marginTop: 48, letterSpacing: '-0.015em', opacity: line, textAlign: 'center', maxWidth: 1500}}>
          Taking on new work is part of the product.
        </div>
      </div>
    </Stage>
  );
};
