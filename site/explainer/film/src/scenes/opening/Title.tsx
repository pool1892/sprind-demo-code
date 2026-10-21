// title: "Can an AI agent engineer a robot?", silent over the music hit. The indigo
// glyph of the AI agent arrives beside it.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {AgentGlyph} from '../../components/agent';
import {color, space, type} from '../../theme';
import {arrive} from './kit';

export const Wordmark: React.FC<{size?: number; style?: React.CSSProperties}> = ({size = 44, style}) => (
  <div style={{display: 'flex', alignItems: 'center', gap: size * 0.32, ...style}}>
    <svg width={size * 0.9} height={size * 0.9} viewBox="0 0 40 40">
      <circle cx={13} cy={20} r={9} fill={color.agent} />
      <circle cx={29} cy={13} r={6} fill={color.ok} />
      <circle cx={29} cy={29} r={6} fill={color.lead} />
    </svg>
    <div style={{fontSize: size, fontWeight: 820, letterSpacing: '-0.03em', color: color.ink}}>Mayfly</div>
  </div>
);

export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const head = arrive(frame, fps, 3);
  const sub = arrive(frame, fps, 14);
  const mark = arrive(frame, fps, 0);
  const glyph = arrive(frame, fps, 20, 'lively');
  return (
    <Stage>
      <div style={{position: 'absolute', inset: 0}}>
        <div style={{position: 'absolute', left: space.marginX + 40, top: 120, opacity: mark, transform: `translateY(${(1 - mark) * 14}px)`}}>
          <Wordmark size={46} />
        </div>
        <div style={{position: 'absolute', left: space.marginX + 40, top: 360}}>
          <div style={{...type.display, fontSize: 128, opacity: head, transform: `translateY(${(1 - head) * 36}px)`}}>
            Can an AI agent
            <br />
            engineer a <span style={{color: color.agent}}>robot?</span>
          </div>
          <div style={{...type.body, fontSize: 44, marginTop: 44, color: color.ink2, opacity: sub, transform: `translateY(${(1 - sub) * 20}px)`}}>
            Two experiments in simulation
          </div>
        </div>
        <div style={{position: 'absolute', left: 1450, top: 380, opacity: glyph, transform: `scale(${0.7 + 0.3 * Math.min(glyph, 1.08)})`}}>
          <AgentGlyph size={250} />
        </div>
      </div>
    </Stage>
  );
};
