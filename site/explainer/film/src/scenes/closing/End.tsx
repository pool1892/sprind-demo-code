// end: how this was recorded (silent). The one line on the simulator and sample size, the
// address of the record and, under it in the same style, the code repository.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Stage, Appear} from '../../components/basics';
import {useNumber} from '../../data';
import {color, font} from '../../theme';
import {ramp} from './parts';

const ADDRESSES = ['pool1892.github.io/sprind-demo-page', 'github.com/pool1892/sprind-demo-code'];

export const End: React.FC = () => {
  const frame = useCurrentFrame();
  const agent = useNumber('r1_agent_model');
  const fewest = useNumber('r1_plain_n');
  const most = useNumber('b_eval_final_n');
  const rows: [string, React.ReactNode][] = [
    ['Simulation', 'LIBERO household tasks, MuJoCo physics'],
    ['Robot model', 'X-VLA, weights unchanged in every run'],
    ['Engineering agent', <span style={{fontFamily: font.mono, color: '#fff'}}>{agent}</span>],
    [
      'Evidence',
      <>
        Every clip is a recorded run. Every count is literal, from {fewest} to {most} runs.
      </>,
    ],
  ];
  return (
    <Stage dark>
      <div style={{position: 'absolute', left: 200, right: 200, top: 190}}>
        <Appear delay={4}>
          <div style={{fontSize: 30, fontWeight: 750, letterSpacing: '0.14em', textTransform: 'uppercase', color: '#8A90A0'}}>How this was recorded</div>
        </Appear>
        <div style={{marginTop: 50, display: 'grid', gridTemplateColumns: '400px 1fr', rowGap: 34, columnGap: 40}}>
          {rows.map(([k, v], i) => (
            <React.Fragment key={k}>
              <div style={{opacity: ramp(frame, 12 + i * 8, 26 + i * 8), fontSize: 36, fontWeight: 700, color: '#8A90A0'}}>{k}</div>
              <div style={{opacity: ramp(frame, 14 + i * 8, 28 + i * 8), fontSize: 38, fontWeight: 560, color: color.nightText, lineHeight: 1.3}}>{v}</div>
            </React.Fragment>
          ))}
        </div>
        <div style={{marginTop: 90, display: 'flex', flexDirection: 'column', gap: 26}}>
          {ADDRESSES.map((a, i) => (
            <div key={a} style={{opacity: ramp(frame, 56 + i * 12, 72 + i * 12), display: 'flex', alignItems: 'center', gap: 22}}>
              <div style={{width: 16, height: 16, borderRadius: '50%', background: color.agent}} />
              <div style={{fontFamily: font.mono, fontSize: 40, color: '#fff'}}>{a}</div>
            </div>
          ))}
        </div>
      </div>
    </Stage>
  );
};
