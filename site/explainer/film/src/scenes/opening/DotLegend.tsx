// The dot legend, introduced in `rules` and carried, small, by every later dot display:
// "● one dot = one try: green delivered, red not". Copy or import it.
//
//   <DotLegend />            the full legend (rules beat)
//   <DotLegend small />      the compact version beside a dot display
//   <DotLegend small wording="attempts" />   for the fleet, whose dots are attempts placed
import React from 'react';
import {color} from '../../theme';

const D: React.FC<{c: string; s: number}> = ({c, s}) => <span style={{display: 'inline-block', width: s, height: s, borderRadius: '50%', background: c, flex: 'none'}} />;

const WORDING = {
  tries: {unit: 'one dot = one try:', ok: 'green delivered,'},
  attempts: {unit: 'one dot = one attempt:', ok: 'green succeeded,'},
} as const;

export const DotLegend: React.FC<{small?: boolean; style?: React.CSSProperties; dark?: boolean; wording?: keyof typeof WORDING}> = ({small, style, dark, wording = 'tries'}) => {
  const w = WORDING[wording];
  const f = small ? 24 : 34;
  const s = small ? 18 : 28;
  const ink = dark ? '#fff' : color.ink;
  const ink2 = dark ? color.nightText : color.ink2;
  return (
    <div style={{display: 'inline-flex', alignItems: 'center', gap: f * 0.45, fontSize: f, fontWeight: 600, color: ink2, whiteSpace: 'nowrap', ...style}}>
      <D c={ink} s={s} />
      <span style={{color: ink, fontWeight: 720}}>{w.unit}</span>
      <D c={color.ok} s={s} />
      <span>{w.ok}</span>
      <D c={color.bad} s={s} />
      <span>red not</span>
    </div>
  );
};
