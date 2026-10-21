// The exam tiles' legend (one tile = one try), in the style of the dot legend introduced in
// `rules`; the dot legend itself is the opening lane's shared one, re-exported here.
import React from 'react';
import {color} from '../../theme';

export {DotLegend} from '../opening/DotLegend';

/** The same legend for the exam tiles: one tile is one try, its border green or red. */
export const TileLegend: React.FC<{style?: React.CSSProperties}> = ({style}) => {
  const f = 24;
  const t = (c: string) => (
    <span style={{display: 'inline-block', width: 22, height: 22, borderRadius: 5, boxShadow: `inset 0 0 0 4px ${c}`, flex: 'none'}} />
  );
  return (
    <div style={{display: 'inline-flex', alignItems: 'center', gap: f * 0.45, fontSize: f, fontWeight: 600, color: color.ink2, whiteSpace: 'nowrap', ...style}}>
      {t(color.ink)}
      <span style={{color: color.ink, fontWeight: 720}}>one tile = one try:</span>
      {t(color.ok)}
      <span>green delivered,</span>
      {t(color.bad)}
      <span>red not</span>
    </div>
  );
};
