// A stand-in for a beat whose scene is not built yet: its title and narration, so the
// whole film plays as an animatic from the first day.
import React from 'react';
import {Stage, Kicker, Headline, Appear} from '../components/basics';
import {useBeat} from '../data';
import {color, space} from '../theme';

export const Placeholder: React.FC = () => {
  const beat = useBeat();
  return (
    <Stage act={`beat · ${beat.id}`}>
      <div style={{position: 'absolute', left: space.marginX, right: space.marginX, top: 300}}>
        <Appear>
          <Kicker style={{marginBottom: 24}}>scene in progress</Kicker>
        </Appear>
        <Appear delay={6}>
          <Headline size="h2" style={{maxWidth: 1500, color: color.ink2, fontWeight: 600, fontSize: 52, lineHeight: 1.3}}>
            {beat.text || '(no narration)'}
          </Headline>
        </Appear>
      </div>
    </Stage>
  );
};
