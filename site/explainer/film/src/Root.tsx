import React from 'react';
import {Composition, staticFile} from 'remotion';
import {Explainer, BeatPreview, beatsOf, totalFrames, type ExplainerProps} from './Explainer';
import {FPS, H, W} from './theme';
import type {FilmData, Timeline} from './data';

const load = async (): Promise<{film: FilmData; timeline: Timeline}> => {
  const [film, timeline] = await Promise.all([
    fetch(staticFile('data/film-data.json')).then((r) => r.json()),
    fetch(staticFile('data/timeline.json')).then((r) => r.json()),
  ]);
  return {film, timeline};
};

export const Root: React.FC = () => (
  <>
    <Composition
      id="Explainer"
      component={Explainer}
      fps={FPS}
      width={W}
      height={H}
      durationInFrames={FPS * 10}
      defaultProps={{film: null, timeline: null} as ExplainerProps}
      calculateMetadata={async () => {
        const {film, timeline} = await load();
        return {durationInFrames: Math.max(1, totalFrames(beatsOf(timeline))), props: {film, timeline}};
      }}
    />
    <Composition
      id="Beat"
      component={BeatPreview}
      fps={FPS}
      width={W}
      height={H}
      durationInFrames={FPS * 10}
      defaultProps={{film: null, timeline: null, beat: 'cold-open'} as ExplainerProps & {beat: string}}
      calculateMetadata={async ({props}) => {
        const {film, timeline} = await load();
        const b = beatsOf(timeline).find((x) => x.id === props.beat);
        return {durationInFrames: Math.max(1, b?.frames ?? FPS * 5), props: {...props, film, timeline}};
      }}
    />
  </>
);
