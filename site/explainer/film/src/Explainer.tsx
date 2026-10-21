// The whole film: the beats in order, each with its narration, joined by short transitions,
// over a music bed that ducks under the voice.
import React, {useState} from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile, continueRender, delayRender, interpolate} from 'remotion';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import {slide} from '@remotion/transitions/slide';
import {BEAT_ORDER, ENTRY, SILENT_BEATS, sceneFor} from './beats';
import {BeatContext, FilmProvider, type BeatInfo, type FilmData, type Timeline} from './data';
import {color, durations, FPS} from './theme';

export type ExplainerProps = {film: FilmData | null; timeline: Timeline | null};

/** The beats with their frame counts, in film order. */
export const beatsOf = (timeline: Timeline): BeatInfo[] => {
  const byId = new Map(timeline.beats.map((b) => [b.id, b]));
  return BEAT_ORDER.map((id) => {
    const b = byId.get(id);
    const seconds = b?.seconds ?? SILENT_BEATS[id] ?? 6;
    return {
      id,
      seconds,
      voice: b?.voice ?? null,
      voice_offset: b?.voice_offset ?? 0,
      voice_seconds: b?.voice_seconds ?? 0,
      text: b?.text ?? '',
      cues: b?.cues ?? {},
      frames: Math.round(seconds * FPS),
    };
  });
};

const T = durations.transition;
const hasTransition = (i: number, id: string) => i > 0 && (ENTRY[id as keyof typeof ENTRY] ?? 'fade') !== 'cut';

/** Total frames: every beat, less the overlap of each transition. */
export const totalFrames = (beats: BeatInfo[]): number =>
  beats.reduce((sum, b, i) => sum + b.frames - (hasTransition(i, b.id) ? T : 0), 0);

/** Absolute start frame of each beat. */
export const beatStarts = (beats: BeatInfo[]): number[] => {
  const starts: number[] = [];
  let at = 0;
  beats.forEach((b, i) => {
    if (hasTransition(i, b.id)) at -= T;
    starts.push(at);
    at += b.frames;
  });
  return starts;
};

const useFontsReady = () => {
  const [handle] = useState(() => delayRender('fonts'));
  useState(() => {
    if (typeof document !== 'undefined' && document.fonts) {
      document.fonts.ready.then(() => continueRender(handle));
    } else continueRender(handle);
    return null;
  });
};

export const Explainer: React.FC<ExplainerProps> = ({film, timeline}) => {
  useFontsReady();
  if (!film || !timeline) return <AbsoluteFill style={{background: color.paper}} />;
  const beats = beatsOf(timeline);
  const starts = beatStarts(beats);
  // the voice's spans in absolute frames, for ducking the music
  const voiced = beats
    .map((b, i) => (b.voice ? [starts[i] + Math.round(b.voice_offset * FPS), starts[i] + Math.round((b.voice_offset + b.voice_seconds) * FPS)] : null))
    .filter((x): x is number[] => x !== null);
  const musicVolume = (f: number) => {
    let duck = 0;
    for (const [a, b] of voiced) duck = Math.max(duck, interpolate(f, [a - 12, a, b, b + 18], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
    const fadeIn = interpolate(f, [0, 20], [0, 1], {extrapolateRight: 'clamp'});
    const end = totalFrames(beats);
    const fadeOut = interpolate(f, [end - 60, end], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
    return (0.2 - 0.14 * duck) * fadeIn * fadeOut; // about 13 dB under the voice while it speaks
  };
  return (
    <FilmProvider film={film}>
      <AbsoluteFill style={{background: color.paper}}>
        <TransitionSeries>
          {beats.flatMap((b, i) => {
            const Scene = sceneFor(b.id);
            const entry = ENTRY[b.id as keyof typeof ENTRY] ?? 'fade';
            const items: React.ReactNode[] = [];
            if (hasTransition(i, b.id)) {
              items.push(
                <TransitionSeries.Transition
                  key={`t-${b.id}`}
                  presentation={entry === 'slide' ? slide({direction: 'from-right'}) : fade()}
                  timing={linearTiming({durationInFrames: T})}
                />,
              );
            }
            items.push(
              <TransitionSeries.Sequence key={b.id} durationInFrames={b.frames} name={b.id}>
                <BeatContext.Provider value={b}>
                  <Scene />
                  {b.voice ? (
                    <Sequence from={Math.round(b.voice_offset * FPS)} name={`voice ${b.id}`}>
                      <Audio src={staticFile(b.voice)} />
                    </Sequence>
                  ) : null}
                </BeatContext.Provider>
              </TransitionSeries.Sequence>,
            );
            return items;
          })}
        </TransitionSeries>
        {timeline.music ? <Audio src={staticFile(timeline.music)} volume={musicVolume} /> : null}
      </AbsoluteFill>
    </FilmProvider>
  );
};

/** One beat on its own, for building and checking a scene. */
export const BeatPreview: React.FC<ExplainerProps & {beat: string}> = ({film, timeline, beat}) => {
  useFontsReady();
  if (!film || !timeline) return <AbsoluteFill style={{background: color.paper}} />;
  const b = beatsOf(timeline).find((x) => x.id === beat);
  if (!b) return <AbsoluteFill style={{background: color.paper}} />;
  const Scene = sceneFor(b.id);
  return (
    <FilmProvider film={film}>
      <BeatContext.Provider value={b}>
        <Scene />
        {b.voice ? (
          <Sequence from={Math.round(b.voice_offset * FPS)}>
            <Audio src={staticFile(b.voice)} />
          </Sequence>
        ) : null}
      </BeatContext.Provider>
    </FilmProvider>
  );
};
