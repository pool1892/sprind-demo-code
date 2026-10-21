// The beats of the film, in order, and the scene that draws each. Beat ids match the
// sections of site/explainer/script.md. A beat without a built scene plays its placeholder.
import React from 'react';
import {Placeholder} from './scenes/Placeholder';
import {SCENES} from './scenes';

export const BEAT_ORDER = [
  'cold-open',
  'title',
  'why',
  'question',
  'station',
  'agent',
  'rules',
  'request',
  'r1-fail',
  'r1-measure',
  'r1-experiment',
  'r1-prove',
  'r1-keep',
  'r2-fail',
  'r2-bottleneck',
  'r2-fix',
  'r-summary',
  'b-fleet',
  'b-planted',
  'b-symptom',
  'b-probe',
  'b-fix',
  'b-result',
  'results',
  'lessons',
  'next',
  'mayfly',
  'end',
] as const;

export type BeatId = (typeof BEAT_ORDER)[number];

/** Beats with no narration and a fixed length, in seconds. */
export const SILENT_BEATS: Partial<Record<BeatId, number>> = {title: 4, end: 12};

/** How a beat enters: a soft crossfade by default, a hard cut where the picture should snap. */
export const ENTRY: Partial<Record<BeatId, 'cut' | 'fade' | 'slide'>> = {
  title: 'cut',
  request: 'slide',
  'r2-fail': 'slide',
  'b-fleet': 'slide',
  results: 'slide',
};

export const sceneFor = (id: string): React.FC => SCENES[id] ?? Placeholder;
