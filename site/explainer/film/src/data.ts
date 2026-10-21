// The film's data contract. Every number and quotation on screen comes from
// public/data/film-data.json, which site/explainer/build_data.py writes from the bound
// script (derived.json, bindings.json) and the recorded runs; never type a number into a
// scene. Timing comes from public/data/timeline.json (the narration's measured durations
// and cue times; estimated at 150 words a minute until the voice is recorded).
import React, {createContext, useContext} from 'react';
import {FPS} from './theme';

export type Proportion = {k: number; n: number; pct: string};

export type Program = {
  job: string; // the run's job id, e.g. j0006
  label: string; // plain words for the screen, e.g. "let the placement finish, then home"
  name: string; // the agent's own name for the program
  outcomes: string; // one character per practice start, "1" delivered, "0" not, in start order
  starts: number[]; // the practice start of each character
  k: number;
  n: number;
  role?: 'plain' | 'winner' | 'loaded' | 'other' | 'confirm';
};

export type TrackFrame = {
  step: number;
  objects: Record<string, {x: number; y: number; depth: number; visible: boolean}>;
  gripper?: {x: number; y: number; depth: number; visible: boolean};
  touch?: Record<string, boolean>; // object -> in contact with the robot (recorded)
  goals?: boolean[]; // requested goals holding (recorded)
  gripper_opening?: number;
  program_step?: number | null;
  instruction?: string | null;
};

export type Clip = {
  id: string;
  src: string; // path under public/, for staticFile()
  fps: number; // the clip's frame rate (20: one frame per control step)
  width: number;
  height: number;
  frames: number;
  first_step: number; // control step of the clip's first frame
  frame_every: number;
  track?: string | null; // overlay track under public/, if any
  run: string; // public id of the recorded run
  episode: string;
  verified: boolean; // hash-verified re-render (or an original recording)
  label: string; // what the clip shows, for the provenance list
  delivered?: boolean;
  start?: number; // the start (arrangement) index
};

export type FilmData = {
  format: string;
  generated: string;
  numbers: Record<string, string>; // every bound placeholder -> its value as shown
  quotes: Record<string, string>; // verbatim text: the agent's notes, requests, lessons
  programs: Record<string, Program[]>; // r1, r2: every program the agent ran on practice starts
  traces: Record<string, any>; // charts drawn from recorded tables
  clips: Record<string, Clip>;
  mosaics: Record<string, string[]>; // clip ids in start order
  fleet: Record<string, any>; // story B: sites, probe, experiments, evaluation, forecast
  provenance: Record<string, any>;
};

export type BeatTiming = {
  id: string;
  seconds: number; // the beat's length, lead-in and tail included
  voice: string | null; // narration audio under public/, if recorded
  voice_offset: number; // seconds from the beat's start to the voice's start
  voice_seconds: number; // the narration's own length
  text: string; // the bound narration (for captions)
  cues: Record<string, number>; // named moments, seconds from the beat's start
};

export type Timeline = {
  format: string;
  source: 'estimate' | 'voice';
  beats: BeatTiming[];
  music?: string | null;
};

export const FilmContext = createContext<FilmData | null>(null);

export const useFilm = (): FilmData => {
  const film = useContext(FilmContext);
  if (!film) throw new Error('useFilm outside a FilmContext');
  return film;
};

/** A bound number or text by its placeholder name; shows {name} loudly if unbound. */
export const useNumber = (name: string): string => {
  const film = useFilm();
  return film.numbers[name] ?? `{${name}}`;
};

export const useQuote = (name: string): string => {
  const film = useFilm();
  return film.quotes[name] ?? `{${name}}`;
};

export type BeatInfo = BeatTiming & {frames: number};

export const BeatContext = createContext<BeatInfo | null>(null);

export const useBeat = (): BeatInfo => {
  const beat = useContext(BeatContext);
  if (!beat) throw new Error('useBeat outside a BeatContext');
  return beat;
};

/**
 * The frame (within the current beat) of a named cue: the moment the narration reaches a
 * phrase listed in site/explainer/cues.json. Falls back to a fraction of the beat when the
 * cue is missing, so a scene still plays before the voice exists.
 */
export const useCue = (name: string, fallbackFraction = 0): number => {
  const beat = useBeat();
  const s = beat.cues[name];
  if (s === undefined) return Math.round(beat.frames * fallbackFraction);
  return Math.round(s * FPS);
};

export const secondsToFrames = (s: number) => Math.round(s * FPS);

export const FilmProvider: React.FC<{film: FilmData; children: React.ReactNode}> = ({film, children}) =>
  React.createElement(FilmContext.Provider, {value: film}, children);
