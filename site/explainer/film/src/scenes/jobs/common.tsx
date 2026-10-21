// Building blocks shared by the job beats (B3): the stage with the loop badge, keyframed
// footage timing, verdict glyphs, captions and the exam mosaic with verdict badges.
import React, {useEffect, useState} from 'react';
import {continueRender, delayRender, Easing, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {ExperimentTag} from '../opening/ExperimentCard';
import {ENGINEERING_STEPS} from '../opening/Why';
import {Footage, type Segment} from '../../components/footage';
import {useFilm, type Clip, type TrackFrame} from '../../data';
import {color, font, FPS, shadow, spring as springs} from '../../theme';

export type ProgramEvent = {program_step: number | null; kind: string; start_step: number; end_step: number; outcome: string; instruction: string | null};
export type TrackFile = {frames: TrackFrame[]; program: ProgramEvent[]};

/**
 * A clip's whole overlay track: its recorded frames and its program (each step's kind, start
 * and end control step, instruction). Null while loading or for a clip without a track.
 */
export const useTrackFile = (id: string): TrackFile | null => {
  const film = useFilm();
  const clip = film.clips[id] ?? null;
  const [data, setData] = useState<TrackFile | null>(null);
  const [handle] = useState(() => (clip?.track ? delayRender(`track file ${id}`) : null));
  useEffect(() => {
    if (!clip?.track || handle === null) return;
    fetch(staticFile(clip.track))
      .then((r) => r.json())
      .then((j) => {
        setData({frames: (Array.isArray(j) ? j : j.frames) ?? [], program: j.program ?? []});
        continueRender(handle);
      })
      .catch((e) => {
        console.error('track file', id, e);
        continueRender(handle);
      });
  }, [clip, handle, id]);
  return data;
};

/** The control step each program step starts at, from the track's program (else `fallback`). */
export const programStarts = (track: TrackFile | null, fallback: readonly number[]): number[] => {
  const steps = (track?.program ?? []).filter((e) => e.program_step !== null).map((e) => e.start_step);
  return steps.length ? steps : [...fallback];
};

type RawObject = {x: number; y: number; visible: boolean; cx?: number | null; cy?: number | null; bbox?: number[] | null};
export type ObjBox = {cx: number; cy: number; w: number; h: number; visible: boolean};

const boxOf = (o: RawObject | undefined): ObjBox | null => {
  if (!o) return null;
  if (o.bbox && o.bbox.length === 4) {
    const [x0, y0, x1, y1] = o.bbox;
    return {cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, visible: o.visible};
  }
  return {cx: o.cx ?? o.x, cy: o.cy ?? o.y, w: 0, h: 0, visible: o.visible};
};

/**
 * An object's on-screen box (its projected bounding box, in clip pixels) at a fractional
 * control step, interpolated between the track's recorded frames.
 */
export const objectBox = (track: TrackFile | null, step: number, name: string): ObjBox | null => {
  const fr = track?.frames;
  if (!fr || fr.length === 0) return null;
  const every = fr.length > 1 ? Math.max(1, fr[1].step - fr[0].step) : 1;
  const idx = Math.max(0, Math.min(fr.length - 1, (step - fr[0].step) / every));
  const i0 = Math.floor(idx);
  const i1 = Math.min(fr.length - 1, i0 + 1);
  const f = idx - i0;
  const a = boxOf(fr[i0].objects[name] as RawObject | undefined);
  const b = boxOf(fr[i1].objects[name] as RawObject | undefined) ?? a;
  if (!a || !b) return a;
  const mix = (p: number, q: number) => p + (q - p) * f;
  return {cx: mix(a.cx, b.cx), cy: mix(a.cy, b.cy), w: mix(a.w, b.w), h: mix(a.h, b.h), visible: a.visible};
};

/** Whether the run recorded `name` in contact with the robot at this step (from the track). */
export const touchingAt = (track: TrackFile | null, step: number, name: string): boolean => {
  const fr = track?.frames;
  if (!fr || fr.length === 0) return false;
  const every = fr.length > 1 ? Math.max(1, fr[1].step - fr[0].step) : 1;
  const i = Math.max(0, Math.min(fr.length - 1, Math.round((step - fr[0].step) / every)));
  return Boolean(fr[i].touch?.[name]);
};

/** A label pinned beside a tracked object's box (offsets in display px from its centre). */
export const TrackedPin: React.FC<{
  toBox: (x: number, y: number) => {x: number; y: number};
  box: ObjBox | null;
  show: number;
  dx: number;
  dy: number;
  tone?: 'light' | 'dark' | 'bad' | 'ok';
  children: React.ReactNode;
}> = ({toBox, box, show, dx, dy, tone = 'light', children}) => {
  if (!box || !box.visible || show <= 0) return null;
  const p = toBox(box.cx, box.cy);
  return (
    <div style={{position: 'absolute', left: p.x + dx, top: p.y + dy, opacity: Math.min(1, show)}}>
      <Chip tone={tone} size={30}>
        {children}
      </Chip>
    </div>
  );
};

/** The first recorded step at which `object` touches the robot at or after `from`. */
export const firstContact = (track: TrackFile | null, object: string, from = 0, fallback = 0): number => {
  const f = track?.frames.find((x) => x.step >= from && x.touch?.[object]);
  return f ? f.step : fallback;
};

export const JOB_ACT = {
  1: 'first job · bowl and drawer',
  2: 'second job · plate and bowl',
} as const;

/**
 * Where the beat stands in the engineering the `why` beat introduced (watch the failures,
 * find the cause, change the system, prove it): the four steps as dots, this one lit and named.
 */
export const StepBadge: React.FC<{step: number}> = ({step}) => (
  <div style={{position: 'absolute', right: 70, top: 46, display: 'flex', alignItems: 'center', gap: 16, padding: '10px 24px', borderRadius: 999, background: color.card, boxShadow: shadow.pill, whiteSpace: 'nowrap'}}>
    <div style={{display: 'flex', gap: 8}}>
      {ENGINEERING_STEPS.map((_, i) => (
        <div key={i} style={{width: 14, height: 14, borderRadius: '50%', background: i === step ? color.agent : 'transparent', boxShadow: i === step ? 'none' : `inset 0 0 0 3px ${i < step ? color.ink3 : color.line}`}} />
      ))}
    </div>
    <div style={{fontSize: 24, fontWeight: 760, letterSpacing: '0.08em', textTransform: 'uppercase', color: color.agent}}>{ENGINEERING_STEPS[step] ?? ''}</div>
  </div>
);

/** A job beat: paper stage, Experiment 1's tag and the job top left, the engineering step top right. */
export const JobStage: React.FC<{job: 1 | 2; step: number | null; children: React.ReactNode}> = ({job, step, children}) => (
  <Stage>
    {children}
    <div style={{position: 'absolute', left: 120, top: 46, display: 'flex', alignItems: 'center', gap: 18}}>
      <ExperimentTag which={1} />
      <div style={{fontSize: 24, fontWeight: 650, color: color.ink2, whiteSpace: 'nowrap'}}>{JOB_ACT[job]}</div>
    </div>
    {step === null ? null : <StepBadge step={step} />}
  </Stage>
);

export type Key = [number, number]; // [frame of this scene, control step shown]

/**
 * Playback segments that pass through [frame, step] keyframes. Equal steps hold the picture;
 * the clip never runs backwards. Returns the segments and the frame playback starts.
 */
export const keyed = (clip: Clip | null, keys: Key[]): {segments: Segment[]; startAt: number} => {
  const fps = clip?.fps ?? 20;
  const segments: Segment[] = [];
  for (let i = 1; i < keys.length; i++) {
    const [f0, s0] = keys[i - 1];
    const [f1, s1] = keys[i];
    const frames = Math.max(1, f1 - f0);
    const steps = Math.max(1e-4, s1 - s0);
    segments.push({from: s0, to: s0 + steps, rate: ((steps / fps) * FPS) / frames});
  }
  return {segments, startAt: keys[0]?.[0] ?? 0};
};

/** The control step on screen at `frame` for the same keyframes (to sync graphics to footage). */
export const stepOfKeys = (frame: number, keys: Key[]): number => {
  if (keys.length === 0) return 0;
  if (frame <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [f0, s0] = keys[i - 1];
    const [f1, s1] = keys[i];
    if (frame <= f1) return s0 + ((frame - f0) / Math.max(1, f1 - f0)) * (s1 - s0);
  }
  return keys[keys.length - 1][1];
};

/** Keeps keyframe frames strictly increasing and inside the beat (voice lengths vary). */
export const orderKeys = (keys: Key[], last: number): Key[] => {
  const out: Key[] = [];
  let prev = -Infinity;
  for (const [f, s] of keys) {
    const ff = Math.min(last, Math.max(f, prev + 1));
    out.push([ff, s]);
    prev = ff;
  }
  return out;
};

/** Whether `step` lies in one of the recorded [from, to) intervals, joining gaps of `join` steps. */
export const inIntervals = (step: number, intervals: [number, number][] | undefined, join = 0): boolean =>
  (intervals ?? []).some(([a, b], i, all) => {
    const nextA = all[i + 1]?.[0];
    const end = nextA !== undefined && nextA - b <= join ? nextA : b;
    return step >= a && step < end;
  });

/** 0..1 progress of a calm arrival at `at`. */
export const useArrive = (at: number, config: keyof typeof springs = 'calm'): number => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - at, fps, config: springs[config], durationInFrames: 32});
};

/** 0..1 linear-eased ramp between two frames. */
export const ramp = (frame: number, a: number, b: number, ease = Easing.inOut(Easing.cubic)): number =>
  interpolate(frame, [a, b], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});

/** A round verdict glyph: ✓ on green or ✗ on red (glyph and colour, never colour alone). */
export const Verdict: React.FC<{ok: boolean; size?: number; style?: React.CSSProperties}> = ({ok, size = 40, style}) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: '50%',
      flex: 'none',
      background: ok ? color.ok : color.bad,
      color: '#fff',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: size * 0.62,
      fontWeight: 800,
      lineHeight: 1,
      boxShadow: '0 0 0 3px rgba(255,255,255,0.9)',
      ...style,
    }}
  >
    {ok ? '✓' : '✕'}
  </div>
);

/** A caption chip over footage or beside it: white, ink type, an optional verdict glyph. */
export const Chip: React.FC<{children: React.ReactNode; ok?: boolean; tone?: 'light' | 'dark' | 'bad' | 'ok'; size?: number; style?: React.CSSProperties}> = ({
  children,
  ok,
  tone = 'light',
  size = 30,
  style,
}) => {
  const bg = tone === 'dark' ? color.night : tone === 'bad' ? color.bad : tone === 'ok' ? color.ok : '#fff';
  const fg = tone === 'light' ? color.ink : '#fff';
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 14,
        padding: `${size * 0.36}px ${size * 0.7}px ${size * 0.36}px ${ok === undefined ? size * 0.7 : size * 0.4}px`,
        borderRadius: 999,
        background: bg,
        color: fg,
        fontSize: size,
        fontWeight: 700,
        whiteSpace: 'nowrap',
        boxShadow: shadow.pill,
        ...style,
      }}
    >
      {ok === undefined ? null : <Verdict ok={ok} size={size * 1.25} style={{boxShadow: 'none'}} />}
      {children}
    </div>
  );
};

/** A small uppercase label, at least 24 px. */
export const Label: React.FC<{children: React.ReactNode; style?: React.CSSProperties; tone?: string}> = ({children, style, tone}) => (
  <div style={{fontSize: 24, fontWeight: 750, letterSpacing: '0.12em', textTransform: 'uppercase', color: tone ?? color.ink3, ...style}}>{children}</div>
);

/** A typed monospace name (a job, a check), wrapping at its hyphens. */
export const MonoName: React.FC<{children: string; size?: number; style?: React.CSSProperties}> = ({children, size = 30, style}) => (
  <div style={{fontFamily: font.mono, fontSize: size, fontWeight: 600, lineHeight: 1.3, color: color.ink, overflowWrap: 'break-word', ...style}}>{children}</div>
);

/** The film's dot motif: one dot per try, filling in turn. */
export const OutcomeDots: React.FC<{outcomes: string; start: number; size?: number; gap?: number; stagger?: number}> = ({outcomes, start, size = 30, gap = 10, stagger = 2}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return (
    <div style={{display: 'flex', gap}}>
      {[...outcomes].map((c, i) => {
        const at = start + i * stagger;
        const p = spring({frame: frame - at, fps, config: {damping: 13, stiffness: 160, mass: 0.6}});
        const on = frame >= at;
        const fill = c === '1' ? color.ok : color.bad;
        return (
          <div key={i} style={{position: 'relative', width: size, height: size, flex: 'none'}}>
            <div
              style={{
                width: size,
                height: size,
                borderRadius: '50%',
                background: on ? fill : 'transparent',
                boxShadow: on ? 'none' : `inset 0 0 0 ${Math.max(2, size * 0.14)}px ${color.line}`,
                transform: `scale(${on ? Math.min(p, 1.15) : 1})`,
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

/**
 * Twenty recorded runs at once (a local variant of the shared Mosaic): each tile plays its whole
 * run fitted to `playFrames`, then at its verdict its border turns green or red and a ✓/✗ badge
 * lands in its corner, so the verdict never rests on colour alone. `hold` freezes every tile on
 * its last frame (for a scene that opens on a finished mosaic).
 */
export const ExamMosaic: React.FC<{
  clips: string[];
  columns?: number;
  tile: number;
  gap?: number;
  startAt: number;
  playFrames: number;
  verdictAt: number;
  appearAt?: number;
  hold?: boolean;
  dimUntil?: number;
}> = ({clips, columns = 5, tile, gap = 12, startAt, playFrames, verdictAt, appearAt = 0, hold, dimUntil}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const film = useFilm();
  return (
    <div style={{display: 'grid', gridTemplateColumns: `repeat(${columns}, ${tile}px)`, gap}}>
      {clips.map((id, i) => {
        const clip = film.clips[id];
        const appear = hold ? 1 : interpolate(frame, [appearAt + i * 0.8, appearAt + i * 0.8 + 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
        const vAt = verdictAt + i * 1.2;
        const verdict = hold || frame >= vAt;
        const pop = hold ? 1 : spring({frame: frame - vAt, fps, config: springs.lively});
        const ok = Boolean(clip?.delivered);
        const border = verdict ? (ok ? color.ok : color.bad) : 'rgba(0,0,0,0)';
        let segments: Segment[] | undefined;
        if (clip) {
          const steps = (clip.frames - 1) * clip.frame_every;
          const rate = ((steps / clip.fps) * FPS) / Math.max(1, playFrames);
          segments = [{from: clip.first_step, to: clip.first_step + steps, rate}];
        }
        const dim = dimUntil !== undefined && !verdict ? interpolate(frame, [dimUntil - 12, dimUntil], [0.45, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}) : 1;
        return (
          <div
            key={id}
            style={{
              position: 'relative',
              opacity: appear * dim,
              transform: `scale(${0.9 + 0.1 * appear})`,
              borderRadius: 16,
              boxShadow: `0 0 0 6px ${border}, ${shadow.card}`,
            }}
          >
            <Footage clip={id} segments={segments} startAt={hold ? -100000 : startAt} width={tile} height={tile} radius={12} />
            {verdict ? (
              <div style={{position: 'absolute', right: -10, top: -10, transform: `scale(${Math.min(pop, 1.2)})`}}>
                <Verdict ok={ok} size={Math.round(tile * 0.24)} />
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
};

/** Delivered count of a mosaic, from the clips' recorded verdicts. */
export const useDelivered = (ids: string[]): number => {
  const film = useFilm();
  return ids.filter((id) => film.clips[id]?.delivered).length;
};
