// Robot footage: a recorded run (or its verified re-render) in a rounded frame, played on a
// time map of control steps, with overlays drawn from the clip's recorded track so they
// follow the real objects. Nothing here invents motion: the step shown at each frame is a
// step of the recorded run.
import React, {useEffect, useState} from 'react';
import {continueRender, delayRender, Freeze, OffthreadVideo, staticFile, useCurrentFrame, interpolate, Easing} from 'remotion';
import {color, FPS, shadow, space, font} from '../theme';
import {useFilm, type Clip, type TrackFrame} from '../data';

/** A segment of playback: steps [from, to] played at `rate` (1 = real time, 20 steps a second). */
export type Segment = {from: number; to: number; rate?: number};

/**
 * Maps a frame (relative to playback start) to a control step, playing the segments in
 * order and holding the last step after the end. Before `startAt` the first step is held.
 */
export const stepAt = (frame: number, clip: Clip, segments: Segment[], startAt = 0): number => {
  let t = frame - startAt; // frames into playback
  if (t <= 0) return segments[0]?.from ?? clip.first_step;
  for (const s of segments) {
    const rate = s.rate ?? 1;
    const steps = s.to - s.from;
    const frames = (steps / clip.fps) * FPS / rate; // composition frames this segment lasts
    if (t <= frames) return s.from + (t / frames) * steps;
    t -= frames;
  }
  const last = segments[segments.length - 1];
  return last ? last.to : clip.first_step;
};

/** Composition frames a list of segments takes to play. */
export const segmentsFrames = (clip: Clip, segments: Segment[]): number =>
  segments.reduce((sum, s) => sum + ((s.to - s.from) / clip.fps) * FPS / (s.rate ?? 1), 0);

export const useClip = (id: string): Clip | null => {
  const film = useFilm();
  return film.clips[id] ?? null;
};

/** The clip's overlay track (per-frame projected positions and recorded flags), if it has one. */
export const useTrack = (clip: Clip | null): TrackFrame[] | null => {
  const [track, setTrack] = useState<TrackFrame[] | null>(null);
  const [handle] = useState(() => (clip?.track ? delayRender(`track ${clip.id}`) : null));
  useEffect(() => {
    if (!clip?.track || handle === null) return;
    fetch(staticFile(clip.track))
      .then((r) => r.json())
      .then((j) => {
        setTrack((j.frames ?? j) as TrackFrame[]);
        continueRender(handle);
      })
      .catch((e) => {
        console.error('track', clip.id, e);
        continueRender(handle);
      });
  }, [clip, handle]);
  return track;
};

/** The track entry at a (fractional) step, linearly interpolated between recorded frames. */
export const trackAt = (track: TrackFrame[] | null, clip: Clip, step: number): TrackFrame | null => {
  if (!track || track.length === 0) return null;
  const idx = (step - clip.first_step) / clip.frame_every;
  const i0 = Math.max(0, Math.min(track.length - 1, Math.floor(idx)));
  const i1 = Math.min(track.length - 1, i0 + 1);
  const f = Math.max(0, Math.min(1, idx - i0));
  const a = track[i0];
  const b = track[i1];
  if (f === 0 || a === b) return a;
  const objects: TrackFrame['objects'] = {};
  for (const k of Object.keys(a.objects)) {
    const pa = a.objects[k];
    const pb = b.objects[k] ?? pa;
    objects[k] = {x: pa.x + (pb.x - pa.x) * f, y: pa.y + (pb.y - pa.y) * f, depth: pa.depth, visible: pa.visible};
  }
  const gripper = a.gripper && b.gripper ? {...a.gripper, x: a.gripper.x + (b.gripper.x - a.gripper.x) * f, y: a.gripper.y + (b.gripper.y - a.gripper.y) * f} : a.gripper;
  return {...a, objects, gripper};
};

export type OverlayContext = {
  step: number; // the control step on screen
  at: TrackFrame | null; // the recorded track at that step
  /** Display coordinates (px inside the frame) of a point in the clip's pixel space. */
  toBox: (x: number, y: number) => {x: number; y: number};
  scale: number;
  box: {w: number; h: number};
};

/**
 * A clip in a rounded frame. `segments` say which control steps play and how fast;
 * `startAt` is the frame (in the parent sequence) when playback starts. Children render
 * overlays from the OverlayContext. A clip that is not yet rendered shows a labelled
 * placeholder, so scenes can be built before the footage exists.
 */
export const Footage: React.FC<{
  clip: string;
  segments?: Segment[];
  startAt?: number;
  width: number; // display box
  height: number;
  fit?: 'cover' | 'contain';
  focus?: {x: number; y: number}; // 0..1 point of the clip kept in view when covering
  zoom?: {from: number; to: number; at: [number, number]; focus?: {x: number; y: number}}; // a slow push-in
  style?: React.CSSProperties;
  radius?: number;
  label?: React.ReactNode; // a caption chip top-left
  children?: (ctx: OverlayContext) => React.ReactNode;
}> = ({clip: id, segments, startAt = 0, width, height, fit = 'cover', focus = {x: 0.5, y: 0.5}, zoom, style, radius = space.radiusFrame, label, children}) => {
  const frame = useCurrentFrame();
  const clip = useClip(id);
  const track = useTrack(clip);
  if (!clip) {
    return (
      <div style={{width, height, borderRadius: radius, background: color.paper2, boxShadow: shadow.frame, display: 'flex', alignItems: 'center', justifyContent: 'center', ...style}}>
        <div style={{fontFamily: font.mono, fontSize: 24, color: color.ink3}}>clip {id} not rendered yet</div>
      </div>
    );
  }
  const segs = segments ?? [{from: clip.first_step, to: clip.first_step + (clip.frames - 1) * clip.frame_every}];
  const step = stepAt(frame, clip, segs, startAt);
  const videoFrame = ((step - clip.first_step) / clip.fps) * FPS; // composition frames into the video
  // cover/contain geometry, then an optional slow zoom about a focus point
  const s0 = fit === 'cover' ? Math.max(width / clip.width, height / clip.height) : Math.min(width / clip.width, height / clip.height);
  let z = 1;
  let zf = focus;
  if (zoom) {
    z = interpolate(frame, zoom.at, [zoom.from, zoom.to], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic)});
    zf = zoom.focus ?? focus;
  }
  const scale = s0 * z;
  const vw = clip.width * scale;
  const vh = clip.height * scale;
  // keep the focus point at the same place in the box
  const ox = Math.min(0, Math.max(width - vw, width / 2 - zf.x * vw));
  const oy = Math.min(0, Math.max(height - vh, height / 2 - zf.y * vh));
  const offX = fit === 'contain' ? (width - vw) / 2 : ox;
  const offY = fit === 'contain' ? (height - vh) / 2 : oy;
  const ctx: OverlayContext = {
    step,
    at: trackAt(track, clip, step),
    toBox: (x, y) => ({x: offX + x * scale, y: offY + y * scale}),
    scale,
    box: {w: width, h: height},
  };
  return (
    <div style={{position: 'relative', width, height, borderRadius: radius, overflow: 'hidden', boxShadow: shadow.frame, background: '#ddd', ...style}}>
      <div style={{position: 'absolute', left: offX, top: offY, width: vw, height: vh}}>
        <Freeze frame={Math.max(0, Math.round(videoFrame))}>
          <OffthreadVideo src={staticFile(clip.src)} muted style={{width: '100%', height: '100%'}} />
        </Freeze>
      </div>
      {children ? <div style={{position: 'absolute', inset: 0}}>{children(ctx)}</div> : null}
      {label ? <div style={{position: 'absolute', left: 22, top: 22}}>{label}</div> : null}
    </div>
  );
};

/** A ring that follows a tracked object; red while `alert`, else ink. */
export const Ring: React.FC<{ctx: OverlayContext; object: string; r?: number; alert?: boolean; show?: number; tone?: string}> = ({ctx, object, r = 70, alert, show = 1, tone}) => {
  const o = ctx.at?.objects[object];
  if (!o || !o.visible || show <= 0) return null;
  const p = ctx.toBox(o.x, o.y);
  const c = tone ?? (alert ? color.bad : color.ink);
  return (
    <div
      style={{
        position: 'absolute',
        left: p.x - r,
        top: p.y - r,
        width: r * 2,
        height: r * 2,
        borderRadius: '50%',
        border: `6px solid ${c}`,
        boxShadow: `0 0 0 4px rgba(255,255,255,0.75)${alert ? `, 0 0 40px ${color.bad}` : ''}`,
        opacity: show,
        transform: `scale(${0.8 + 0.2 * show})`,
      }}
    />
  );
};

/** A label chip pinned near a tracked object (offset in px). */
export const PinLabel: React.FC<{ctx: OverlayContext; object: string; dx?: number; dy?: number; children: React.ReactNode; tone?: 'plain' | 'ok' | 'bad' | 'dark'; show?: number}> = ({
  ctx,
  object,
  dx = 40,
  dy = -60,
  children,
  tone = 'plain',
  show = 1,
}) => {
  const o = ctx.at?.objects[object];
  if (!o || !o.visible || show <= 0) return null;
  const p = ctx.toBox(o.x, o.y);
  const bg = tone === 'ok' ? color.ok : tone === 'bad' ? color.bad : tone === 'dark' ? color.night : '#fff';
  const fg = tone === 'plain' ? color.ink : '#fff';
  return (
    <div style={{position: 'absolute', left: p.x + dx, top: p.y + dy, opacity: show, background: bg, color: fg, padding: '8px 18px', borderRadius: 999, fontSize: 26, fontWeight: 700, whiteSpace: 'nowrap', boxShadow: shadow.pill}}>
      {children}
    </div>
  );
};

/** Whether the recorded run had `object` in contact with the robot at this step. */
export const touching = (ctx: OverlayContext, object: string): boolean => Boolean(ctx.at?.touch?.[object]);
