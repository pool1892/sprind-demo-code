// Diagram parts shared by the opening scenes: the two-tile job that snaps and cracks, a status
// pill, a person icon, a clock and a cost meter (no digits: they show that time and cost run,
// not how much), and a flat robot-cell illustration.
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {color, shadow} from '../../theme';
import {arrive, mix, span} from './kit';

const TILE_W = 440;
const TILE_H = 120;

/** The job: two dark tiles that slide together at `at`; `crack` 0..1 pushes them apart and tilts them. */
export const JobTiles: React.FC<{x: number; y: number; at: number; crack: number; labels?: [string, string]; w?: number; h?: number; fontSize?: number}> = ({
  x,
  y,
  at,
  crack,
  labels = ['bowl on the cabinet', 'drawer open'],
  w = TILE_W,
  h = TILE_H,
  fontSize = 40,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = arrive(frame, fps, at - 4);
  const snap = span(frame, at + 8, at + 30);
  const apart = crack * 18;
  const tiles = [
    {label: labels[0], left: x - apart, rot: -1.6 * crack},
    {label: labels[1], left: mix(snap, x + w + 160, x + w) + apart, rot: 1.6 * crack},
  ];
  return (
    <>
      {tiles.map((t, i) => (
        <div
          key={t.label}
          style={{
            position: 'absolute',
            left: t.left,
            top: y,
            width: w,
            height: h,
            boxSizing: 'border-box',
            borderRadius: i === 0 ? '24px 0 0 24px' : '0 24px 24px 0',
            background: color.ink,
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize,
            fontWeight: 740,
            boxShadow: shadow.card,
            opacity: p,
            transform: `translateY(${(1 - p) * 30}px) rotate(${t.rot}deg)`,
            transformOrigin: i === 0 ? 'right center' : 'left center',
            borderLeft: i === 1 && snap > 0.98 ? '2px solid #3A3D46' : undefined,
          }}
        >
          {t.label}
        </div>
      ))}
      <div style={{position: 'absolute', left: x + w + 80 - 16, top: y + h / 2 - 32, fontSize: 52, fontWeight: 700, color: color.ink3, opacity: p * (1 - snap)}}>+</div>
    </>
  );
};

/** A red zigzag down a seam at (x, y), drawn by `p`. */
export const Crack: React.FC<{x: number; y: number; p: number; height?: number}> = ({x, y, p, height = 120}) => {
  if (p <= 0.01) return null;
  const k = height / 120;
  const pts = [
    [0, -14],
    [-16, 18],
    [14, 42],
    [-12, 70],
    [16, 98],
    [-10, 124],
    [4, 140],
  ];
  const d = pts.map(([dx, dy], i) => `${i ? 'L' : 'M'}${x + dx},${y + dy * k - 4}`).join(' ');
  const len = 260 * k;
  return (
    <svg style={{position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none'}}>
      <path d={d} fill="none" stroke="#fff" strokeWidth={16} strokeLinejoin="round" strokeLinecap="round" strokeDasharray={len} strokeDashoffset={len * (1 - p)} />
      <path d={d} fill="none" stroke={color.bad} strokeWidth={8} strokeLinejoin="round" strokeLinecap="round" strokeDasharray={len} strokeDashoffset={len * (1 - p)} />
    </svg>
  );
};

export const StatusPill: React.FC<{x: number; y: number; tone: 'ok' | 'bad'; show: number; children: React.ReactNode}> = ({x, y, tone, show, children}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      transform: `translateY(-50%) scale(${0.85 + 0.15 * Math.min(1, show)})`,
      transformOrigin: 'left center',
      opacity: Math.min(1, show * 1.5),
      padding: '12px 26px',
      borderRadius: 999,
      background: tone === 'ok' ? color.ok : color.bad,
      color: '#fff',
      fontSize: 32,
      fontWeight: 720,
      whiteSpace: 'nowrap',
      boxShadow: shadow.pill,
    }}
  >
    {children}
  </div>
);

/** A person: a flat silhouette in a disc. */
export const Person: React.FC<{size: number; tone?: string; style?: React.CSSProperties}> = ({size, tone = color.ink2, style}) => (
  <div style={{width: size, height: size, borderRadius: '50%', background: tone, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', overflow: 'hidden', ...style}}>
    <svg width={size * 0.78} height={size * 0.78} viewBox="0 0 100 100">
      <circle cx={50} cy={38} r={20} fill="#fff" />
      <path d="M12,100 C14,70 32,62 50,62 C68,62 86,70 88,100 Z" fill="#fff" />
    </svg>
  </div>
);

/** A clock face whose hand turns `turns` times (no digits). */
export const Clock: React.FC<{size: number; turns: number; tone?: string}> = ({size, turns, tone = color.ink}) => {
  const a = turns * 2 * Math.PI - Math.PI / 2;
  const b = (turns / 12) * 2 * Math.PI - Math.PI / 2;
  return (
    <svg width={size} height={size} viewBox="0 0 100 100">
      <circle cx={50} cy={50} r={44} fill="#fff" stroke={tone} strokeWidth={7} />
      <line x1={50} y1={50} x2={50 + 32 * Math.cos(a)} y2={50 + 32 * Math.sin(a)} stroke={tone} strokeWidth={6} strokeLinecap="round" />
      <line x1={50} y1={50} x2={50 + 20 * Math.cos(b)} y2={50 + 20 * Math.sin(b)} stroke={tone} strokeWidth={8} strokeLinecap="round" />
    </svg>
  );
};

/** A cost meter: a coin and a bar that fills (no digits). */
export const CostMeter: React.FC<{width: number; fill: number; tone?: string}> = ({width, fill, tone = color.ink}) => (
  <div style={{display: 'flex', alignItems: 'center', gap: 14}}>
    <div style={{width: 52, height: 52, borderRadius: '50%', background: tone, color: '#fff', fontSize: 30, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>€</div>
    <div style={{width, height: 22, borderRadius: 11, background: color.line, overflow: 'hidden'}}>
      <div style={{width: `${Math.min(1, fill) * 100}%`, height: '100%', background: tone, borderRadius: 11}} />
    </div>
  </div>
);

/** A flat robot cell: a table, an arm, a conveyor with boxes. An illustration, not footage. */
export const RobotCell: React.FC<{width: number; t?: number}> = ({width, t = 0}) => {
  const h = width * 0.62;
  const shift = (t * 60) % 60;
  return (
    <svg width={width} height={h} viewBox="0 0 500 310">
      <rect x={0} y={250} width={500} height={14} rx={7} fill={color.line} />
      <rect x={250} y={190} width={240} height={26} rx={13} fill={color.ink2} />
      {[0, 1, 2, 3].map((i) => (
        <rect key={i} x={262 + i * 60 + shift} y={156} width={40} height={34} rx={5} fill={color.wood} opacity={262 + i * 60 + shift > 470 ? 0 : 1} />
      ))}
      <rect x={60} y={216} width={110} height={34} rx={8} fill={color.ink} />
      <path d="M115,216 L115,150 L190,96 L262,128" fill="none" stroke={color.ink} strokeWidth={22} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={115} cy={150} r={16} fill={color.paper2} stroke={color.ink} strokeWidth={8} />
      <circle cx={190} cy={96} r={16} fill={color.paper2} stroke={color.ink} strokeWidth={8} />
      <path d="M262,128 L262,150 M250,150 L274,150" stroke={color.ink} strokeWidth={10} strokeLinecap="round" />
    </svg>
  );
};
