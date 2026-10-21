// r1-measure: the diagnosis as a measurement, one idea at a time, on the agent's own first runs
// (practice layout 2; nothing from the final exam).
//   1. The run becomes data: the bowl's recorded height, the recorded hand contact and the
//      recorded bowl goal over the control steps, the switch marked, a playhead in step with a
//      small video of the same run; the hand-off (contact still on as the next skill starts)
//      marked in amber.
//   2. The check the agent wrote, and its finding as amber hand-off markers, one per first
//      run: all of them switched while the bowl was touched and moving. Red and green stay
//      reserved for delivery (the plain chain delivered some of these runs).
//   3. The lesson, in one line.
// No quotation in this beat.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useBeat, useCue, useFilm, useNumber} from '../../data';
import {color, font, shadow} from '../../theme';
import {CLIPS} from './clips';
import {Chip, JobStage, Label, MonoName, Verdict, keyed, orderKeys, ramp, stepOfKeys, useArrive, type Key} from './common';
import {handoffMoments, practiceStartOf, type HandoffTrace} from './moments';

const CHART = {x: 120, y: 236, w: 1150, h: 640};
const VIDEO = {x: 1330, y: 236, w: 470};

/** An amber hand-off marker: a diamond (never a dot, so it never reads as a delivery verdict). */
export const HandoffMark: React.FC<{size?: number; on: boolean; pop?: number}> = ({size = 34, on, pop = 1}) => (
  <div style={{width: size, height: size, flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
    <div
      style={{
        width: size * 0.74,
        height: size * 0.74,
        transform: `rotate(45deg) scale(${on ? Math.min(1.15, pop) : 1})`,
        borderRadius: size * 0.12,
        background: on ? color.lead : 'transparent',
        boxShadow: on ? 'none' : `inset 0 0 0 ${Math.max(2, size * 0.1)}px ${color.line}`,
      }}
    />
  </div>
);

type Strip = {label: string; color: string; intervals: [number, number][]};

/** The recorded run as a chart (a local variant of the shared TraceChart, at film size). */
const RunChart: React.FC<{
  steps: [number, number];
  y: [number, number];
  values: number[];
  strips: Strip[];
  switchStep: number;
  handoff: [number, number];
  yLabels: {value: number; label: string}[];
  reveal: number;
  handoffShow: number;
}> = ({steps, y, values, strips, switchStep, handoff, yLabels, reveal, handoffShow}) => {
  const {w: W, h: H} = CHART;
  const padL = 250;
  const padR = 40;
  const padT = 56;
  const plotH = 340;
  const stripH = 44;
  const stripTop = padT + plotH + 46;
  const stripsBottom = stripTop + strips.length * (stripH + 16) - 16;
  const X = (s: number) => padL + ((s - steps[0]) / (steps[1] - steps[0])) * (W - padL - padR);
  const Y = (v: number) => padT + plotH - ((v - y[0]) / (y[1] - y[0])) * plotH;
  const upTo = Math.max(steps[0], Math.min(steps[1], reveal));
  const pts: string[] = [];
  for (let s = Math.ceil(steps[0]); s <= upTo && s < values.length; s++) pts.push(`${pts.length ? 'L' : 'M'}${X(s).toFixed(1)},${Y(values[s]).toFixed(1)}`);
  const tipS = Math.min(values.length - 1, Math.floor(upTo));
  return (
    <svg width={W} height={H} style={{overflow: 'visible', fontFamily: font.sans}}>
      {/* the switch */}
      <line x1={X(switchStep)} x2={X(switchStep)} y1={padT - 34} y2={stripsBottom + 8} stroke={color.ink} strokeWidth={3} strokeDasharray="8 8" />
      <text x={X(switchStep) + 12} y={padT - 10} fontSize={26} fontWeight={750} fill={color.ink}>
        next skill starts
      </text>
      {/* the two heights, named; no numbers */}
      <line x1={padL} x2={W - padR} y1={padT + plotH} y2={padT + plotH} stroke={color.line} strokeWidth={2} />
      {yLabels.map((l, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={Y(l.value)} y2={Y(l.value)} stroke={color.line} strokeWidth={2} strokeDasharray="2 10" strokeLinecap="round" />
          <text x={padL - 18} y={Y(l.value) + 9} textAnchor="end" fontSize={26} fontWeight={600} fill={color.ink2}>
            {l.label}
          </text>
        </g>
      ))}
      <text x={padL - 18} y={padT - 10} textAnchor="end" fontSize={24} fontWeight={700} fill={color.ink3}>
        bowl height
      </text>
      <path d={pts.join(' ')} fill="none" stroke={color.ink} strokeWidth={7} strokeLinejoin="round" strokeLinecap="round" />
      {pts.length ? <circle cx={X(upTo)} cy={Y(values[tipS])} r={10} fill={color.ink} stroke="#fff" strokeWidth={3} /> : null}
      {/* the recorded flags, as recorded */}
      {strips.map((st, i) => {
        const yy = stripTop + i * (stripH + 16);
        return (
          <g key={i}>
            <text x={padL - 18} y={yy + 27} textAnchor="end" fontSize={25} fontWeight={600} fill={color.ink2}>
              {st.label}
            </text>
            <rect x={padL} y={yy} width={W - padL - padR} height={stripH} rx={8} fill="#E9E4D9" />
            {st.intervals
              .filter(([a]) => a <= upTo && a < steps[1])
              .map(([a, b], k) => (
                <rect key={k} x={X(Math.max(a, steps[0]))} y={yy} width={Math.max(4, X(Math.min(b, upTo, steps[1])) - X(Math.max(a, steps[0])))} height={stripH} rx={6} fill={st.color} />
              ))}
          </g>
        );
      })}
      {/* the hand-off: contact still on as the next skill starts */}
      {handoffShow > 0 ? (
        <g opacity={handoffShow}>
          <rect x={X(handoff[0]) - 8} y={stripTop - 12} width={X(handoff[1]) - X(handoff[0]) + 16} height={stripsBottom - stripTop + 24} rx={12} fill="none" stroke={color.lead} strokeWidth={6} />
          <text x={X(handoff[0]) - 8} y={stripsBottom + 50} fontSize={28} fontWeight={800} fill={color.lead}>
            the hand-off: the bowl still touched
          </text>
        </g>
      ) : null}
      {/* the playhead */}
      {reveal > steps[0] && reveal < steps[1] ? <line x1={X(upTo)} x2={X(upTo)} y1={padT - 12} y2={stripsBottom + 4} stroke={color.agent} strokeWidth={4} /> : null}
    </svg>
  );
};

export const R1Measure: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const chain = film.traces['r1-practice2-chain'] as HandoffTrace;
  const m = handoffMoments(chain);
  const layout = practiceStartOf(film, 'r1', chain.episode);
  const last = beat.frames - 1;
  const checkName = useNumber('r1_check_name');
  const firstRuns = useNumber('r1_plain_n');
  const nRuns = Number(firstRuns) || 0;

  const studiesF = useCue('studies', 0.03);
  const checkF = useCue('check', 0.17);
  const findingF = useCue('finding', 0.33);
  const touchedF = useCue('touched', 0.55);
  const lessonF = useCue('lesson', 0.65);

  // phase 1: the playhead and the small video share these keys
  const x0 = 20;
  const x1 = Math.min(chain.height_m.length - 1, m.fallEnd + 30);
  const keys: Key[] = orderKeys(
    [
      [studiesF + 4, x0],
      [checkF, m.switchStep],
      [checkF + 20, m.holdEnd],
      [Math.max(checkF + 40, findingF - 24), x1],
    ],
    last,
  );
  const videoId = film.clips[CLIPS.r1ChainPractice2] ? CLIPS.r1ChainPractice2 : null;
  const clip = useClip(videoId ?? CLIPS.r1ChainPractice2);
  const play = keyed(clip, keys);
  const step = stepOfKeys(frame, keys);
  const peak = Math.max(...chain.height_m);
  const yRange: [number, number] = [m.table - (peak - m.table) * 0.12, peak + (peak - m.table) * 0.08];

  const p1In = useArrive(0);
  const p1Out = ramp(frame, findingF - 14, findingF - 2);
  const handoffShow = ramp(frame, checkF + 22, checkF + 34);
  const p2In = useArrive(findingF - 4);
  const p2Out = ramp(frame, lessonF - 12, lessonF);
  const p3In = useArrive(lessonF - 2);
  const captionIn = ramp(frame, touchedF - 4, touchedF + 10);

  return (
    <JobStage job={1} step={1}>
      {/* 1: the run becomes data */}
      {p1Out < 1 ? (
        <div style={{position: 'absolute', inset: 0, opacity: p1In * (1 - p1Out)}}>
          <div style={{position: 'absolute', left: CHART.x, top: 136}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>
              The run becomes data{' '}
              <span style={{fontSize: 36, fontWeight: 650, color: color.ink3}}>· {layout === null ? 'a practice layout' : `practice layout ${layout}`}, the plain chain</span>
            </div>
          </div>
          <div style={{position: 'absolute', left: CHART.x, top: CHART.y}}>
            <RunChart
              steps={[x0, x1]}
              y={yRange}
              values={chain.height_m}
              strips={[
                {label: 'hand on the bowl', color: color.ink, intervals: chain.touch},
                {label: 'bowl on the cabinet', color: color.teal, intervals: chain.goal},
              ]}
              switchStep={m.switchStep}
              handoff={[m.switchStep, m.holdEnd]}
              yLabels={[
                {value: m.cabinet, label: 'cabinet top'},
                {value: m.table, label: 'table'},
              ]}
              reveal={step}
              handoffShow={handoffShow}
            />
          </div>
          {videoId ? (
            <div style={{position: 'absolute', left: VIDEO.x, top: VIDEO.y}}>
              <Footage clip={videoId} segments={play.segments} startAt={play.startAt} width={VIDEO.w} height={Math.round((VIDEO.w * 9) / 16)} radius={20} label={<Chip size={24}>the same run</Chip>} />
            </div>
          ) : null}
        </div>
      ) : null}

      {/* 2: the check, and what it found in every first run */}
      {frame >= findingF - 8 && p2Out < 1 ? (
        <div style={{position: 'absolute', inset: 0, opacity: Math.min(1, p2In) * (1 - p2Out)}}>
          <div
            style={{
              position: 'absolute',
              left: 260,
              right: 260,
              top: 236,
              background: color.card,
              borderRadius: 28,
              boxShadow: shadow.card,
              borderTop: `14px solid ${color.teal}`,
              padding: '36px 56px 44px',
              transform: `translateY(${(1 - Math.min(1, p2In)) * 30}px)`,
            }}
          >
            <Label tone={color.teal} style={{marginBottom: 14}}>
              check · written by the agent
            </Label>
            <MonoName size={40}>{checkName}</MonoName>
            <div style={{fontSize: 36, fontWeight: 560, color: color.ink2, marginTop: 16, lineHeight: 1.3}}>Does the bowl stay put while the next skill runs?</div>
            <div style={{marginTop: 36, paddingTop: 30, borderTop: `2px solid ${color.line}`}}>
              <div style={{display: 'flex', alignItems: 'center', gap: 14}}>
                {Array.from({length: nRuns}, (_, i) => {
                  const at = findingF + 6 + i * 4;
                  const on = frame >= at;
                  return <HandoffMark key={i} size={56} on={on} pop={on ? 1 + 0.15 * Math.max(0, 1 - (frame - at) / 8) : 1} />;
                })}
              </div>
              <div style={{marginTop: 22, opacity: captionIn}}>
                <div style={{fontSize: 40, fontWeight: 800, color: color.ink, lineHeight: 1.2}}>all {firstRuns} first runs switched while the bowl was touched and moving</div>
              </div>
              <div style={{marginTop: 18, display: 'flex', alignItems: 'center', gap: 10, fontSize: 24, fontWeight: 600, color: color.ink2}}>
                <HandoffMark size={26} on />
                <span style={{color: color.ink, fontWeight: 720}}>one marker = one first run,</span>
                <span>flagged at the hand-off</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* 3: the lesson */}
      {frame >= lessonF - 4 ? (
        <div style={{position: 'absolute', left: 260, right: 260, top: 300, opacity: Math.min(1, p3In), transform: `translateY(${(1 - Math.min(1, p3In)) * 30}px)`}}>
          <Label style={{marginBottom: 22}}>what it shows</Label>
          <div style={{fontSize: 64, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.12, color: color.ink}}>The placing skill reports success before the job is done.</div>
          <div style={{marginTop: 44, display: 'flex', alignItems: 'center', gap: 22, flexWrap: 'wrap'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 14, background: color.night, color: '#fff', borderRadius: 999, padding: '12px 30px 12px 14px', fontSize: 30, fontWeight: 700, boxShadow: shadow.pill}}>
              <Verdict ok size={44} style={{boxShadow: 'none'}} />
              bowl on cabinet
            </div>
            <div style={{fontSize: 32, fontWeight: 600, color: color.ink2}}>while</div>
            <div style={{display: 'flex', alignItems: 'center', gap: 12, background: color.card, borderRadius: 999, padding: '10px 28px 10px 14px', fontSize: 30, fontWeight: 700, color: color.ink, boxShadow: `0 0 0 4px ${color.lead}, ${shadow.pill}`}}>
              <HandoffMark size={40} on />
              the hand still holds the bowl
            </div>
          </div>
        </div>
      ) : null}
    </JobStage>
  );
};
