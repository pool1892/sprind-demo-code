// r2-fail: a harder job, described before it fails. First the task: the
// request card (its first sentence, verbatim) with its two goals lighting as the voice names
// them, a flat drawing of the station from above with each goal marked on the same words, and
// the robot's two skills as tiles, each with its move drawn, on "pushes" and "places". Then the
// plain chain: the drawing gives way to the recorded failures on two practice layouts (the
// push into the cheese, the hand on the cheese), the cheese named by a label as the voice names
// it, no ring, and the baseline on the locked layouts as dots.
import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useFilm, type Clip} from '../../data';
import {useBeat, useCue, useNumber, useQuote} from '../../data';
import {clipOr, practiceStartOf} from './moments';
import {DotLegend} from './DotLegend';
import {color, font, shadow} from '../../theme';
import {CLIPS} from './clips';
import {Chip, JobStage, Label, OutcomeDots, firstContact, keyed, objectBox, orderKeys, ramp, useArrive, useTrackFile, type Key, type ObjBox, type TrackFile} from './common';

const CHEESE = 'cream_cheese_1';
const PLATE = 'plate_1';

/** Where to centre a slow push-in on an overhead clip: between the plate and the cheese at `step`. */
const focusOf = (track: TrackFile | null, clip: Clip | null, step: number): {x: number; y: number} => {
  const c = objectBox(track, step, CHEESE);
  const p = objectBox(track, step, PLATE);
  if (!c || !p || !clip) return {x: 0.5, y: 0.5};
  return {x: (c.cx + p.cx) / 2 / clip.width, y: (c.cy + p.cy) / 2 / clip.height};
};
const FOOT = {x: 770, y: 150, w: 1030, h: Math.round((1030 * 9) / 16)};
const DIAG = {x: 840, y: 196, scale: 1.48};

/**
 * The station from above, drawn (an illustration, not footage): the robot, cabinet, bowl, stove
 * and plate; `plateGoal` and `bowlGoal` mark where each must end up, `push` and `place` draw
 * each skill's move.
 */
const StationDiagram: React.FC<{build: number; plateGoal: number; bowlGoal: number; push: number; place: number}> = ({build, plateGoal, bowlGoal, push, place}) => {
  const a = (i: number) => interpolate(build, [i * 0.14, i * 0.14 + 0.3], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const plate = {x: 168, y: 318};
  const target = {x: 488, y: 300};
  const label = (x: number, y: number, t: string, o: number, fill: string = color.ink2, weight = 650) => (
    <text x={x} y={y} textAnchor="middle" fontSize={26} fontWeight={weight} fill={fill} opacity={o} fontFamily={font.sans}>
      {t}
    </text>
  );
  // the push: a dashed line from the plate to its goal, drawn on with the voice
  const px0 = plate.x + 56;
  const px1 = px0 + (target.x - 70 - px0) * push;
  const py = (x: number) => plate.y + ((target.y - plate.y) * (x - plate.x)) / (target.x - plate.x);
  // the placement: an arc from the bowl up onto the cabinet
  const arc = 'M244 140 Q190 60 132 104';
  return (
    <svg width={600} height={440} viewBox="0 0 600 440" style={{overflow: 'visible'}}>
      <rect x={14} y={44} width={572} height={380} rx={28} fill="#EFE7D8" stroke={color.line} strokeWidth={3} opacity={a(0)} />
      {/* the robot, at the back */}
      <path d="M250 44 a50 50 0 0 1 100 0 Z" fill="#C9CBD3" opacity={a(0)} />
      {label(300, 30, 'robot', a(0))}
      {/* cabinet, and the bowl's goal on top of it */}
      <rect x={40} y={80} width={128} height={136} rx={10} fill="#3A3330" opacity={a(1)} />
      {label(104, 196, 'cabinet', a(1), '#fff')}
      <rect x={58} y={94} width={92} height={70} rx={14} fill="rgba(224,138,30,0.22)" stroke={color.lead} strokeWidth={4} strokeDasharray="12 9" opacity={bowlGoal} />
      {label(124, 250, 'bowl goes here', bowlGoal, color.lead, 720)}
      {/* bowl */}
      <circle cx={268} cy={168} r={36} fill="#8E929C" opacity={a(2)} />
      <circle cx={268} cy={168} r={22} fill="#B4B8C0" opacity={a(2)} />
      {label(268, 234, 'bowl', a(2))}
      {/* stove, and the plate's goal in front of it */}
      <rect x={420} y={84} width={136} height={124} rx={12} fill="#A3A8B0" opacity={a(3)} />
      <circle cx={488} cy={146} r={40} fill="none" stroke="#7D838C" strokeWidth={6} opacity={a(3)} />
      <circle cx={488} cy={146} r={20} fill="none" stroke="#7D838C" strokeWidth={6} opacity={a(3)} />
      {label(488, 76, 'stove', a(3))}
      <rect x={424} y={248} width={128} height={104} rx={18} fill="rgba(224,138,30,0.10)" stroke={color.lead} strokeWidth={4} strokeDasharray="12 9" opacity={plateGoal} />
      {label(478, 386, 'plate goes here', plateGoal, color.lead, 720)}
      {/* plate */}
      <circle cx={plate.x} cy={plate.y} r={50} fill="#fff" stroke="#D9A0A0" strokeWidth={8} opacity={a(4)} />
      <circle cx={plate.x} cy={plate.y} r={30} fill="none" stroke="#E7C2C2" strokeWidth={4} opacity={a(4)} />
      {label(plate.x, plate.y + 82, 'plate', a(4))}
      {/* the two skills' moves */}
      {push > 0 ? (
        <g>
          <line x1={px0} y1={py(px0)} x2={px1} y2={py(px1)} stroke={color.ink} strokeWidth={6} strokeLinecap="round" strokeDasharray="14 10" />
          <path d={`M${px1 - 16},${py(px1) - 12} L${px1},${py(px1)} L${px1 - 16},${py(px1) + 12}`} fill="none" stroke={color.ink} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" opacity={Math.min(1, push * 3)} />
        </g>
      ) : null}
      {place > 0 ? (
        <g opacity={Math.min(1, place * 2)}>
          <path d={arc} fill="none" stroke={color.ink} strokeWidth={6} strokeLinecap="round" strokeDasharray="14 10" />
          <path d="M146,90 L132,104 L151,110" fill="none" stroke={color.ink} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" opacity={place >= 0.9 ? 1 : 0} />
        </g>
      ) : null}
    </svg>
  );
};

/**
 * A label above a tracked object with a short leader line down to it: the cheese is named, not
 * ringed. Positions in display px inside the footage frame.
 */
const LeaderPin: React.FC<{
  toBox: (x: number, y: number) => {x: number; y: number};
  box: ObjBox | null;
  show: number;
  rise?: number;
  tone: 'dark' | 'bad';
  children: React.ReactNode;
}> = ({toBox, box, show, rise = 128, tone, children}) => {
  if (!box || !box.visible || show <= 0) return null;
  const p = toBox(box.cx, box.cy);
  const labelY = p.y - rise;
  const ink = tone === 'bad' ? color.bad : color.night;
  return (
    <>
      <svg width={1} height={1} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', opacity: Math.min(1, show)}}>
        <line x1={p.x} y1={labelY + 26} x2={p.x} y2={p.y - 24} stroke={ink} strokeWidth={4} strokeLinecap="round" />
      </svg>
      <div style={{position: 'absolute', left: p.x, top: labelY - 29, transform: 'translateX(-50%)', opacity: Math.min(1, show)}}>
        <Chip tone={tone} size={30}>
          {children}
        </Chip>
      </div>
    </>
  );
};

/** One of the robot's skills as a tile, lit when the voice names it. */
const SkillTile: React.FC<{text: string; show: number}> = ({text, show}) => (
  <div
    style={{
      opacity: 0.25 + 0.75 * show,
      transform: `translateY(${(1 - show) * 14}px)`,
      background: color.card,
      borderRadius: 18,
      boxShadow: shadow.card,
      padding: '16px 26px',
      fontSize: 30,
      fontWeight: 720,
      color: show > 0.5 ? color.ink : color.ink3,
      whiteSpace: 'nowrap',
    }}
  >
    {text}
  </div>
);

export const R2Fail: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const last = beat.frames - 1;
  const request = useQuote('r2_request');
  const firstSentence = (request.match(/^[^.]+\./) ?? [request])[0];

  const requestF = useCue('request', 0.02);
  const goalsF = useCue('goals', 0.09);
  const plateF = useCue('plate', 0.17);
  const bowlF = useCue('bowl', 0.24);
  const skillsF = useCue('skills', 0.31);
  const pushF = useCue('push', 0.39);
  const placeF = useCue('place', 0.44);
  const chainF = useCue('chain', 0.5);
  const deliversF = useCue('delivers', 0.54);
  const troubleF = useCue('trouble', 0.68);
  const cheeseF = useCue('cheese', 0.7);
  const wayF = useCue('way', 0.74);
  const failF = useCue('fail', 0.79);
  const grabF = useCue('grab', 0.84);
  const naive = useNumber('r2_naive_delivered');
  const examN = useNumber('r2_release_n');
  // the baseline: the plain chain on the reserved layouts, as recorded (layout order)
  const baseline = (film.mosaics['r2-chain'] ?? []).map((id) => (film.clips[id]?.delivered ? '1' : '0')).join('');
  const layoutOf = (id: string) => practiceStartOf(film, 'r2', film.clips[id]?.episode);
  const label = (id: string, above: boolean) => {
    const n = layoutOf(id);
    return `the plain chain · ${n === null ? 'a practice layout' : `practice layout ${n}`}${above ? ' · from above' : ''}`;
  };

  const track1 = useTrackFile(CLIPS.r2Fail1);
  const track5 = useTrackFile(CLIPS.r2Fail5);
  // the same two runs from above, when their framed overhead clips (with tracks) exist
  const top1 = clipOr(film, CLIPS.r2Fail1Top, CLIPS.r2Fail1, true);
  const top5 = clipOr(film, CLIPS.r2Fail5Top, CLIPS.r2Fail5, true);
  const trackTop1 = useTrackFile(top1);
  const trackTop5 = useTrackFile(top5);
  const clipTop1 = useClip(top1);
  const clipTop5 = useClip(top5);
  const contact1 = firstContact(track1, CHEESE, 100, 165);
  const contact5 = firstContact(track5, CHEESE, 100, 239);
  const clip1 = useClip(CLIPS.r2Fail1);
  const clip5 = useClip(CLIPS.r2Fail5);
  // practice layout 1 plays from "Chained plainly" up to its push into the cheese at the voice's
  // "runs into it"; at "grabs" the cut to practice layout 5, its hand closing on the cheese
  const keys1: Key[] = orderKeys([[chainF, 0], [failF, contact1], [grabF, contact1 + 30]], last);
  const keys5: Key[] = orderKeys([[grabF, contact5 - 10], [last, contact5 + 50]], last);
  const play1 = keyed(clip1, keys1);
  const play5 = keyed(clip5, keys5);
  const cut = frame >= grabF;
  const above1 = top1 !== CLIPS.r2Fail1 && frame >= troubleF - 4 && !cut;
  const above5 = top5 !== CLIPS.r2Fail5;

  // 1: the task
  const cardIn = useArrive(requestF);
  const goal1 = useArrive(plateF);
  const goal2 = useArrive(bowlF);
  const diagBuild = ramp(frame, goalsF - 20, goalsF + 30);
  const skillsIn = ramp(frame, skillsF, skillsF + 12);
  const push = ramp(frame, pushF, pushF + 22);
  const place = ramp(frame, placeF, placeF + 22);
  const diagOut = ramp(frame, chainF - 8, chainF + 8);
  // 2: the plain chain, recorded
  const footIn = useArrive(chainF - 2);
  const chainArrow = ramp(frame, chainF, chainF + 12);
  const dotsIn = ramp(frame, deliversF - 4, deliversF + 10);
  const nameLabel = ramp(frame, cheeseF + 8, cheeseF + 18) * (1 - ramp(frame, failF - 6, failF));
  const failLabel = ramp(frame, failF, failF + 8) * (cut ? 0 : 1);
  const grabLabel = ramp(frame, grabF + 4, grabF + 12);

  const overlay = (track: TrackFile | null, which: 1 | 5) => (ctx: {step: number; toBox: (x: number, y: number) => {x: number; y: number}; scale: number}) => {
    const box = objectBox(track, ctx.step, CHEESE);
    return which === 1 ? (
      <>
        <LeaderPin toBox={ctx.toBox} box={box} tone="dark" show={nameLabel}>
          cream cheese
        </LeaderPin>
        <LeaderPin toBox={ctx.toBox} box={box} tone="bad" show={failLabel}>
          the push runs into it
        </LeaderPin>
      </>
    ) : (
      <LeaderPin toBox={ctx.toBox} box={box} tone="bad" show={grabLabel}>
        the hand grabs the cheese
      </LeaderPin>
    );
  };

  return (
    <JobStage job={2} step={frame >= chainF ? 0 : null}>
      {/* the request and the robot's two skills */}
      <div style={{position: 'absolute', left: 120, top: 150, width: 610, display: 'flex', flexDirection: 'column', gap: 40}}>
        <div
          style={{
            background: color.card,
            borderRadius: 24,
            boxShadow: shadow.card,
            borderTop: `14px solid ${color.lead}`,
            padding: '24px 30px 26px',
            opacity: cardIn,
            transform: `translateY(${(1 - cardIn) * 24}px)`,
          }}
        >
          <Label tone={color.lead} style={{marginBottom: 12}}>
            new request · the shift lead
          </Label>
          <div style={{fontSize: 29, fontWeight: 600, color: color.ink, lineHeight: 1.3}}>“{firstSentence}”</div>
          <div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 12, marginTop: 20}}>
            <div style={{opacity: 0.2 + 0.8 * goal1, transform: `scale(${0.92 + 0.08 * goal1})`, transformOrigin: 'left center'}}>
              <Chip size={27} style={{background: goal1 > 0.5 ? color.lead : '#fff', color: goal1 > 0.5 ? '#fff' : color.ink3}}>
                1 · plate at the front of the stove
              </Chip>
            </div>
            <div style={{opacity: 0.2 + 0.8 * goal2, transform: `scale(${0.92 + 0.08 * goal2})`, transformOrigin: 'left center'}}>
              <Chip size={27} style={{background: goal2 > 0.5 ? color.lead : '#fff', color: goal2 > 0.5 ? '#fff' : color.ink3}}>
                2 · bowl on top of the cabinet
              </Chip>
            </div>
          </div>
        </div>
        <div style={{opacity: skillsIn, transform: `translateY(${(1 - skillsIn) * 18}px)`}}>
          <Label style={{marginBottom: 14}}>{frame >= chainF ? 'the plain chain' : 'the robot’s skills'}</Label>
          <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
            <SkillTile text="push the plate" show={ramp(frame, pushF - 4, pushF + 8)} />
            <div style={{fontSize: 40, color: color.ink2, opacity: chainArrow, width: 30}}>→</div>
            <SkillTile text="place the bowl" show={ramp(frame, placeF - 4, placeF + 8)} />
          </div>
        </div>
      </div>

      {/* the task, drawn */}
      {diagOut < 1 ? (
        <div style={{position: 'absolute', left: DIAG.x, top: DIAG.y, opacity: Math.min(1, diagBuild * 3) * (1 - diagOut), transform: `scale(${DIAG.scale})`, transformOrigin: 'left top'}}>
          <Label style={{marginBottom: 4, fontSize: 18}}>from above · an illustration</Label>
          <StationDiagram build={diagBuild} plateGoal={goal1} bowlGoal={goal2} push={push} place={place} />
        </div>
      ) : null}

      {/* the plain chain, recorded */}
      {footIn > 0.001 ? (
        <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y, opacity: Math.min(1, footIn), transform: `scale(${0.97 + 0.03 * Math.min(1, footIn)})`}}>
          {cut ? (
            <Footage
              clip={top5}
              segments={play5.segments}
              startAt={play5.startAt}
              width={FOOT.w}
              height={FOOT.h}
              zoom={above5 ? {from: 1.45, to: 1.6, at: [grabF, last], focus: focusOf(trackTop5, clipTop5, contact5)} : undefined}
              label={<Chip size={26}>{label(CLIPS.r2Fail5, above5)}</Chip>}
            >
              {overlay(trackTop5, 5)}
            </Footage>
          ) : above1 ? (
            <Footage
              clip={top1}
              segments={play1.segments}
              startAt={play1.startAt}
              width={FOOT.w}
              height={FOOT.h}
              zoom={{from: 1.35, to: 1.55, at: [troubleF, grabF], focus: focusOf(trackTop1, clipTop1, contact1)}}
              label={<Chip size={26}>{label(CLIPS.r2Fail1, true)}</Chip>}
            >
              {overlay(trackTop1, 1)}
            </Footage>
          ) : (
            <Footage clip={CLIPS.r2Fail1} segments={play1.segments} startAt={play1.startAt} width={FOOT.w} height={FOOT.h} label={<Chip size={26}>{label(CLIPS.r2Fail1, false)}</Chip>}>
              {overlay(track1, 1)}
            </Footage>
          )}
        </div>
      ) : null}

      {/* the baseline: the plain chain on the reserved layouts */}
      <div style={{position: 'absolute', left: FOOT.x, top: FOOT.y + FOOT.h + 30, opacity: dotsIn, transform: `translateY(${(1 - dotsIn) * 16}px)`}}>
        <Label style={{marginBottom: 12}}>the plain chain on the {examN} locked layouts</Label>
        <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
          <OutcomeDots outcomes={baseline} start={deliversF + 4} size={30} gap={9} stagger={2} />
          <span style={{fontSize: 44, fontWeight: 840, color: color.ink, fontVariantNumeric: 'tabular-nums', opacity: ramp(frame, deliversF + 40, deliversF + 48)}}>
            {naive} <span style={{fontSize: 30, fontWeight: 700, color: color.ink2}}>of {examN}</span>
          </span>
        </div>
        <DotLegend small style={{marginTop: 14}} />
      </div>
    </JobStage>
  );
};
