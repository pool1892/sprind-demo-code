// r1-experiment: one change at a time, in three movements.
//   1. The lab notebook: the programs the agent ran on the same practice layouts (dot i =
//      layout i). The plain chain first, then each change the voice names: open first, home
//      first, finish then home. The agent's other programs stay dimmed.
//   2. The same layout, two programs side by side: open then home loses the bowl, finish then
//      home keeps it (both runs on one time map, so the difference shows at the same moment).
//   3. The working program plays on, large, through to the open drawer, its recorded program
//      steps lighting in turn; then its count, as dots: "10 of 10 successful jobs".
import React from 'react';
import {Easing, interpolate, useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useBeat, useCue, useFilm, useNumber, type Program} from '../../data';
import {color, FPS, shadow} from '../../theme';
import {CLIPS, PROGRAM_STEPS} from './clips';
import {Chip, JobStage, OutcomeDots, Verdict, keyed, orderKeys, programStarts, ramp, stepOfKeys, useArrive, useTrackFile, type Key} from './common';
import {DotLegend} from './DotLegend';

/** The practice layout both comparison clips show (identity only). */
const COMPARE_START = 2;
const ROW = {x: 120, y: 262, h: 84, label: 620, dot: 34, gap: 11};

type Focus = 'focus' | 'named' | 'dim' | 'plain';

const NotebookRow: React.FC<{p: Program; y: number; arrive: number; focus: Focus; win?: boolean; start: number}> = ({p, y, arrive, focus, win, start}) => {
  const frame = useCurrentFrame();
  const counted = frame >= start + p.outcomes.length * 2 + 2;
  const lift = focus === 'focus' ? 1 : 0;
  const bg = focus === 'focus' ? (win ? 'rgba(18,162,108,0.14)' : color.card) : 'transparent';
  return (
    <div
      style={{
        position: 'absolute',
        left: ROW.x - 18,
        top: y,
        height: ROW.h - 8,
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        padding: '0 18px',
        borderRadius: 18,
        background: bg,
        boxShadow: lift ? (win ? `0 0 0 4px ${color.ok}, ${shadow.card}` : shadow.card) : 'none',
        opacity: arrive * (focus === 'dim' ? 0.3 : 1),
        transform: `translateX(${(1 - arrive) * -24}px)`,
      }}
    >
      <div style={{width: ROW.label, fontSize: 32, fontWeight: focus === 'focus' ? 760 : 560, color: color.ink, whiteSpace: 'nowrap'}}>{p.label}</div>
      <OutcomeDots outcomes={p.outcomes} start={start} size={ROW.dot} gap={ROW.gap} />
      <div style={{width: 64, textAlign: 'right', fontSize: 38, fontWeight: 800, fontVariantNumeric: 'tabular-nums', color: color.ink, opacity: counted ? 1 : 0}}>{p.k}</div>
    </div>
  );
};

/** One step of the working program, lit while the recorded run is in it. */
const StepRow: React.FC<{text: string; state: 'idle' | 'active' | 'done'; arrive: number}> = ({text, state, arrive}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        height: 66,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        padding: '0 22px 0 14px',
        borderRadius: 16,
        background: color.card,
        boxShadow: state === 'active' ? `0 0 0 4px ${color.agent}, ${shadow.card}` : shadow.card,
        opacity: arrive * (state === 'idle' ? 0.6 : 1),
        transform: `translateX(${(1 - arrive) * 24}px)`,
      }}
    >
      {state === 'done' ? (
        <Verdict ok size={36} style={{boxShadow: 'none'}} />
      ) : (
        <div style={{width: 36, height: 36, borderRadius: '50%', flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: `inset 0 0 0 4px ${state === 'active' ? color.agent : color.line}`}}>
          {state === 'active' ? <div style={{width: 14, height: 14, borderRadius: '50%', background: color.agent, opacity: 0.55 + 0.45 * Math.sin(frame / 4)}} /> : null}
        </div>
      )}
      <div style={{fontSize: 30, fontWeight: state === 'active' ? 760 : 600, color: color.ink, whiteSpace: 'nowrap'}}>{text}</div>
    </div>
  );
};

export const R1Experiment: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const last = beat.frames - 1;
  const programs = (film.programs.r1 ?? []).filter((p) => p.role !== 'confirm');
  const nLayouts = useNumber('r1_plain_n');
  const winK = useNumber('r1_finish_home_k');
  const winN = useNumber('r1_finish_home_n');

  const sameF = useCue('same', 0.12);
  const openF = useCue('open', 0.22);
  const dragF = useCue('drag', 0.29);
  const homeF = useCue('home', 0.37);
  const snagF = useCue('snag', 0.43);
  const worksF = useCue('works', 0.51);
  const finishF = useCue('finish', 0.57);
  const liftF = useCue('lift', 0.63);
  const drawerF = useCue('drawer', 0.77);
  // the count lands as the voice says it ("... then open the drawer: 10 of 10 successful
  // jobs"): the number is spoken about two thirds of a second before the cue's words
  const voiceEndF = Math.round((beat.voice_offset + beat.voice_seconds) * FPS);
  const countF = Math.min(last - 30, useCue('count', (voiceEndF - 30) / Math.max(1, beat.frames)) - 22);

  // the rows the voice names, by their place in the notebook
  const idx = (pred: (p: Program) => boolean) => programs.findIndex(pred);
  const plainIdx = idx((p) => p.role === 'plain');
  const openIdx = idx((p) => p.name === 'bowl-open20-drawer');
  const homeIdx = idx((p) => p.name === 'bowl-home-drawer');
  const winIdx = idx((p) => p.role === 'winner');
  const openHomeIdx = idx((p) => p.name === 'bowl-open20-home-drawer');
  const named: {idx: number; at: number; noteAt: number; note: string}[] = [
    {idx: openIdx, at: openF, noteAt: dragF, note: 'they still catch the bowl'},
    {idx: homeIdx, at: homeF, noteAt: snagF, note: 'the way home snags it'},
    {idx: winIdx, at: worksF, noteAt: worksF + 10, note: 'lets go, lifts, then home'},
  ];
  const isNamed = (i: number) => i === plainIdx || named.some((n) => n.idx === i);
  const current = [...named].reverse().find((n) => frame >= n.at) ?? null;
  const focusOf = (i: number): Focus => {
    if (!isNamed(i)) return 'dim';
    if (current && current.idx === i) return 'focus';
    if (i === plainIdx || named.some((n) => n.idx === i && frame >= n.at)) return 'plain';
    return 'dim';
  };
  const rowAt = (i: number): number => {
    if (i === plainIdx) return 10;
    const n = named.find((x) => x.idx === i);
    return n ? n.at - 6 : sameF + 4 + i * 3;
  };

  // movements 2 and 3: both comparison runs on one time map of control steps
  const track = useTrackFile(CLIPS.r1FinishHome);
  const steps = programStarts(track, PROGRAM_STEPS.r1FinishHome); // bowl, finish, home, drawer
  const clipA = useClip(CLIPS.r1OpenHome);
  const clipB = useClip(CLIPS.r1FinishHome);
  const endStep = clipB ? (clipB.frames - 1) * clipB.frame_every : 282;
  const keys: Key[] = orderKeys(
    [
      [finishF - 10, steps[1] - 25],
      [finishF + 15, steps[1]],
      [liftF + 10, steps[2] - 3],
      [drawerF - 5, steps[3]],
      [Math.max(drawerF + 60, countF - 10), endStep],
    ],
    last,
  );
  const playA = keyed(clipA, keys);
  const playB = keyed(clipB, keys);
  const step = stepOfKeys(frame, keys);
  const pA = programs[openHomeIdx];
  const pB = programs[winIdx];
  const outcomeAt = (p: Program | undefined) => p?.outcomes[p.starts.indexOf(COMPARE_START)] === '1';

  const headIn = useArrive(0);
  const sameIn = ramp(frame, sameF - 6, sameF + 8);
  const notebookOut = ramp(frame, finishF - 12, finishF);
  const sideIn = ramp(frame, finishF - 10, finishF + 4);
  const verdictIn = useArrive(drawerF - 16, 'lively');
  const grow = interpolate(frame, [drawerF + 4, drawerF + 28], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic)});
  const aOut = ramp(frame, drawerF, drawerF + 14);
  const stepsIn = (i: number) => ramp(frame, drawerF + 18 + i * 4, drawerF + 30 + i * 4);
  const countIn = useArrive(countF, 'lively');
  const finished = step >= endStep - 1;

  const dotsX = ROW.x + ROW.label + 24 + 18;
  const dotsW = 10 * ROW.dot + 9 * ROW.gap;
  const CW = 820;
  const CH = 600;
  const top = 214;
  // the working run's box: half size beside the other, then large
  const bx = interpolate(grow, [0, 1], [ROW.x + CW + 40, ROW.x]);
  const bw = interpolate(grow, [0, 1], [CW, 1120]);
  const bh = interpolate(grow, [0, 1], [CH, Math.round((1120 * 9) / 16)]);
  const stepLabels = ['put the bowl on the cabinet', 'let the placement finish', 'home', 'open the middle drawer'];
  const activeStep = steps.reduce((acc, s, i) => (step >= s ? i : acc), 0);

  return (
    <JobStage job={1} step={2}>
      {/* 1: the lab notebook */}
      {notebookOut < 1 ? (
        <div style={{position: 'absolute', inset: 0, opacity: 1 - notebookOut}}>
          <div style={{position: 'absolute', left: ROW.x, top: 136, opacity: headIn, transform: `translateY(${(1 - headIn) * 16}px)`}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>One change at a time</div>
          </div>
          <div
            style={{
              position: 'absolute',
              left: dotsX - 10,
              top: 214,
              width: dotsW + 20,
              textAlign: 'center',
              fontSize: 26,
              fontWeight: 700,
              color: sameIn > 0.5 ? color.ink : color.ink3,
              borderBottom: `3px solid ${sameIn > 0.5 ? color.ink : color.line}`,
              paddingBottom: 6,
            }}
          >
            the same {nLayouts} practice layouts
          </div>
          {programs.map((p, i) => {
            const at = rowAt(i);
            return <NotebookRow key={p.job} p={p} y={ROW.y + i * ROW.h} arrive={ramp(frame, at - 6, at + 4)} focus={focusOf(i)} win={p.role === 'winner'} start={at} />;
          })}
          {named.map((n) => {
            const show = ramp(frame, n.noteAt - 6, n.noteAt + 6);
            const win = n.idx === winIdx;
            return (
              <div key={n.idx} style={{position: 'absolute', left: dotsX + dotsW + 112, top: ROW.y + n.idx * ROW.h + 6, opacity: show, transform: `translateX(${(1 - show) * 20}px)`}}>
                <Chip tone={win ? 'ok' : 'bad'} size={27}>
                  {n.note}
                </Chip>
              </div>
            );
          })}
          <div style={{position: 'absolute', left: ROW.x, top: ROW.y + programs.length * ROW.h + 18, opacity: ramp(frame, 14, 26)}}>
            <DotLegend small />
          </div>
        </div>
      ) : null}

      {/* 2 and 3: the same layout, two programs; then the working one, large */}
      {sideIn > 0 ? (
        <div style={{position: 'absolute', inset: 0, opacity: sideIn}}>
          <div style={{position: 'absolute', left: ROW.x, top: 136, opacity: 1 - grow}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>The same layout, two programs</div>
          </div>
          <div style={{position: 'absolute', left: ROW.x, top: 136, opacity: grow}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>The working program</div>
          </div>
          {aOut < 1 ? (
            <div style={{position: 'absolute', left: ROW.x, top, width: CW, opacity: 1 - aOut}}>
              <div style={{borderRadius: 24, boxShadow: frame >= drawerF - 16 ? `0 0 0 6px ${outcomeAt(pA) ? color.ok : color.bad}, ${shadow.frame}` : shadow.frame}}>
                <Footage clip={CLIPS.r1OpenHome} segments={playA.segments} startAt={playA.startAt} width={CW} height={CH} radius={20} focus={{x: 0.42, y: 0.5}} label={<Chip size={27}>{pA?.label ?? ''}</Chip>}>
                  {() => (
                    <div style={{position: 'absolute', right: 22, bottom: 22, opacity: verdictIn, transform: `scale(${Math.min(1.1, 0.8 + 0.2 * verdictIn)})`}}>
                      <Chip ok={outcomeAt(pA)} size={28}>
                        bowl dragged off
                      </Chip>
                    </div>
                  )}
                </Footage>
              </div>
            </div>
          ) : null}
          <div style={{position: 'absolute', left: bx, top, width: bw}}>
            <div style={{borderRadius: 24, boxShadow: frame >= drawerF - 16 ? `0 0 0 6px ${outcomeAt(pB) ? color.ok : color.bad}, ${shadow.frame}` : shadow.frame}}>
              <Footage clip={CLIPS.r1FinishHome} segments={playB.segments} startAt={playB.startAt} width={bw} height={bh} radius={20} focus={{x: 0.42, y: 0.5}} label={<Chip size={27}>{pB?.label ?? ''}</Chip>}>
                {() => (
                  <div style={{position: 'absolute', right: 22, bottom: 22, opacity: grow > 0 ? (finished ? 1 : 0) : verdictIn, transform: `scale(${Math.min(1.1, 0.8 + 0.2 * verdictIn)})`}}>
                    <Chip ok={outcomeAt(pB)} size={28}>
                      {grow > 0 ? 'delivered' : 'bowl stays'}
                    </Chip>
                  </div>
                )}
              </Footage>
            </div>
          </div>
          {/* movement 3: the program, step by step, and its count */}
          {grow > 0 ? (
            <div style={{position: 'absolute', left: ROW.x + 1120 + 50, top, width: 1800 - (ROW.x + 1120 + 50), display: 'flex', flexDirection: 'column', gap: 14}}>
              {stepLabels.map((t, i) => (
                <StepRow key={i} text={t} state={finished || i < activeStep ? 'done' : i === activeStep ? 'active' : 'idle'} arrive={stepsIn(i)} />
              ))}
            </div>
          ) : null}
          <div style={{position: 'absolute', left: ROW.x, top: top + Math.round((1120 * 9) / 16) + 34, display: 'flex', alignItems: 'center', gap: 26, opacity: Math.min(1, countIn)}}>
            <OutcomeDots outcomes={pB?.outcomes ?? ''} start={countF} size={34} gap={11} />
            <span style={{fontSize: 56, fontWeight: 840, color: color.ink, fontVariantNumeric: 'tabular-nums', transform: `scale(${0.8 + 0.2 * Math.min(1.1, countIn)})`, display: 'inline-block'}}>
              {winK} of {winN} <span style={{fontSize: 40, fontWeight: 700, color: color.ink2}}>successful jobs</span>
            </span>
            <DotLegend small style={{marginLeft: 18}} />
          </div>
        </div>
      ) : null}
    </JobStage>
  );
};
