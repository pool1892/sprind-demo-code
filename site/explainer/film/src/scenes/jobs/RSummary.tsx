// r-summary: what experiment one showed, and why it matters. Calm: two job
// cards side by side, each with its request in a few words and its final exam before and after
// as two dot rows (the plain chain, then the agent's job; every dot a recorded run of the exam,
// the counts from the bound record). The plain chain's rows arrive with "Twice, a new job
// failed"; under both cards three phrases light with the voice (found the cause · changed the
// job · proved it on unseen layouts), and the agent's rows fill on "proved". Then "no person in
// the loop", the line on why it matters, and each card's time from request to proven job as the
// voice says it.
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Stage} from '../../components/basics';
import {useCue, useFilm, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {ExperimentTag} from '../opening/ExperimentCard';
import {Label, OutcomeDots, ramp, useArrive} from './common';
import {DotLegend} from './DotLegend';

const CARD = {y: 226, w: 800, gap: 80, pad: 36};
const DOT = {size: 22, gap: 6};

const Row: React.FC<{name: string; outcomes: string; start: number; count: string; of: string; show: number; strong?: boolean}> = ({name, outcomes, start, count, of, show, strong}) => {
  const frame = useCurrentFrame();
  return (
  <div style={{opacity: show, transform: `translateY(${(1 - show) * 12}px)`}}>
    <div style={{fontSize: 26, fontWeight: 700, color: strong ? color.ink : color.ink2, marginBottom: 10}}>{name}</div>
    <div style={{display: 'flex', alignItems: 'center', gap: 22}}>
      <OutcomeDots outcomes={outcomes} start={start} size={DOT.size} gap={DOT.gap} stagger={1} />
      <span style={{fontSize: 40, fontWeight: 840, color: color.ink, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap', opacity: ramp(frame, start + outcomes.length, start + outcomes.length + 8)}}>
        {count} <span style={{fontSize: 28, fontWeight: 700, color: color.ink2}}>of {of}</span>
      </span>
    </div>
  </div>
  );
};

const JobCard: React.FC<{
  x: number;
  kicker: string;
  request: string;
  chain: string;
  job: string;
  naive: string;
  delivered: string;
  n: string;
  minutes: string;
  arrive: number;
  chainAt: number;
  jobAt: number;
  minutesAt: number;
}> = ({x, kicker, request, chain, job, naive, delivered, n, minutes, arrive, chainAt, jobAt, minutesAt}) => {
  const frame = useCurrentFrame();
  const jobIn = ramp(frame, jobAt - 6, jobAt + 6);
  const timeIn = useArrive(minutesAt, 'lively');
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: CARD.y,
        width: CARD.w,
        boxSizing: 'border-box',
        padding: `28px ${CARD.pad}px 30px`,
        borderRadius: 26,
        background: color.card,
        boxShadow: shadow.card,
        borderTop: `14px solid ${color.lead}`,
        opacity: arrive,
        transform: `translateY(${(1 - arrive) * 24}px)`,
      }}
    >
      <Label tone={color.lead}>{kicker}</Label>
      <div style={{fontSize: 38, fontWeight: 780, letterSpacing: '-0.01em', color: color.ink, marginTop: 8, marginBottom: 18}}>{request}</div>
      <div style={{fontSize: 24, fontWeight: 650, color: color.ink3, marginBottom: 14}}>the final exam · {n} locked layouts</div>
      <div style={{display: 'flex', flexDirection: 'column', gap: 22}}>
        <Row name="the plain chain" outcomes={chain} start={chainAt} count={naive} of={n} show={ramp(frame, chainAt - 8, chainAt + 4)} />
        <Row name="the agent’s job" outcomes={job} start={jobAt} count={delivered} of={n} show={0.25 + 0.75 * jobIn} strong />
      </div>
      <div
        style={{
          marginTop: 26,
          height: 52,
          display: 'flex',
          alignItems: 'baseline',
          gap: 14,
          opacity: Math.min(1, timeIn),
          transform: `translateY(${(1 - Math.min(1, timeIn)) * 10}px)`,
        }}
      >
        <span style={{fontSize: 26, fontWeight: 650, color: color.ink2}}>request to proven job:</span>
        <span style={{fontSize: 44, fontWeight: 840, color: color.agent, fontVariantNumeric: 'tabular-nums'}}>{minutes}</span>
        <span style={{fontSize: 30, fontWeight: 700, color: color.agent}}>minutes</span>
      </div>
    </div>
  );
};

export const RSummary: React.FC = () => {
  const frame = useCurrentFrame();
  const film = useFilm();
  const outcomes = (mosaic: string) => (film.mosaics[mosaic] ?? []).map((id) => (film.clips[id]?.delivered ? '1' : '0')).join('');

  const questionF = useCue('question', 0.02);
  const failedF = useCue('failed', 0.1);
  const twiceF = useCue('twice', 0.24);
  const causeF = useCue('cause', 0.28);
  const changedF = useCue('changed', 0.33);
  const provedF = useCue('proved', 0.4);
  const nobodyF = useCue('nobody', 0.49);
  const mattersF = useCue('matters', 0.56);
  const minutesF = useCue('minutes', 0.76);

  const n1 = useNumber('r1_release_n');
  const n2 = useNumber('r2_release_n');

  const titleIn = useArrive(questionF);
  const cardsIn = useArrive(failedF);
  // the three things the agent did, each lit as the voice names it
  const phrases: {text: string; at: number}[] = [
    {text: 'found the cause', at: causeF},
    {text: 'changed the job', at: changedF},
    {text: 'proved it on unseen layouts', at: provedF},
  ];
  const phrasesIn = ramp(frame, twiceF - 6, twiceF + 8);
  const nobodyIn = useArrive(nobodyF);
  const mattersIn = ramp(frame, mattersF, mattersF + 14);
  // the two times land as the voice speaks them, after "Here, an agent did it, in"
  const time1 = minutesF + 40;
  const time2 = minutesF + 60;

  return (
    <Stage>
      <div style={{position: 'absolute', left: 120, top: 46}}>
        <ExperimentTag which={1} />
      </div>
      <div style={{position: 'absolute', left: 120, top: 128, opacity: titleIn, transform: `translateY(${(1 - titleIn) * 16}px)`}}>
        <div style={{fontSize: 56, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>What the first experiment showed</div>
      </div>

      <JobCard
        x={120}
        kicker="first job"
        request="bowl on the cabinet, drawer open"
        chain={outcomes('r1-chain')}
        job={outcomes('r1-job')}
        naive={useNumber('r1_naive_delivered')}
        delivered={useNumber('r1_delivered')}
        n={n1}
        minutes={useNumber('r1_wall_minutes')}
        arrive={cardsIn}
        chainAt={failedF + 8}
        jobAt={provedF + 6}
        minutesAt={time1}
      />
      <JobCard
        x={120 + CARD.w + CARD.gap}
        kicker="second job"
        request="plate at the stove, bowl on the cabinet"
        chain={outcomes('r2-chain')}
        job={outcomes('r2-job')}
        naive={useNumber('r2_naive_delivered')}
        delivered={useNumber('r2_delivered')}
        n={n2}
        minutes={useNumber('r2_wall_minutes')}
        arrive={ramp(frame, failedF + 6, failedF + 22)}
        chainAt={failedF + 20}
        jobAt={provedF + 14}
        minutesAt={time2}
      />

      {/* what the agent did, both times */}
      <div style={{position: 'absolute', left: 120, right: 120, top: 790, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 28, opacity: phrasesIn}}>
        {phrases.map((p, i) => {
          const on = ramp(frame, p.at - 2, p.at + 8);
          return (
            <React.Fragment key={p.text}>
              {i > 0 ? <span style={{fontSize: 40, color: color.ink3}}>·</span> : null}
              <span style={{fontSize: 44, fontWeight: 780, letterSpacing: '-0.01em', color: on > 0.5 ? color.agent : color.ink3, opacity: 0.45 + 0.55 * on, whiteSpace: 'nowrap'}}>{p.text}</span>
            </React.Fragment>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 868, display: 'flex', justifyContent: 'center', opacity: Math.min(1, nobodyIn), transform: `translateY(${(1 - Math.min(1, nobodyIn)) * 12}px)`}}>
        <div style={{fontSize: 34, fontWeight: 700, color: color.ink}}>with no person in the loop</div>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 934, display: 'flex', justifyContent: 'center', opacity: mattersIn}}>
        <div style={{fontSize: 30, fontWeight: 560, color: color.ink2}}>
          <b style={{color: color.ink, fontWeight: 760}}>Why it matters:</b> this is the engineering that makes every new job slow and expensive today
        </div>
      </div>
      <div style={{position: 'absolute', right: 120, top: 1010, opacity: ramp(frame, failedF + 8, failedF + 20)}}>
        <DotLegend small />
      </div>
    </Stage>
  );
};
