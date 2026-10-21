// r1-prove: two results, kept apart in time, with room for each to sink in.
// First the confirmation, in a grey practice frame: the saved program on fresh practice layouts
// the agent had never used, its dots, then its count as the voice says it; it holds still for a
// breath. Then the dark vault rises, unlocks and opens on the final exam: the reserved layouts,
// locked away until now, play at once and come to rest; the plain chain's verdicts land as the
// voice names it (one green); a beat of stillness; then the agent's tiles turn, all green, as
// the voice gives its count, and the finished panel holds to the end of the beat. Rescued and
// broken small, after the count.
import React from 'react';
import {Easing, interpolate, useCurrentFrame} from 'remotion';
import {useBeat, useCue, useFilm, useNumber} from '../../data';
import {color, shadow} from '../../theme';
import {JobStage, Label, OutcomeDots, ramp, useArrive} from './common';
import {DotLegend} from './DotLegend';
import {EXAM, ExamPanel, VERDICT_RIPPLE, Vault, mosaicH, mosaicW} from './exam';

const TITLE = {x: EXAM.x, y: 116};
const PRACTICE_GREY = '#AEB2BB';

export const R1Prove: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const last = beat.frames - 1;
  const confirm = (film.programs.r1 ?? []).find((p) => p.role === 'confirm');
  const k = useNumber('r1_confirm_k');
  const n = useNumber('r1_confirm_n');
  const examN = useNumber('r1_release_n');

  const confirmsF = useCue('confirms', 0.08);
  const neverF = useCue('never', 0.23);
  const examF = useCue('exam', 0.37);
  const chainF = useCue('chain', 0.54);
  const jobF = useCue('job', 0.67);

  // 1: the confirmation on fresh practice layouts; its count as the voice says it, then a still
  // breath before the exam
  const practiceIn = useArrive(0);
  const dotsAt = confirmsF + 4;
  // the count between "never used" and the exam: the voice speaks it in the pause between them
  const countAt = Math.round((neverF + examF) / 2) - 4;
  const countIn = useArrive(countAt, 'lively');
  const practiceOut = ramp(frame, examF - 10, examF + 4, Easing.in(Easing.cubic));
  // 2: the vault rises, unlocks, opens; the tiles play
  const rise = interpolate(frame, [examF - 2, examF + 16], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const open = ramp(frame, examF + 18, examF + 48);
  const appearAt = examF + 32;
  const chainStart = examF + 32;
  const jobStart = examF + 38;
  // both sides play their runs while the voice opens the exam and come to rest before the plain
  // chain's verdicts; the agent's verdicts wait for the pause after "The agent's job:", so its
  // count lands with the spoken number
  const playEnd = chainF - 6;
  const jobVerdictAt = jobF + 24;
  const examTitleIn = ramp(frame, examF + 2, examF + 14);
  const W = 2 * mosaicW() + EXAM.between;

  return (
    <JobStage job={1} step={3}>
      {/* 1: the saved program on fresh practice layouts it had never used */}
      {practiceOut < 1 ? (
        <>
          <div style={{position: 'absolute', left: TITLE.x, top: TITLE.y, opacity: practiceIn * (1 - practiceOut), transform: `translateY(${-practiceOut * 40}px)`}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>
              {n} fresh practice layouts <span style={{color: color.ink3, fontWeight: 650, fontSize: 40}}>· never used before</span>
            </div>
          </div>
          <div
            style={{
              position: 'absolute',
              left: EXAM.x,
              top: EXAM.top,
              width: W,
              height: mosaicH(),
              borderRadius: 28,
              background: color.card,
              boxShadow: `0 0 0 8px ${PRACTICE_GREY}, ${shadow.card}`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 36,
              opacity: practiceIn * (1 - practiceOut),
              transform: `translateY(${-practiceOut * 120}px) scale(${1 - 0.06 * practiceOut})`,
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', gap: 16}}>
              <div style={{background: PRACTICE_GREY, color: '#fff', borderRadius: 999, padding: '6px 18px', fontSize: 24, fontWeight: 760, letterSpacing: '0.12em', textTransform: 'uppercase'}}>practice</div>
              <Label>the saved program · a confirmation run</Label>
            </div>
            <OutcomeDots outcomes={confirm?.outcomes ?? ''} start={dotsAt} size={56} gap={16} stagger={3} />
            <div style={{display: 'flex', alignItems: 'baseline', gap: 24}}>
              <span style={{fontSize: 112, fontWeight: 840, letterSpacing: '-0.035em', color: color.ink, opacity: Math.min(1, countIn), transform: `scale(${0.8 + 0.2 * Math.min(1.1, countIn)})`, display: 'inline-block', fontVariantNumeric: 'tabular-nums'}}>
                {k} of {n}
              </span>
              <span style={{fontSize: 44, fontWeight: 650, color: color.ink2, opacity: Math.min(1, countIn)}}>delivered</span>
            </div>
            <DotLegend small style={{opacity: ramp(frame, dotsAt, dotsAt + 12)}} />
          </div>
        </>
      ) : null}

      {/* 2: the final exam */}
      <div style={{position: 'absolute', left: TITLE.x, top: TITLE.y, opacity: examTitleIn, transform: `translateY(${(1 - examTitleIn) * 24}px)`}}>
        <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>
          The final exam <span style={{color: color.ink3, fontWeight: 650, fontSize: 40}}>· {examN} reserved layouts, locked away until now</span>
        </div>
      </div>
      {frame >= appearAt - 2 ? (
        <ExamPanel
          job="r1"
          appearAt={appearAt}
          chainPlay={[chainStart, Math.max(30, playEnd - chainStart)]}
          jobPlay={[jobStart, Math.max(30, playEnd - jobStart)]}
          chainVerdictAt={chainF}
          jobVerdictAt={jobVerdictAt}
          footerAt={Math.min(last - 20, jobVerdictAt + VERDICT_RIPPLE + 45)}
          dim="job"
        />
      ) : null}
      {rise > 0 ? (
        <div style={{position: 'absolute', left: EXAM.x, top: EXAM.top, width: W, height: mosaicH(), opacity: rise, transform: `translateY(${(1 - rise) * 160}px)`}}>
          <Vault open={open} show={1} label={`${examN} reserved layouts · locked away`} width={W} height={mosaicH()} />
        </div>
      ) : null}
    </JobStage>
  );
};
