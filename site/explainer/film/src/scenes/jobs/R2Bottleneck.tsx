// r2-bottleneck: reuse, then a moving bottleneck.
//   1. Job 1's check comes off the installation's library shelf and grows a second row: the
//      extended check of this job, whose recorded purpose says it extends the library's
//      hand-off check.
//   2. The agent's programs on the same practice layouts stream in; the cheese-loading row is
//      named; a two-stage pipeline shows the plate fixed and the bowl failing, then the marker
//      sliding from plate to bowl; the loaded run carries the bowl. No quotation.
import React from 'react';
import {Easing, interpolate, useCurrentFrame} from 'remotion';
import {Footage, useClip} from '../../components/footage';
import {useBeat, useCue, useFilm, useNumber, type Program} from '../../data';
import {color, font, shadow} from '../../theme';
import {CLIPS, PROGRAM_STEPS} from './clips';
import {Chip, JobStage, Label, MonoName, OutcomeDots, Verdict, keyed, orderKeys, programStarts, ramp, useArrive, useTrackFile, type Key} from './common';
import {DotLegend} from './DotLegend';

const NB = {x: 120, y: 204, h: 32, label: 510, dot: 22, gap: 6};
const RIGHT = {x: 1080, w: 700};

const Stage2: React.FC<{name: string; state: 'idle' | 'ok' | 'bad'; show: number}> = ({name, state, show}) => (
  <div
    style={{
      width: 236,
      height: 92,
      borderRadius: 20,
      background: color.card,
      boxShadow: state === 'bad' ? `0 0 0 4px ${color.bad}, ${shadow.card}` : state === 'ok' ? `0 0 0 4px ${color.ok}, ${shadow.card}` : shadow.card,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 14,
      opacity: show,
      fontSize: 32,
      fontWeight: 760,
      color: color.ink,
    }}
  >
    {state === 'idle' ? null : <Verdict ok={state === 'ok'} size={38} style={{boxShadow: 'none'}} />}
    {name}
  </div>
);

/** The library's check card, growing a second row: the check this job extended from it. */
const LibraryCheck: React.FC<{arrive: number; grow: number; first: string; firstQ: string; second: string; secondQ: string}> = ({arrive, grow, first, firstQ, second, secondQ}) => (
  <div
    style={{
      width: 1180,
      background: color.card,
      borderRadius: 28,
      boxShadow: shadow.card,
      borderTop: `14px solid ${color.teal}`,
      padding: '30px 44px 34px',
      opacity: Math.min(1, arrive * 1.3),
      transform: `translateX(${(1 - Math.min(1, arrive)) * -160}px)`,
    }}
  >
    <Label tone={color.teal} style={{marginBottom: 14}}>
      check · from the installation's library · saved in the first job
    </Label>
    <MonoName size={40}>{first}</MonoName>
    <div style={{fontSize: 32, fontWeight: 560, color: color.ink2, marginTop: 10, lineHeight: 1.3}}>{firstQ}</div>
    <div style={{overflow: 'hidden', maxHeight: 260 * grow, opacity: grow}}>
      <div style={{marginTop: 26, paddingTop: 24, borderTop: `2px solid ${color.line}`}}>
        <Label tone={color.teal} style={{marginBottom: 12}}>
          + extended for this job
        </Label>
        <MonoName size={40}>{second}</MonoName>
        <div style={{fontSize: 32, fontWeight: 560, color: color.ink2, marginTop: 10, lineHeight: 1.3}}>{secondQ}</div>
      </div>
    </div>
  </div>
);

export const R2Bottleneck: React.FC = () => {
  const frame = useCurrentFrame();
  const beat = useBeat();
  const film = useFilm();
  const last = beat.frames - 1;
  const programs = (film.programs.r2 ?? []).filter((p) => p.role !== 'confirm');
  const nPrograms = useNumber('r2_programs');
  const nLayouts = useNumber('r2_plain_n');
  const loadedK = useNumber('r2_loaded_k');
  const loadedN = useNumber('r2_loaded_n');
  const checkOne = useNumber('r1_check_name');
  const checkTwo = useNumber('r2_check_name');
  const purposeTwo = useNumber('r2_check_purpose');
  // the extended check's recorded purpose ("Extends the library handoff check with both goals'
  // first losses, including losses before handoff and in the final tail"), in plain words
  const extends_ = /both goals/.test(purposeTwo) ? 'Now watches both goals over the whole job.' : purposeTwo;

  const libraryF = useCue('library', 0.03);
  const programsF = useCue('programs', 0.17);
  const loadedF = useCue('loaded', 0.29);
  const pushF = useCue('push', 0.43);
  const movedF = useCue('moved', 0.53);
  const deliversF = useCue('delivers', 0.65);
  const bottleneckF = useCue('bottleneck', 0.78);

  // 1: the library check, and its extension
  const cardIn = useArrive(libraryF);
  const grow = interpolate(frame, [libraryF + 16, libraryF + 36], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.cubic)});
  const cardOut = ramp(frame, programsF + 24, programsF + 36);
  const nbFrom = programsF + 34;

  // 2: the loaded run's bowl step, from its recorded program
  const track = useTrackFile(CLIPS.r2Loaded);
  const starts = programStarts(track, PROGRAM_STEPS.r2Loaded);
  const bowlFrom = starts[4] ?? starts[starts.length - 1];
  const clip = useClip(CLIPS.r2Loaded);
  const clipLast = clip ? (clip.frames - 1) * clip.frame_every : bowlFrom + 300;
  const keys: Key[] = orderKeys([[movedF - 8, bowlFrom - 20], [last, Math.min(clipLast, bowlFrom + 300)]], last);
  const play = keyed(clip, keys);

  const rowEvery = 3;
  const isLoaded = (p: Program) => p.role === 'loaded';
  const loadedIn = ramp(frame, loadedF - 4, loadedF + 10);
  const nbDim = ramp(frame, movedF, movedF + 16) * 0.4;
  const pipeIn = useArrive(loadedF + 6);
  const plateOk = frame >= pushF;
  const bowlBad = frame >= movedF + 10;
  const countIn = useArrive(deliversF + 6, 'lively');
  const slide = ramp(frame, bottleneckF - 6, bottleneckF + 16);
  const clipIn = useArrive(movedF - 10);
  const headIn = useArrive(nbFrom - 6);

  // stage positions in the pipeline (row 1: the plain chain; row 2: cheese loaded first)
  const P = {x0: RIGHT.x + 0, x1: RIGHT.x + 330, row1: 210, row2: 420};
  const marker = {
    x: P.x0 + (P.x1 - P.x0) * slide,
    y: P.row1 + 104 + (P.row2 - P.row1) * slide,
  };

  return (
    <JobStage job={2} step={2}>
      {/* 1: starting from the check it saved in the first job */}
      {cardOut < 1 ? (
        <div style={{position: 'absolute', left: 370, top: 290, opacity: 1 - cardOut, transform: `translateY(${-cardOut * 40}px)`}}>
          <LibraryCheck arrive={cardIn} grow={grow} first={checkOne} firstQ="Does the bowl stay put while the next skill runs?" second={checkTwo} secondQ={extends_} />
        </div>
      ) : null}

      {/* 2: the programs */}
      {frame >= nbFrom - 8 ? (
        <>
          <div style={{position: 'absolute', left: NB.x, top: 136, opacity: headIn, transform: `translateY(${(1 - headIn) * 16}px)`}}>
            <div style={{fontSize: 50, fontWeight: 800, letterSpacing: '-0.02em', color: color.ink}}>
              {nPrograms} programs <span style={{fontSize: 34, fontWeight: 600, color: color.ink3}}>· the same {nLayouts} practice layouts</span>
            </div>
          </div>
          <div style={{position: 'absolute', left: 0, top: 0, opacity: 1 - nbDim}}>
            {programs.map((p, i) => {
              const at = nbFrom + i * rowEvery;
              const shown = ramp(frame, at - 4, at + 2);
              const hl = isLoaded(p) ? loadedIn : 0;
              return (
                <div
                  key={p.job}
                  style={{
                    position: 'absolute',
                    left: NB.x - 12,
                    top: NB.y + 24 + i * NB.h,
                    height: NB.h - 3,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 16,
                    padding: '0 12px',
                    borderRadius: 10,
                    background: hl ? color.card : 'transparent',
                    boxShadow: hl ? `0 0 0 ${3 * hl}px ${color.ink}, ${shadow.card}` : 'none',
                    opacity: shown,
                    transform: `translateX(${(1 - shown) * -16}px)`,
                  }}
                >
                  <div style={{width: NB.label, fontSize: 24, fontWeight: isLoaded(p) && hl ? 760 : 520, color: color.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{p.label}</div>
                  <OutcomeDots outcomes={p.outcomes} start={at} size={NB.dot} gap={NB.gap} stagger={1} />
                  <div style={{width: 34, textAlign: 'right', fontSize: 26, fontWeight: 800, color: color.ink, fontVariantNumeric: 'tabular-nums', opacity: frame >= at + 12 ? 1 : 0}}>{p.k}</div>
                </div>
              );
            })}
          </div>
          <div style={{position: 'absolute', left: NB.x, top: NB.y + 24 + programs.length * NB.h + 26, opacity: ramp(frame, nbFrom + 10, nbFrom + 24)}}>
            <DotLegend small />
          </div>
        </>
      ) : null}

      {/* the bottleneck moves */}
      {frame >= loadedF ? (
        <>
          <div style={{position: 'absolute', left: RIGHT.x, top: P.row1 - 46, opacity: pipeIn}}>
            <Label>the plain chain</Label>
          </div>
          <div style={{position: 'absolute', left: P.x0, top: P.row1, display: 'flex', alignItems: 'center', gap: 18, opacity: pipeIn}}>
            <Stage2 name="plate" state="bad" show={1} />
            <div style={{fontSize: 40, color: color.ink3}}>→</div>
            <Stage2 name="bowl" state="idle" show={0.55} />
          </div>
          <div style={{position: 'absolute', left: RIGHT.x, top: P.row2 - 46, opacity: pipeIn}}>
            <Label>cheese loaded into the bowl first</Label>
          </div>
          <div style={{position: 'absolute', left: P.x0, top: P.row2, display: 'flex', alignItems: 'center', gap: 18, opacity: pipeIn}}>
            <Stage2 name="plate" state={plateOk ? 'ok' : 'idle'} show={1} />
            <div style={{fontSize: 40, color: color.ink3}}>→</div>
            <Stage2 name="bowl" state={bowlBad ? 'bad' : 'idle'} show={1} />
            <div style={{marginLeft: 10, opacity: Math.min(1, countIn), transform: `scale(${0.8 + 0.2 * Math.min(1.1, countIn)})`, transformOrigin: 'left center'}}>
              <div style={{fontSize: 46, fontWeight: 840, color: color.ink, lineHeight: 1, fontVariantNumeric: 'tabular-nums'}}>
                {loadedK} <span style={{fontSize: 30, color: color.ink2, fontWeight: 700}}>of {loadedN}</span>
              </div>
              <div style={{fontSize: 24, fontWeight: 650, color: color.ink2}}>delivered</div>
            </div>
          </div>
          {/* the red marker: where it fails */}
          <div style={{position: 'absolute', left: marker.x, top: marker.y, width: 236, display: 'flex', justifyContent: 'center', opacity: pipeIn}}>
            <div style={{display: 'flex', alignItems: 'center', gap: 8, background: color.bad, color: '#fff', borderRadius: 999, padding: '6px 18px', fontSize: 24, fontWeight: 760, boxShadow: shadow.pill, whiteSpace: 'nowrap'}}>
              ▲ {slide > 0.5 ? 'the bottleneck moved' : 'fails here'}
            </div>
          </div>
        </>
      ) : null}

      {/* the loaded run, carrying the bowl */}
      {frame >= movedF - 12 ? (
        <div style={{position: 'absolute', left: RIGHT.x, top: 590, opacity: clipIn, transform: `translateY(${(1 - Math.min(1, clipIn)) * 24}px)`}}>
          <Footage clip={CLIPS.r2Loaded} segments={play.segments} startAt={play.startAt} width={RIGHT.w} height={Math.round((RIGHT.w * 9) / 16)} label={<Chip size={24}>cheese loaded · carrying the bowl</Chip>} />
        </div>
      ) : null}
    </JobStage>
  );
};
