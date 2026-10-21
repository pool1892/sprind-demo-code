// request: experiment one opens with its chapter card, which settles into a corner tag. Then
// the request as recorded (its first sentence, verbatim), what "delivered" means as ticks on
// their cues, and the bar: 20 outlined dots with the 18 required marked, set before any test,
// and the bar in the voice's own words, each phrase on its cue: "the whole
// job completed without a failure / on at least 18 of 20 / fresh layouts".
import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Stage} from '../../components/basics';
import {Dot} from '../../components/dots';
import {useCue, useNumber, useQuote} from '../../data';
import {color, shadow, type} from '../../theme';
import {arrive, mix, span} from './kit';
import {ExperimentCard, ExperimentTag} from './ExperimentCard';
import {DotLegend} from './DotLegend';

const LX = 150;
const RX = 1150;

const TICKS = ['bowl on top of the cabinet', 'middle drawer open', 'everything at rest', 'hand clear'];

/** The request's first sentence: up to and including the first full stop that ends a sentence. */
const firstSentence = (q: string): string => {
  const m = q.match(/^.*?\.(?=\s|$)/);
  return m ? m[0] : q;
};

const Avatar: React.FC<{size: number}> = ({size}) => (
  <div style={{width: size, height: size, borderRadius: '50%', background: color.lead, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', overflow: 'hidden'}}>
    <svg width={size * 0.78} height={size * 0.78} viewBox="0 0 100 100">
      <circle cx={50} cy={38} r={20} fill="#fff" />
      <path d="M12,100 C14,70 32,62 50,62 C68,62 86,70 88,100 Z" fill="#fff" />
    </svg>
  </div>
);

export const Request: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const cChapter = useCue('chapter', 0.01);
  const cLead = useCue('request', 0.14);
  const cTicks = [useCue('goals', 0.34), useCue('drawer', 0.42), useCue('rest', 0.54), useCue('hand', 0.6)];
  const cBar = useCue('bar', 0.7);
  const cWhole = useCue('whole', 0.75);
  const cLeast = useCue('least', 0.82);
  const cCount = useCue('count', 0.86);
  const request = firstSentence(useQuote('r1_request'));
  const k = useNumber('r1_release_k');
  const n = useNumber('r1_release_n');
  const kNum = Number(k);
  const nNum = Number(n);

  const lead = arrive(frame, fps, cLead - 6);
  const bubble = arrive(frame, fps, cLead + 8);
  const typed = Math.max(0, Math.floor(((frame - (cLead + 10)) / 30) * 85));
  const shown = request.slice(0, typed);

  const DOT = 54;
  const DGAP = 16;
  const markStart = cWhole + 4;
  const whole = arrive(frame, fps, cWhole);
  const least = arrive(frame, fps, cLeast - 2);
  const num = arrive(frame, fps, cLeast + 6, 'lively');
  const label = arrive(frame, fps, cCount - 2);
  const card = arrive(frame, fps, cChapter - 10);
  const settle = span(frame, cLead - 22, cLead - 4);
  const tag = arrive(frame, fps, cLead - 8);

  return (
    <Stage>
      {/* the chapter card, then its tag in the corner */}
      {settle < 0.99 ? (
        <div style={{position: 'absolute', left: mix(settle, 960, 400), top: mix(settle, 540, 120), transform: `translate(-50%, -50%) scale(${mix(settle, 1, 0.4) * (0.9 + 0.1 * Math.min(1, card))})`, opacity: Math.min(1, card * 1.4) * (1 - settle)}}>
          <ExperimentCard which={1} width={820} />
        </div>
      ) : null}
      <div style={{position: 'absolute', left: 120, top: 46, opacity: tag}}>
        <ExperimentTag which={1} />
      </div>
      {/* the shift lead and the request */}
      <div style={{position: 'absolute', left: LX, top: 150, display: 'flex', alignItems: 'center', gap: 26, opacity: lead, transform: `translateY(${(1 - lead) * 20}px)`}}>
        <Avatar size={104} />
        <div>
          <div style={{fontSize: 40, fontWeight: 780, color: color.ink}}>Shift lead</div>
          <div style={{...type.small, fontSize: 28}}>the request, as a shift lead would write it</div>
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: LX,
          top: 296,
          width: 860,
          boxSizing: 'border-box',
          background: color.card,
          borderRadius: '8px 30px 30px 30px',
          boxShadow: shadow.card,
          padding: '32px 40px',
          borderLeft: `10px solid ${color.lead}`,
          opacity: bubble,
          transform: `translateY(${(1 - bubble) * 24}px)`,
        }}
      >
        {/* reserve the full text's height so the bubble does not grow while typing */}
        <div style={{position: 'relative', fontSize: 40, fontWeight: 560, lineHeight: 1.38, color: color.ink}}>
          <div style={{visibility: 'hidden'}}>“{request}”</div>
          <div style={{position: 'absolute', inset: 0}}>
            “{shown}
            {typed < request.length ? <span style={{color: color.lead}}>▍</span> : '”'}
          </div>
        </div>
      </div>

      {/* delivered means */}
      <div style={{position: 'absolute', left: LX, top: 650, ...type.kicker, opacity: arrive(frame, fps, cTicks[0] - 8)}}>delivered means</div>
      <div style={{position: 'absolute', left: LX, top: 704, display: 'flex', flexDirection: 'column', gap: 22}}>
        {TICKS.map((t, i) => {
          const p = arrive(frame, fps, cTicks[i]);
          const tick = arrive(frame, fps, cTicks[i] + 6, 'lively');
          return (
            <div key={t} style={{display: 'flex', alignItems: 'center', gap: 22, opacity: p, transform: `translateX(${(1 - p) * -24}px)`}}>
              <div style={{width: 48, height: 48, borderRadius: '50%', background: color.ok, color: '#fff', fontSize: 28, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', transform: `scale(${Math.min(1.15, tick)})`}}>✓</div>
              <div style={{fontSize: 38, fontWeight: 600, color: color.ink}}>{t}</div>
            </div>
          );
        })}
      </div>

      {/* the bar */}
      <div style={{position: 'absolute', left: RX, top: 230, ...type.kicker, opacity: arrive(frame, fps, cBar - 6)}}>the bar · set before any test</div>
      <div style={{position: 'absolute', left: RX, top: 290, display: 'grid', gridTemplateColumns: `repeat(10, ${DOT}px)`, gap: DGAP}}>
        {Array.from({length: nNum || 0}, (_, i) => {
          const show = span(frame, cBar + i, cBar + i + 8);
          const marked = i < kNum && frame >= markStart + i * 2;
          return (
            <div key={i} style={{opacity: show, transform: `scale(${(0.6 + 0.4 * show) * (marked ? 1 : 1)})`}}>
              <Dot state={marked ? 'outline-ok' : 'outline'} size={DOT} appear={0} />
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: RX, top: 450, width: 680, fontSize: 40, fontWeight: 760, lineHeight: 1.2, color: color.ink, opacity: Math.min(1, whole * 1.4), transform: `translateY(${(1 - Math.min(1, whole)) * 16}px)`}}>
        the whole job completed
        <br />
        without a failure
      </div>
      <div style={{position: 'absolute', left: RX, top: 578, fontSize: 36, fontWeight: 650, color: color.ink2, opacity: least}}>on at least</div>
      <div style={{position: 'absolute', left: RX, top: 624, opacity: Math.min(1, num * 1.4), transform: `translateY(${(1 - Math.min(1, num)) * 20}px) scale(${0.88 + 0.12 * num})`, transformOrigin: 'left center'}}>
        <div style={{fontSize: 140, fontWeight: 830, letterSpacing: '-0.04em', lineHeight: 1, color: color.ink, whiteSpace: 'nowrap'}}>
          <span style={{color: color.ok}}>{k}</span>
          <span style={{color: color.ink3, fontWeight: 700, fontSize: 90}}> of </span>
          {n}
        </div>
      </div>
      <div style={{position: 'absolute', left: RX, top: 784, fontSize: 44, fontWeight: 760, color: color.ink, opacity: label, transform: `translateY(${(1 - label) * 16}px)`}}>fresh layouts</div>
      <div style={{position: 'absolute', left: RX, top: 872, opacity: label}}>
        <DotLegend small />
      </div>
    </Stage>
  );
};
