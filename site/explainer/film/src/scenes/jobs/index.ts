// Scenes of the "jobs" group. Map each beat id this group owns to its scene component.
import type React from 'react';
import {R1Fail} from './R1Fail';
import {R1Measure} from './R1Measure';
import {R1Experiment} from './R1Experiment';
import {R1Prove} from './R1Prove';
import {R1Keep} from './R1Keep';
import {R2Fail} from './R2Fail';
import {R2Bottleneck} from './R2Bottleneck';
import {R2Fix} from './R2Fix';
import {RSummary} from './RSummary';

export const SCENES_JOBS: Record<string, React.FC> = {
  'r1-fail': R1Fail,
  'r1-measure': R1Measure,
  'r1-experiment': R1Experiment,
  'r1-prove': R1Prove,
  'r1-keep': R1Keep,
  'r2-fail': R2Fail,
  'r2-bottleneck': R2Bottleneck,
  'r2-fix': R2Fix,
  'r-summary': RSummary,
};
