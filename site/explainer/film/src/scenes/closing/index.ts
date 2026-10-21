// Scenes of the "closing" group. Map each beat id this group owns to its scene component.
import type React from 'react';
import {BFleet} from './BFleet';
import {BPlanted} from './BPlanted';
import {BSymptom} from './BSymptom';
import {BProbe} from './BProbe';
import {BFix} from './BFix';
import {BResult} from './BResult';
import {Results} from './Results';
import {Lessons} from './Lessons';
import {Next} from './Next';
import {Mayfly} from './Mayfly';
import {End} from './End';

export const SCENES_CLOSING: Record<string, React.FC> = {
  'b-fleet': BFleet,
  'b-planted': BPlanted,
  'b-symptom': BSymptom,
  'b-probe': BProbe,
  'b-fix': BFix,
  'b-result': BResult,
  results: Results,
  lessons: Lessons,
  next: Next,
  mayfly: Mayfly,
  end: End,
};
