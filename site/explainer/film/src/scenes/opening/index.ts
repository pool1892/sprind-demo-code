// Scenes of the "opening" group. Map each beat id this group owns to its scene component.
import type React from 'react';
import {ColdOpen} from './ColdOpen';
import {Title} from './Title';
import {Why} from './Why';
import {Question} from './Question';
import {Station} from './Station';
import {Agent} from './Agent';
import {Rules} from './Rules';
import {Request} from './Request';

export const SCENES_OPENING: Record<string, React.FC> = {
  'cold-open': ColdOpen,
  title: Title,
  why: Why,
  question: Question,
  station: Station,
  agent: Agent,
  rules: Rules,
  request: Request,
};
