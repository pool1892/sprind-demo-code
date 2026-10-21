// All built scenes by beat id. Each group owns its own folder and index; this file only
// merges them, so builders working in parallel never edit the same file.
import type React from 'react';
import {SCENES_OPENING} from './opening';
import {SCENES_JOBS} from './jobs';
import {SCENES_CLOSING} from './closing';

export const SCENES: Record<string, React.FC> = {...SCENES_OPENING, ...SCENES_JOBS, ...SCENES_CLOSING};
