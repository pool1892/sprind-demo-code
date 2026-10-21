// What we planted in experiment two's fleet, in the plain words the narration uses. One
// source for b-planted's cards and b-result's checklist, so the two match one to one. The
// sites and the robot are read from the film data (the public site note, the regular work,
// the change log); never the internal fault names.
import {color} from '../../theme';

type Site = {robots: Record<string, {k: number; n: number}>; k: number; n: number};
type ChangeRow = {day: number; scope: string; change: string};

export type PlantedKey = 'delay' | 'controller' | 'software' | 'calibration';

export const PLANTED: {key: PlantedKey; tone: string; title: string; check: string}[] = [
  {key: 'delay', tone: color.lead, title: 'a delay on the line to the data centre', check: 'delay compensated'},
  {key: 'controller', tone: color.violet, title: 'an overloaded controller', check: 'controller serviced'},
  {key: 'software', tone: color.teal, title: 'a software update that crops the camera images', check: 'software rolled back'},
  {key: 'calibration', tone: color.ink3, title: 'a calibration record rewritten the same day', check: 'calibration left alone'},
];

/** The weakest robot outside the linked site, from the regular work's records. */
export const lagRobotOf = (work: Record<string, Site>, linked: string) => {
  let best = '';
  let rate = 2;
  Object.keys(work)
    .filter((s) => s !== linked)
    .forEach((s) =>
      Object.entries(work[s].robots).forEach(([r, c]) => {
        if (c.k / c.n < rate) {
          rate = c.k / c.n;
          best = r;
        }
      }),
    );
  return best;
};

/** The site whose change log has two entries on one day, and that day. */
export const sameDayPairOf = (log: ChangeRow[]) => {
  for (const r of log) {
    const same = log.filter((x) => x.scope === r.scope && x.day === r.day);
    if (same.length >= 2 && r.scope !== 'fleet') return {site: r.scope, day: r.day};
  }
  return {site: '', day: -1};
};

/** The fleet's cast: the linked site, the robot that lags elsewhere, the site with two same-day changes. */
export const castOf = (fleet: Record<string, any>) => {
  const work = fleet.regular_work.sites as Record<string, Site>;
  const linked: string = fleet.b_linked_site?.value ?? '';
  const lagRobot = lagRobotOf(work, linked);
  const pair = sameDayPairOf((fleet.console?.change_log ?? []) as ChangeRow[]);
  const lagSite = Object.keys(work).find((s) => lagRobot in work[s].robots) ?? '';
  const siteOf: Record<PlantedKey, string> = {delay: linked, controller: lagSite, software: pair.site, calibration: pair.site};
  return {work, linked, lagRobot, lagSite, third: pair.site, day: pair.day, siteOf};
};
