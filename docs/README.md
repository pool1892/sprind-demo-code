# Documentation map

The [README](../README.md) describes the demo and how to set it up; the film and
its records are on the [demo page](https://pool1892.github.io/sprind-demo-page/).

## Design notes

- [Contracts](design/contracts.md): the interfaces of the test bed: fleets,
  logs, faults, actions, worlds, the oracle, the agents' tools and skills.
  The Python side is `src/fleetlab/contracts/`.
- [Faults](design/faults.md): the fault mechanism, the library and the measured
  dose-response curves; [the fault library on film](design/faults-films/index.md)
  shows each fault type beside its healthy twin.
- [Jobs](design/jobs.md): the interfaces of the jobs runtime and the forms the
  agents' job skills and task text are written against.
- [The improvement loop](design/loop.md): the interfaces of the two-round
  investigation over a fleet; [learned skills](design/learned.md) and
  [the reuse run](design/reuse-run.md) describe how an investigation's own
  diagnostic is kept and given to a later one;
  [the demonstration's recipe](design/m3-recipe.md) records how its fleets and
  runs were selected.

Where a note and the code differ, the code decides. Module READMEs under
`src/fleetlab/` and the spike READMEs under `spikes/` describe the code and the
development measurements.

## Results

- [Live risk](results/live-risk/tables.md): measured forecasts of failure while an
  attempt runs, with the [metrics](results/live-risk/metrics.json) and the
  [selection receipt](results/live-risk/selection.json).
