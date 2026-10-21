# The explainer film

A new demo film, written for a viewer who knows nothing of the project: the problem, the
setup, how the agent works, two repairs shown end to end, the fleet instrument, the results,
what we learned and what it means.

- `script.md`: the narration, one section per beat; every number a placeholder.
- `fields.json`, `records.json`: what each placeholder means and which record holds it (the
  reliability record page's fields and records, plus `r2_programs`).
- `film/`: the Remotion project that renders the film from the bound data (work in
  progress; `node_modules/`, `public/` and `out/` are not committed).

Bind the script (every number from the records; `--final` refuses anything unbound):

```sh
uv run --locked python -m fleetlab.film.record bind site/explainer/script.md \
    --fields site/explainer/fields.json --records site/explainer/records.json \
    --out ~/.cache/robots-probes/demo/explainer/script-v1 --final
```

The bound script is `<out>/page/index.md`; `<out>/derived.json` holds every derived value
(counts, verbatim quotes) with the records and checksums it was read from, and is what the
film's graphics read.
