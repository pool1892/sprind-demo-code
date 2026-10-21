# The reliability record page

Source of the record page for the reliability story: two new jobs
built by the engineering agent and story B unchanged. The site's templates and style
belong to the site builder; `site/bundles/reliability-final-1-2026-10-20.json` names the
bound page, the film folder and the public address. The earlier record page
(`site/record/`) and the film's script and fields (`site/reliability/script.md`,
`fields.json`) stay as they are.

- `record.md`: the page. Every number and quotation is a placeholder; HTML comments are
  notes for its writers and never reach the bound page.
- `fields.json` (`fleetlab.film-fields/1`): what each placeholder means and which record
  holds it (`r1_` the first job, `r2_` the second, `b_`/`b2_` story B), the derived values
  (Wilson intervals, the receipts' pairs, verbatim quotes, start ranges, re-render
  counts), plain titles for the records table, and the patterns the bound page must never
  show (hidden world names, truth files, host paths, host screens, the investigations that
  did not qualify, the word film).
- `records-final-2-2026-10-20.json` (`fleetlab.film-records/1`): the immutable files that
  stand for those records in the final page. `records-final-1-2026-10-20.json` is the same
  without the recount; its output stays in `record-final-1`.
- `recount_second_job.py`: writes the recount record
  `reliability/record-recounts/second-job-diagnostics.json`, one row per episode, from the
  confirmation batch's public goal table and the coordinator's cheese record, so that the
  binder's existing Wilson intervals cover two counts no other record holds as data: the
  15 confirmation runs whose bowl goal stopped holding after arrival and held again (the
  agent's account gives that count only in prose; the recount agrees), and the reserved
  runs whose cheese ends on the table.
- `records-preview-1-2026-10-20.json`: preview 1 (first job only), kept unchanged; it binds
  the page as it was at commit `b0866a6`, whose output stays in `record-preview-1`.

## Bind

From this checkout, with its own environment (`make setup`, that is `uv sync --locked`):

```sh
out=~/.cache/robots-probes/demo/2026-10-20/reliability/site/record-final-2
tools/heavy-run --light -m 2G uv run --locked python -m fleetlab.film.record bind \
    site/reliability/record/record.md --fields site/reliability/record/fields.json \
    --records site/reliability/record/records-final-2-2026-10-20.json \
    --out "$out" --final
```

`$out/page/index.md` is the bound page the bundle names; `$out/bindings.json` holds every
value with its record, field and the SHA-256 of each file read; `$out/derived.json` the
derived values. `--final` refuses an unbound number or a gap, and every bind refuses a page
that shows a forbidden pattern. The site itself is built from the bundle once the film
folder it names exists (`python -m fleetlab.film site`).

## What the page reads

Only recorded files a reader may know about: each job's investigation record, release
receipt and the coordinator's episode-by-episode reconciliation, the development tables
the agent queried, the saved jobs, checks and learned skills, the library indexes, the
earlier saved job the first investigation started from (saved, not released; its own
receipt unchanged), the re-render verification reports and recordings of each job's
reserved start 31 (the first job's wrist reports are the corrected ones from
`replays-release-camera-fixed`), the coordinator's read of the cheese's final state in the
second job's reserved runs, the transcript of the first job's second session (one quoted
message), and story B's regular work, engagement record and later-fleet recipe run. No
truth table, hidden fault record, release trace or host feasibility run is read; the
feasibility runs appear only as one methods sentence.

Intervals are the binder's Wilson 95% intervals in whole percent.

## Two observations not bound to a field

- The second job's reserved start-31 pair: the plain chain reaches the plate goal at
  control step 145 and runs out of time on the bowl at 445, both goals false at the end;
  the page describes it in the methods section, separately from the development
  plate-push failures in the narrative.
- "The hand brushes the bowl on the way": in the cheese-aside program's ten tuning runs
  the robot touched the bowl before the plate step in nine, moving it 0 to 26 mm (public
  object table of session 5). The binder has no step-window filter for this count, so the
  page states it without a number.
