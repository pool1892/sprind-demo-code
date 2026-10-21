"""The integration chain of stage 2: the miniature fleet end to end.

`chain` runs the steps (world, run, oracle, export, analysis, fresh replays, score,
fixtures) under one root directory; `oracle` makes the replays and the oracle table;
`scoring_step` connects the scorer (`fleetlab.scoring`) and the fixtures. See
README.md in this directory for what each step produced on the miniature fleet.
"""
