# Stored-world oracle comparison, 2026-10-09

Generated from the stored JSON reports at report commit `0fae966`; D1 fresh replays ran at `2beb1ea` (the replay code is identical at both revisions). Old records are read-only. D1 adds its original missing random-draw joint replays; D2 and D3 use stored replays only. Every changed numeric field is listed, even if it does not appear in the Markdown summary; added numeric fields are listed separately. Arrays of records are matched by stable keys. `null` means unavailable; `<absent>` means the old report had no such field.

Counts are episodes, rows, jobs, chains or draws as named. `points`, `low` and `high` are percentage points over 336 scoring episodes. `share` is net gain divided by achievable gain; `above_zero` is the fraction of draws with positive net. `bounds.<statistic>.0` and `.1` are the lower and upper worst-case bounds over all draws. Interaction detail counts are signed episodes: standalone harm is the joint-removal net minus the opposite single-removal net; their sum is `summed_harm_net`.

## w41

Old: `/home/cs/.cache/robots-probes/worlds/w41/oracle/oracle.json`. New: `/home/cs/.cache/robots-probes/worlds/w41/oracle-review-20261009/oracle.json`.

### Changed numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| chance.4 h.above_zero | 0.204 | 0.235 | missing-draw repair |
| chance.4 h.mean_points | 1.47 | 1.72 | missing-draw repair |
| chance.4 h.mean_share | 0.052 | 0.0608 | missing-draw repair |
| chance.4 h.skipped_fresh_replay | 14 | 0 | missing-draw repair |
| chance.4 h.valued | 186 | 200 | missing-draw repair |
| chance.8 h.above_zero | 0.279 | 0.31 | missing-draw repair |
| chance.8 h.mean_points | 2.059 | 2.335 | missing-draw repair |
| chance.8 h.mean_share | 0.0558 | 0.0633 | missing-draw repair |
| chance.8 h.p95_points | 8.467 | 8.943 | missing-draw repair |
| chance.8 h.p95_share | 0.2294 | 0.2423 | missing-draw repair |
| chance.8 h.skipped_fresh_replay | 28 | 0 | missing-draw repair |
| chance.8 h.valued | 172 | 200 | missing-draw repair |
| chance.k = 3.above_zero | 0.04 | 0.045 | missing-draw repair |
| chance.k = 3.mean_points | 0.313 | 0.353 | missing-draw repair |
| chance.k = 3.mean_share | 0.0157 | 0.0177 | missing-draw repair |
| chance.k = 3.skipped_fresh_replay | 1 | 0 | missing-draw repair |
| chance.k = 3.valued | 199 | 200 | missing-draw repair |
| chance.k = 6.above_zero | 0.101 | 0.11 | missing-draw repair |
| chance.k = 6.mean_points | 0.744 | 0.807 | missing-draw repair |
| chance.k = 6.mean_share | 0.0202 | 0.0219 | missing-draw repair |
| chance.k = 6.p95_points | 6.31 | 8.333 | missing-draw repair |
| chance.k = 6.p95_share | 0.171 | 0.2258 | missing-draw repair |
| chance.k = 6.skipped_fresh_replay | 2 | 0 | missing-draw repair |
| chance.k = 6.valued | 198 | 200 | missing-draw repair |
| rows | 2156 | 3080 | missing-draw repair (new D1 joint replays) |

### Added numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| all_removed_known_failures | <absent> | 6 | fault-free-outcome guard |
| all_removed_missing | <absent> | 0 | fault-free-outcome guard |
| interactions.f01+f05.detail.standalone_harm_net.0 | <absent> | -1 | interaction-verdict repair |
| interactions.f01+f05.detail.standalone_harm_net.1 | <absent> | 11 | interaction-verdict repair |
| interactions.f01+f05.detail.summed_harm_net | <absent> | 10 | interaction-verdict repair |
| interactions.f01+f06.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f01+f06.detail.standalone_harm_net.1 | <absent> | 13 | interaction-verdict repair |
| interactions.f01+f06.detail.summed_harm_net | <absent> | 14 | interaction-verdict repair |
| interactions.f01+f07.detail.standalone_harm_net.0 | <absent> | 5 | interaction-verdict repair |
| interactions.f01+f07.detail.standalone_harm_net.1 | <absent> | 24 | interaction-verdict repair |
| interactions.f01+f07.detail.summed_harm_net | <absent> | 29 | interaction-verdict repair |
| interactions.f05+f06.detail.standalone_harm_net.0 | <absent> | 11 | interaction-verdict repair |
| interactions.f05+f06.detail.standalone_harm_net.1 | <absent> | 20 | interaction-verdict repair |
| interactions.f05+f06.detail.summed_harm_net | <absent> | 31 | interaction-verdict repair |
| interactions.f05+f07.detail.standalone_harm_net.0 | <absent> | 8 | interaction-verdict repair |
| interactions.f05+f07.detail.standalone_harm_net.1 | <absent> | 27 | interaction-verdict repair |
| interactions.f05+f07.detail.summed_harm_net | <absent> | 35 | interaction-verdict repair |
| policy_failures_on_healthy_robot | <absent> | 6 | healthy-robot wording |
| runs./home/cs/.cache/robots-probes/worlds/w41/oracle-review-20261009/fresh/chance/run.as_run | <absent> | 0 | missing-draw repair (new D1 joint replays) |
| runs./home/cs/.cache/robots-probes/worlds/w41/oracle-review-20261009/fresh/chance/run.replays | <absent> | 924 | missing-draw repair (new D1 joint replays) |
| runs./home/cs/.cache/robots-probes/worlds/w41/oracle-review-20261009/fresh/chance/run.rows | <absent> | 947 | missing-draw repair (new D1 joint replays) |
| runs./home/cs/.cache/robots-probes/worlds/w41/oracle-review-20261009/fresh/chance/run.unmodified | <absent> | 23 | missing-draw repair (new D1 joint replays) |

### Changed verdicts

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| none | | | |

## w42

Old: `/home/cs/.cache/robots-probes/worlds/w42/oracle/oracle.json`. New: `/home/cs/.cache/robots-probes/worlds/w42/oracle-review-20261009/oracle.json`.

### Changed numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| chance.4 h.above_zero | 0.214 | null | missing-draw repair |
| chance.4 h.mean_points | 0.79 | null | missing-draw repair |
| chance.4 h.mean_share | 0.0379 | null | missing-draw repair |
| chance.4 h.p5_points | -0.298 | null | missing-draw repair |
| chance.4 h.p5_share | -0.0143 | null | missing-draw repair |
| chance.4 h.p95_points | 4.464 | null | missing-draw repair |
| chance.4 h.p95_share | 0.2143 | null | missing-draw repair |
| chance.8 h.above_zero | 0.347 | null | missing-draw repair |
| chance.8 h.mean_points | 1.719 | null | missing-draw repair |
| chance.8 h.mean_share | 0.0438 | null | missing-draw repair |
| chance.8 h.p5_points | -0.417 | null | missing-draw repair |
| chance.8 h.p5_share | -0.0106 | null | missing-draw repair |
| chance.8 h.p95_points | 7.44 | null | missing-draw repair |
| chance.8 h.p95_share | 0.1894 | null | missing-draw repair |
| chance.k = 6.above_zero | 0.181 | null | missing-draw repair |
| chance.k = 6.mean_points | 0.655 | null | missing-draw repair |
| chance.k = 6.mean_share | 0.0159 | null | missing-draw repair |
| chance.k = 6.p5_points | 0.0 | null | missing-draw repair |
| chance.k = 6.p5_share | 0.0 | null | missing-draw repair |
| chance.k = 6.p95_points | 4.196 | null | missing-draw repair |
| chance.k = 6.p95_share | 0.1022 | null | missing-draw repair |

### Added numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| all_removed_known_failures | <absent> | 8 | fault-free-outcome guard |
| all_removed_missing | <absent> | 0 | fault-free-outcome guard |
| chance.4 h.bounds.above_zero.0 | <absent> | 0.21 | missing-draw repair |
| chance.4 h.bounds.above_zero.1 | <absent> | 0.23 | missing-draw repair |
| chance.4 h.bounds.mean_points.0 | <absent> | -0.357 | missing-draw repair |
| chance.4 h.bounds.mean_points.1 | <absent> | 1.643 | missing-draw repair |
| chance.4 h.bounds.mean_share.0 | <absent> | -0.0171 | missing-draw repair |
| chance.4 h.bounds.mean_share.1 | <absent> | 0.0789 | missing-draw repair |
| chance.4 h.bounds.p5_points.0 | <absent> | -0.298 | missing-draw repair |
| chance.4 h.bounds.p5_points.1 | <absent> | -0.298 | missing-draw repair |
| chance.4 h.bounds.p5_share.0 | <absent> | -0.0143 | missing-draw repair |
| chance.4 h.bounds.p5_share.1 | <absent> | -0.0143 | missing-draw repair |
| chance.4 h.bounds.p95_points.0 | <absent> | 4.464 | missing-draw repair |
| chance.4 h.bounds.p95_points.1 | <absent> | 4.866 | missing-draw repair |
| chance.4 h.bounds.p95_share.0 | <absent> | 0.2143 | missing-draw repair |
| chance.4 h.bounds.p95_share.1 | <absent> | 0.2336 | missing-draw repair |
| chance.8 h.bounds.above_zero.0 | <absent> | 0.335 | missing-draw repair |
| chance.8 h.bounds.above_zero.1 | <absent> | 0.37 | missing-draw repair |
| chance.8 h.bounds.mean_points.0 | <absent> | -0.32 | missing-draw repair |
| chance.8 h.bounds.mean_points.1 | <absent> | 3.18 | missing-draw repair |
| chance.8 h.bounds.mean_share.0 | <absent> | -0.0081 | missing-draw repair |
| chance.8 h.bounds.mean_share.1 | <absent> | 0.0809 | missing-draw repair |
| chance.8 h.bounds.p5_points.0 | <absent> | -0.595 | missing-draw repair |
| chance.8 h.bounds.p5_points.1 | <absent> | -0.312 | missing-draw repair |
| chance.8 h.bounds.p5_share.0 | <absent> | -0.0152 | missing-draw repair |
| chance.8 h.bounds.p5_share.1 | <absent> | -0.008 | missing-draw repair |
| chance.8 h.bounds.p95_points.0 | <absent> | 7.44 | missing-draw repair |
| chance.8 h.bounds.p95_points.1 | <absent> | 15.774 | missing-draw repair |
| chance.8 h.bounds.p95_share.0 | <absent> | 0.1894 | missing-draw repair |
| chance.8 h.bounds.p95_share.1 | <absent> | 0.4015 | missing-draw repair |
| chance.k = 6.bounds.above_zero.0 | <absent> | 0.18 | missing-draw repair |
| chance.k = 6.bounds.above_zero.1 | <absent> | 0.185 | missing-draw repair |
| chance.k = 6.bounds.mean_points.0 | <absent> | 0.369 | missing-draw repair |
| chance.k = 6.bounds.mean_points.1 | <absent> | 0.869 | missing-draw repair |
| chance.k = 6.bounds.mean_share.0 | <absent> | 0.009 | missing-draw repair |
| chance.k = 6.bounds.mean_share.1 | <absent> | 0.0212 | missing-draw repair |
| chance.k = 6.bounds.p5_points.0 | <absent> | 0.0 | missing-draw repair |
| chance.k = 6.bounds.p5_points.1 | <absent> | 0.0 | missing-draw repair |
| chance.k = 6.bounds.p5_share.0 | <absent> | 0.0 | missing-draw repair |
| chance.k = 6.bounds.p5_share.1 | <absent> | 0.0 | missing-draw repair |
| chance.k = 6.bounds.p95_points.0 | <absent> | 4.182 | missing-draw repair |
| chance.k = 6.bounds.p95_points.1 | <absent> | 4.464 | missing-draw repair |
| chance.k = 6.bounds.p95_share.0 | <absent> | 0.1018 | missing-draw repair |
| chance.k = 6.bounds.p95_share.1 | <absent> | 0.1087 | missing-draw repair |
| interactions.f01+f02.detail.standalone_harm_net.0 | <absent> | -1 | interaction-verdict repair |
| interactions.f01+f02.detail.standalone_harm_net.1 | <absent> | -1 | interaction-verdict repair |
| interactions.f01+f02.detail.summed_harm_net | <absent> | -2 | interaction-verdict repair |
| interactions.f01+f03.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f01+f03.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f01+f03.detail.summed_harm_net | <absent> | 1 | interaction-verdict repair |
| interactions.f01+f07.detail.standalone_harm_net.0 | <absent> | 6 | interaction-verdict repair |
| interactions.f01+f07.detail.standalone_harm_net.1 | <absent> | -6 | interaction-verdict repair |
| interactions.f01+f07.detail.summed_harm_net | <absent> | 0 | interaction-verdict repair |
| interactions.f01+f12.detail.standalone_harm_net.0 | <absent> | 0 | interaction-verdict repair |
| interactions.f01+f12.detail.standalone_harm_net.1 | <absent> | 3 | interaction-verdict repair |
| interactions.f01+f12.detail.summed_harm_net | <absent> | 3 | interaction-verdict repair |
| interactions.f01+f13.detail.standalone_harm_net.0 | <absent> | 5 | interaction-verdict repair |
| interactions.f01+f13.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f01+f13.detail.summed_harm_net | <absent> | 5 | interaction-verdict repair |
| interactions.f02+f07.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f02+f07.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f02+f07.detail.summed_harm_net | <absent> | 1 | interaction-verdict repair |
| interactions.f03+f07.detail.standalone_harm_net.0 | <absent> | 2 | interaction-verdict repair |
| interactions.f03+f07.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f03+f07.detail.summed_harm_net | <absent> | 2 | interaction-verdict repair |
| interactions.f05+f06.detail.standalone_harm_net.0 | <absent> | 55 | interaction-verdict repair |
| interactions.f05+f06.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f05+f06.detail.summed_harm_net | <absent> | 55 | interaction-verdict repair |
| interactions.f05+f08.detail.standalone_harm_net.0 | <absent> | 52 | interaction-verdict repair |
| interactions.f05+f08.detail.standalone_harm_net.1 | <absent> | -4 | interaction-verdict repair |
| interactions.f05+f08.detail.summed_harm_net | <absent> | 48 | interaction-verdict repair |
| interactions.f05+f09.detail.standalone_harm_net.0 | <absent> | 17 | interaction-verdict repair |
| interactions.f05+f09.detail.standalone_harm_net.1 | <absent> | 6 | interaction-verdict repair |
| interactions.f05+f09.detail.summed_harm_net | <absent> | 23 | interaction-verdict repair |
| interactions.f05+f10.detail.standalone_harm_net.0 | <absent> | 11 | interaction-verdict repair |
| interactions.f05+f10.detail.standalone_harm_net.1 | <absent> | 26 | interaction-verdict repair |
| interactions.f05+f10.detail.summed_harm_net | <absent> | 37 | interaction-verdict repair |
| interactions.f06+f08.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f06+f08.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f06+f08.detail.summed_harm_net | <absent> | 1 | interaction-verdict repair |
| interactions.f06+f09.detail.standalone_harm_net.0 | <absent> | 2 | interaction-verdict repair |
| interactions.f06+f09.detail.standalone_harm_net.1 | <absent> | 10 | interaction-verdict repair |
| interactions.f06+f09.detail.summed_harm_net | <absent> | 12 | interaction-verdict repair |
| interactions.f06+f10.detail.standalone_harm_net.0 | <absent> | -1 | interaction-verdict repair |
| interactions.f06+f10.detail.standalone_harm_net.1 | <absent> | 14 | interaction-verdict repair |
| interactions.f06+f10.detail.summed_harm_net | <absent> | 13 | interaction-verdict repair |
| interactions.f07+f12.detail.standalone_harm_net.0 | <absent> | -2 | interaction-verdict repair |
| interactions.f07+f12.detail.standalone_harm_net.1 | <absent> | 5 | interaction-verdict repair |
| interactions.f07+f12.detail.summed_harm_net | <absent> | 3 | interaction-verdict repair |
| interactions.f07+f13.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f07+f13.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f07+f13.detail.summed_harm_net | <absent> | 1 | interaction-verdict repair |
| interactions.f08+f09.detail.standalone_harm_net.0 | <absent> | 0 | interaction-verdict repair |
| interactions.f08+f09.detail.standalone_harm_net.1 | <absent> | 10 | interaction-verdict repair |
| interactions.f08+f09.detail.summed_harm_net | <absent> | 10 | interaction-verdict repair |
| interactions.f08+f10.detail.standalone_harm_net.0 | <absent> | -3 | interaction-verdict repair |
| interactions.f08+f10.detail.standalone_harm_net.1 | <absent> | 12 | interaction-verdict repair |
| interactions.f08+f10.detail.summed_harm_net | <absent> | 9 | interaction-verdict repair |
| policy_failures_on_healthy_robot | <absent> | 8 | healthy-robot wording |

### Changed verdicts

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| patterns.interaction:f02,f01.verdict | not confirmed | weak | interaction-verdict repair |
| patterns.interaction:f03,f01.verdict | not confirmed | weak | interaction-verdict repair |

## w43

Old: `/home/cs/.cache/robots-probes/worlds/w43/oracle/oracle.json`. New: `/home/cs/.cache/robots-probes/worlds/w43/oracle-review-20261009/oracle.json`.

### Changed numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| chance.4 h.above_zero | 0.447 | null | missing-draw repair |
| chance.4 h.mean_points | 2.043 | null | missing-draw repair |
| chance.4 h.mean_share | 0.0624 | null | missing-draw repair |
| chance.4 h.p5_points | -0.595 | null | missing-draw repair |
| chance.4 h.p5_share | -0.0182 | null | missing-draw repair |
| chance.4 h.p95_points | 7.738 | null | missing-draw repair |
| chance.4 h.p95_share | 0.2364 | null | missing-draw repair |
| chance.8 h.above_zero | 0.511 | null | missing-draw repair |
| chance.8 h.mean_points | 2.521 | null | missing-draw repair |
| chance.8 h.mean_share | 0.0712 | null | missing-draw repair |
| chance.8 h.p5_points | -0.595 | null | missing-draw repair |
| chance.8 h.p5_share | -0.0168 | null | missing-draw repair |
| chance.8 h.p95_points | 9.003 | null | missing-draw repair |
| chance.8 h.p95_share | 0.2542 | null | missing-draw repair |
| chance.k = 3.above_zero | 0.101 | null | missing-draw repair |
| chance.k = 3.mean_points | 0.29 | null | missing-draw repair |
| chance.k = 3.mean_share | 0.015 | null | missing-draw repair |
| chance.k = 3.p5_points | 0.0 | null | missing-draw repair |
| chance.k = 3.p5_share | 0.0 | null | missing-draw repair |
| chance.k = 3.p95_points | 3.274 | null | missing-draw repair |
| chance.k = 3.p95_share | 0.1692 | null | missing-draw repair |
| chance.k = 6.above_zero | 0.177 | null | missing-draw repair |
| chance.k = 6.mean_points | 0.685 | null | missing-draw repair |
| chance.k = 6.mean_share | 0.0211 | null | missing-draw repair |
| chance.k = 6.p5_points | -0.298 | null | missing-draw repair |
| chance.k = 6.p5_share | -0.0092 | null | missing-draw repair |
| chance.k = 6.p95_points | 5.357 | null | missing-draw repair |
| chance.k = 6.p95_share | 0.1651 | null | missing-draw repair |

### Added numbers

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| all_removed_known_failures | <absent> | 11 | fault-free-outcome guard |
| all_removed_missing | <absent> | 0 | fault-free-outcome guard |
| chance.4 h.bounds.above_zero.0 | <absent> | 0.425 | missing-draw repair |
| chance.4 h.bounds.above_zero.1 | <absent> | 0.475 | missing-draw repair |
| chance.4 h.bounds.mean_points.0 | <absent> | -1.155 | missing-draw repair |
| chance.4 h.bounds.mean_points.1 | <absent> | 3.845 | missing-draw repair |
| chance.4 h.bounds.mean_share.0 | <absent> | -0.0353 | missing-draw repair |
| chance.4 h.bounds.mean_share.1 | <absent> | 0.1175 | missing-draw repair |
| chance.4 h.bounds.p5_points.0 | <absent> | -3.943 | missing-draw repair |
| chance.4 h.bounds.p5_points.1 | <absent> | -0.595 | missing-draw repair |
| chance.4 h.bounds.p5_share.0 | <absent> | -0.1205 | missing-draw repair |
| chance.4 h.bounds.p5_share.1 | <absent> | -0.0182 | missing-draw repair |
| chance.4 h.bounds.p95_points.0 | <absent> | 7.738 | missing-draw repair |
| chance.4 h.bounds.p95_points.1 | <absent> | 16.324 | missing-draw repair |
| chance.4 h.bounds.p95_share.0 | <absent> | 0.2364 | missing-draw repair |
| chance.4 h.bounds.p95_share.1 | <absent> | 0.4986 | missing-draw repair |
| chance.8 h.bounds.above_zero.0 | <absent> | 0.45 | missing-draw repair |
| chance.8 h.bounds.above_zero.1 | <absent> | 0.57 | missing-draw repair |
| chance.8 h.bounds.mean_points.0 | <absent> | -5.21 | missing-draw repair |
| chance.8 h.bounds.mean_points.1 | <absent> | 6.79 | missing-draw repair |
| chance.8 h.bounds.mean_share.0 | <absent> | -0.1471 | missing-draw repair |
| chance.8 h.bounds.mean_share.1 | <absent> | 0.1917 | missing-draw repair |
| chance.8 h.bounds.p5_points.0 | <absent> | -61.905 | missing-draw repair |
| chance.8 h.bounds.p5_points.1 | <absent> | -0.595 | missing-draw repair |
| chance.8 h.bounds.p5_share.0 | <absent> | -1.7479 | missing-draw repair |
| chance.8 h.bounds.p5_share.1 | <absent> | -0.0168 | missing-draw repair |
| chance.8 h.bounds.p95_points.0 | <absent> | 8.929 | missing-draw repair |
| chance.8 h.bounds.p95_points.1 | <absent> | 38.095 | missing-draw repair |
| chance.8 h.bounds.p95_share.0 | <absent> | 0.2521 | missing-draw repair |
| chance.8 h.bounds.p95_share.1 | <absent> | 1.0756 | missing-draw repair |
| chance.k = 3.bounds.above_zero.0 | <absent> | 0.1 | missing-draw repair |
| chance.k = 3.bounds.above_zero.1 | <absent> | 0.105 | missing-draw repair |
| chance.k = 3.bounds.mean_points.0 | <absent> | -0.021 | missing-draw repair |
| chance.k = 3.bounds.mean_points.1 | <absent> | 0.479 | missing-draw repair |
| chance.k = 3.bounds.mean_share.0 | <absent> | -0.0011 | missing-draw repair |
| chance.k = 3.bounds.mean_share.1 | <absent> | 0.0248 | missing-draw repair |
| chance.k = 3.bounds.p5_points.0 | <absent> | 0.0 | missing-draw repair |
| chance.k = 3.bounds.p5_points.1 | <absent> | 0.0 | missing-draw repair |
| chance.k = 3.bounds.p5_share.0 | <absent> | 0.0 | missing-draw repair |
| chance.k = 3.bounds.p5_share.1 | <absent> | 0.0 | missing-draw repair |
| chance.k = 3.bounds.p95_points.0 | <absent> | 3.274 | missing-draw repair |
| chance.k = 3.bounds.p95_points.1 | <absent> | 3.289 | missing-draw repair |
| chance.k = 3.bounds.p95_share.0 | <absent> | 0.1692 | missing-draw repair |
| chance.k = 3.bounds.p95_share.1 | <absent> | 0.17 | missing-draw repair |
| chance.k = 6.bounds.above_zero.0 | <absent> | 0.175 | missing-draw repair |
| chance.k = 6.bounds.above_zero.1 | <absent> | 0.185 | missing-draw repair |
| chance.k = 6.bounds.mean_points.0 | <absent> | 0.06 | missing-draw repair |
| chance.k = 6.bounds.mean_points.1 | <absent> | 1.06 | missing-draw repair |
| chance.k = 6.bounds.mean_share.0 | <absent> | 0.0018 | missing-draw repair |
| chance.k = 6.bounds.mean_share.1 | <absent> | 0.0327 | missing-draw repair |
| chance.k = 6.bounds.p5_points.0 | <absent> | -0.298 | missing-draw repair |
| chance.k = 6.bounds.p5_points.1 | <absent> | -0.298 | missing-draw repair |
| chance.k = 6.bounds.p5_share.0 | <absent> | -0.0092 | missing-draw repair |
| chance.k = 6.bounds.p5_share.1 | <absent> | -0.0092 | missing-draw repair |
| chance.k = 6.bounds.p95_points.0 | <absent> | 5.357 | missing-draw repair |
| chance.k = 6.bounds.p95_points.1 | <absent> | 5.655 | missing-draw repair |
| chance.k = 6.bounds.p95_share.0 | <absent> | 0.1651 | missing-draw repair |
| chance.k = 6.bounds.p95_share.1 | <absent> | 0.1743 | missing-draw repair |
| interactions.f01+f02.detail.standalone_harm_net.0 | <absent> | 1 | interaction-verdict repair |
| interactions.f01+f02.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f01+f02.detail.summed_harm_net | <absent> | 1 | interaction-verdict repair |
| interactions.f01+f06.detail.standalone_harm_net.0 | <absent> | 13 | interaction-verdict repair |
| interactions.f01+f06.detail.standalone_harm_net.1 | <absent> | 2 | interaction-verdict repair |
| interactions.f01+f06.detail.summed_harm_net | <absent> | 15 | interaction-verdict repair |
| interactions.f02+f06.detail.standalone_harm_net.0 | <absent> | 2 | interaction-verdict repair |
| interactions.f02+f06.detail.standalone_harm_net.1 | <absent> | 0 | interaction-verdict repair |
| interactions.f02+f06.detail.summed_harm_net | <absent> | 2 | interaction-verdict repair |
| interactions.f03+f04.detail.standalone_harm_net.0 | <absent> | 18 | interaction-verdict repair |
| interactions.f03+f04.detail.standalone_harm_net.1 | <absent> | 11 | interaction-verdict repair |
| interactions.f03+f04.detail.summed_harm_net | <absent> | 29 | interaction-verdict repair |
| interactions.f03+f05.detail.standalone_harm_net.0 | <absent> | 12 | interaction-verdict repair |
| interactions.f03+f05.detail.standalone_harm_net.1 | <absent> | 1 | interaction-verdict repair |
| interactions.f03+f05.detail.summed_harm_net | <absent> | 13 | interaction-verdict repair |
| interactions.f04+f05.detail.standalone_harm_net.0 | <absent> | 13 | interaction-verdict repair |
| interactions.f04+f05.detail.standalone_harm_net.1 | <absent> | -1 | interaction-verdict repair |
| interactions.f04+f05.detail.summed_harm_net | <absent> | 12 | interaction-verdict repair |
| interactions.f04+f08.detail.standalone_harm_net.0 | <absent> | 11 | interaction-verdict repair |
| interactions.f04+f08.detail.standalone_harm_net.1 | <absent> | 28 | interaction-verdict repair |
| interactions.f04+f08.detail.summed_harm_net | <absent> | 39 | interaction-verdict repair |
| interactions.f04+f10.detail.standalone_harm_net.0 | <absent> | 7 | interaction-verdict repair |
| interactions.f04+f10.detail.standalone_harm_net.1 | <absent> | 26 | interaction-verdict repair |
| interactions.f04+f10.detail.summed_harm_net | <absent> | 33 | interaction-verdict repair |
| interactions.f05+f08.detail.standalone_harm_net.0 | <absent> | -3 | interaction-verdict repair |
| interactions.f05+f08.detail.standalone_harm_net.1 | <absent> | 14 | interaction-verdict repair |
| interactions.f05+f08.detail.summed_harm_net | <absent> | 11 | interaction-verdict repair |
| interactions.f05+f10.detail.standalone_harm_net.0 | <absent> | 0 | interaction-verdict repair |
| interactions.f05+f10.detail.standalone_harm_net.1 | <absent> | 19 | interaction-verdict repair |
| interactions.f05+f10.detail.summed_harm_net | <absent> | 19 | interaction-verdict repair |
| policy_failures_on_healthy_robot | <absent> | 11 | healthy-robot wording |

### Changed verdicts

| Field | Old | New | Explaining repair |
| --- | ---: | ---: | --- |
| patterns.interaction:f02,f01.verdict | not confirmed | weak | interaction-verdict repair |

The wording change in every world is the healthy-robot wording: failures with every fault removed are failures of the policy itself on a healthy robot, the measure behind "needs new training data". Counts are unchanged. The fault-free-outcome guard and the other replay and proof guards of the same repair change no stored-world gain or interval; their regression cases are synthetic or mock command-status cases. None of the old marked pairs is relabelled as additive: the old world truth is kept intact.
