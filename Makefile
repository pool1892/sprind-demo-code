# `--locked` everywhere: uv refuses to run when uv.lock does not match
# pyproject.toml, instead of quietly resolving again.

.PHONY: setup check check-sim check-model check-agents agents-gate smoke smoke-world nightly soak-test \
	dashboard mini world world-check supplement score analyse leak-audit live

# Build .venv from uv.lock.
setup:
	uv sync --locked

# Blind agent analysis of an existing, cleared development-world export.
# ANALYSE_FLAGS supplies --world-root, --export, --out (new), --condition.
# Only experiment arms take the heavy lock, in the integration command itself.
analyse:
	FLEETLAB_MIOPEN_DIR=$(MIOPEN_DIR) uv run --locked python -m fleetlab.integration.analyse run $(ANALYSE_FLAGS)

# One schema-checked analysis of a development world, to a new output directory.
score:
	uv run --locked python -m fleetlab.scoring $(SCORE_FLAGS)
# Claim-4 pass/sweep. CPU-only scoring/figures use the module without heavy-run.
# Example: make live LIVE_FLAGS="pass --world WORLD --rule timeout --out NEW_DIR"
live:
	tools/heavy-run uv run --locked python -m fleetlab.live.pass_cli $(LIVE_FLAGS)

# Lint, types and unit tests. Needs no GPU and no simulator.
check:
	uv run --locked ruff check
	uv run --locked ruff format --check
	uv run --locked mypy
	uv run --locked pytest -n 4

# The tests that build LIBERO scenes and render them (marked `sim`): resets,
# control modes, rendering contexts, start states, import order, LeRobot's own
# environments. No model is loaded, so they run as a light job, a few simulator
# processes at most and one at a time.
check-sim:
	tools/heavy-run --light uv run --locked pytest -m sim

# The tests that load a policy onto the GPU (marked `model`): the same chunk bits
# in another process, rows independent of their neighbours, the batch size fixed.
# Heavy, so they queue behind other heavy jobs. pytest exits with 5 when it
# collects nothing; until the engine brings these tests that is not a failure.
check-model:
	tools/heavy-run uv run --locked pytest -m model; status=$$?; \
	if [ $$status -eq 5 ]; then echo "check-model: no model tests yet"; else exit $$status; fi

# The agent runtime against both CLIs for real: first the free checks of
# the pinned CLI versions (agents-gate), then about forty calls through the CLIs' logins on
# the toy world, at most eight at once. Latency, tokens and the blinding verdict are
# written into src/fleetlab/agents/README.md. No GPU and no simulator, so no heavy lock.
check-agents:
	uv run --locked python -m fleetlab.agents.measure

# Only the free checks (no model call): sandbox probes, wire, isolation, source audit.
# A batch refuses to start on a CLI version without a passing record from here.
agents-gate:
	uv run --locked python -m fleetlab.agents.checks $(AGENTS_GATE_FLAGS)

# One LIBERO episode with the baseline policy on the GPU, written to data/smoke/.
# Loads a model onto the GPU, so it queues behind other heavy jobs.
smoke:
	tools/heavy-run uv run --locked python -m fleetlab.smoke

# The ladder of worlds (src/fleetlab/engine/ladder.py). The pinned MIOpen
# database the references were made with is machine-wide; FLEETLAB_MIOPEN_DIR overrides.
MIOPEN_DIR ?= $(or $(FLEETLAB_MIOPEN_DIR),$(HOME)/.cache/robots-probes/engine/miopen)

# The smoke world (w30): one robot, 8 episodes, every chain checked bit for bit against
# src/fleetlab/engine/smoke_reference.json; non-zero exit when one differs. Run it on every
# merge. Heavy (the model on the GPU), about two minutes; resumes when killed, because the
# run directory under data/smoke-world/ is named after the commit.
smoke-world:
	tools/heavy-run uv run --locked python -m fleetlab.engine.ladder smoke --miopen-dir $(MIOPEN_DIR)

# The nightly worlds (w20 with its fixed seed, w21 with the date as seed), their oracle
# replays and live passes, the canary, the scores and the comparison with the previous
# night, in one heavy job under ~/.cache/robots-probes/nightly/<date>/. The comparison is
# comparison.md there, for `make dashboard DASHBOARD_FLAGS="--nightly FILE"`.
nightly:
	tools/heavy-run uv run --locked python -m fleetlab.engine.ladder nightly --miopen-dir $(MIOPEN_DIR)

# The soak test: randomized nightly worlds one after another until HOURS of
# running (6), the canary after every world, then the record (soak-summary.md) in SOAK_ROOT.
# One heavy job of hours. Resumable: run it again with the same SOAK_ROOT.
SOAK_ROOT ?= $(HOME)/.cache/robots-probes/soak/$(shell date +%F)
soak-test:
	tools/heavy-run uv run --locked python -m fleetlab.engine.ladder soak-run --root $(SOAK_ROOT) \
		--hours $(or $(HOURS),6) --miopen-dir $(MIOPEN_DIR)

# The dashboard (src/fleetlab/dashboard/README.md): one page per run
# directory in RUNS, one per world root in WORLDS (both runs of a world on one page)
# and the progress page, built into data/dashboard and published with `hub publish`.
# Artifact ids go to dashboard/ids.json. Without RUNS and WORLDS
# only the progress page is rebuilt. TRUTH=1 adds the hidden side (development worlds
# only); DASHBOARD_FLAGS passes more, e.g. --nightly FILE.
dashboard:
	uv run --locked python -m fleetlab.dashboard publish --out data/dashboard \
		$(if $(TRUTH),--truth) $(DASHBOARD_FLAGS) $(if $(WORLDS),--world-roots $(WORLDS)) \
		--runs $(RUNS)

# The miniature fleet end to end (src/fleetlab/integration/README.md):
# world, both runs and the oracle's replays (heavy, under tools/heavy-run inside),
# oracle, export, two analyses by opus (logs alone; with experiments, whose arms are
# heavy), fresh replays if the scorer needs any (heavy), score, fixtures (the version
# of fleetlab.worlds.mini). Resumes in MINI_DIR; MINI_FLAGS passes more (--through STEP,
# --only STEP, --workers N, --model M, --fixtures-version V).
MINI_DIR ?= $(HOME)/.cache/robots-probes/mini/w10-v2
mini:
	FLEETLAB_MIOPEN_DIR=$(HOME)/.cache/robots-probes/engine/miopen \
		uv run --locked python -m fleetlab.integration --dir $(MINI_DIR) $(MINI_FLAGS)

# The worlds of stage 3 (src/fleetlab/worlds/README.md). A whole
# world as one heavy job, resumable: its files, the observed and the later run, the fleet's
# records and the derived events in each, the export with the leak audit, the summary, under
# ~/.cache/robots-probes/worlds/<world id>/. WORLD is D1, D2 or D3; WORLD_FLAGS passes more
# (--through STEP, --only STEP, --workers N, --root DIR).
WORLD ?= D1
world:
	FLEETLAB_MIOPEN_DIR=$(MIOPEN_DIR) tools/heavy-run uv run --locked python -m fleetlab.worlds \
		run $(WORLD) $(WORLD_FLAGS)

# Compose D1 to D3 on the CPU and report the expected difficulty and the gate's sizing rule.
world-check:
	FLEETLAB_MIOPEN_DIR=$(MIOPEN_DIR) uv run --locked python -m fleetlab.worlds check

# What the worlds need beyond the faults' calibration: libero_object and the interaction pairs'
# joint points (fleetlab.worlds.supplement). The run is heavy (about 2,300 episodes); the
# analysis writes src/fleetlab/worlds/curves_supplement.json, which the generator reads.
supplement:
	FLEETLAB_MIOPEN_DIR=$(MIOPEN_DIR) tools/heavy-run uv run --locked python -m fleetlab.worlds \
		supplement run
	uv run --locked python -m fleetlab.worlds supplement analyse

# Claim-3 baselines 1–4: visible-input methods, 200 seeded draws, four strong-tier
# subscription calls. BASELINES_OUT must be new; --offline omits the model methods.
.PHONY: baselines
baselines:
	uv run --locked python -m fleetlab.baselines --export $(EXPORT) --out $(BASELINES_OUT) \
		$(if $(SEED),--seed $(SEED)) $(if $(CHANCE_TABLE),--chance-table $(CHANCE_TABLE)) $(BASELINES_FLAGS)

# Audit the real analysis sandbox. Always writes a fresh record; exit 1 means
# a leak, a case to judge, or incomplete coverage. No GPU except a requested
# experiment, which the audit sends through heavy-run in a new output root.
leak-audit:
	uv run --locked python -m fleetlab.audit --world "$(WORLD)" --export "$(EXPORT)" $(AUDIT_FLAGS)
