# Working in this repository

Rules for people and coding agents who build or run things here. The
[README](README.md) says what the demo is and what the machine needs.

## Python and imports

- Python goes through `uv`. `make setup` builds `.venv` from `uv.lock`; every
  command runs as `uv run --locked ...`. `[tool.uv]` in
  [`pyproject.toml`](pyproject.toml) restricts uv to interpreters it manages
  itself; do not loosen it, and do not add a `uv.toml`, which makes uv ignore
  the whole `[tool.uv]` table.
- **torch before mujoco.** A process that loads mujoco before torch cannot use
  the GPU through PyTorch: two copies of LLVM meet and the first GPU call
  crashes. `import fleetlab` before anything that imports mujoco, robosuite,
  LIBERO or LeRobot's environments; the package keeps the order. A script that
  does not import `fleetlab` imports torch first.
- `fleetlab` and LIBERO are installed editable, so every process a job starts
  reads the files as they are at that moment. Do not edit `src/` or
  `third_party/` in a checkout while a measurement started from it is running.

## Simulator and GPU jobs

- Simulator and GPU jobs go through [`tools/heavy-run`](tools/heavy-run). A job
  that loads a model onto the GPU or starts more than four simulation processes
  is heavy: heavy jobs run one at a time under a memory ceiling, because the GPU
  shares system memory and the machine hangs when it fills. Everything else
  that simulates uses `tools/heavy-run --light`.
- Render headless (EGL) and hand over files: video, images, logs, HTML.
- Measurement results go under `~/.cache/robots-probes`, never into a
  checkout. The ignored `data/` directory holds only local outputs, such as
  those of `make smoke` and `make smoke-world`.

## Checks

- `make check`: lint, types and unit tests, no GPU. Run one at a time on the
  machine, as a light job:
  `flock "$XDG_RUNTIME_DIR/robots-check.lock" tools/heavy-run --light -m 12G make check`.
- Targeted tests: `tools/heavy-run --light -m 2G uv run --locked pytest <files>`
  with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. A change
  runs the test files it touches; the full check runs once per candidate for
  the main branch.
- `make check-sim`: the tests that build LIBERO scenes, without a model.
- `make smoke-world`: eight episodes replayed against the stored chain
  reference `src/fleetlab/engine/smoke_reference.json`. A change that moves
  those chains on purpose writes a new reference with
  `python -m fleetlab.engine.ladder smoke --update-reference` in the same
  commit and says why.
- The website is separate from the rest of the repository: everything under
  `site/`, the page and film builder `src/fleetlab/film` with its tests,
  `tests/test_film_framing*.py` and the `tools/film-*` scripts. Nothing else
  imports or reads them. A change that touches only these needs only their
  own checks, well under a minute, instead of the full check and the smoke
  world:

  ```sh
  W="site src/fleetlab/film tests/test_film_framing.py tests/test_film_framing_sim.py"
  uv run --locked ruff check $W && uv run --locked ruff format --check $W
  uv run --locked mypy src/fleetlab/film tests/test_film_framing.py tests/test_film_framing_sim.py
  tools/heavy-run --light -m 2G uv run --locked pytest src/fleetlab/film/tests tests/test_film_framing.py tests/test_film_framing_sim.py
  ```

## Investigations

The investigating agent is never given the answer. Scenario answer keys, the
simulator's measured response curves and the host-fixed checks stay outside
its sandbox; statistical and ordinary robotics knowledge belongs in its
starting skills under [`skills/`](skills). Every clip and number shown from a
run resolves to a recorded run.
