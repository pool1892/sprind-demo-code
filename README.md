# Agents that engineer a robot installation

This repository holds the source of a simulation demo in which AI engineering agents
improve a robot installation whose policy never changes. The film and the records of
the demo are on the demo page: **https://pool1892.github.io/sprind-demo-page/**.

## What the demo shows

A simulated robot arm runs a pretrained vision-language-action policy, X-VLA, on LIBERO
tasks in the MuJoCo physics simulator. Not one weight of the policy changes. The agents
work on everything around it: how the robot's skills are sequenced, how the scene is
prepared, and which settings, software versions and services the installation uses.

- **New jobs from existing skills.** The robot has a skill for each half of a job and
  none for the whole. An engineering agent finds out why the plain chain of skills
  fails, writes checks over the run records, tests one change at a time and releases a
  job program. Two jobs: the bowl on the cabinet with the middle drawer open, and the
  plate at the front of the stove with the bowl on the cabinet. On 20 reserved starts,
  held back until the end, both jobs delivered 20 of 20; the plain chains delivered 1
  and 5. The jobs, their checks and the agent's learned skills stay in the
  installation's library.
- **An instrument the agent builds.** A fleet of arms at several sites shows the same
  lag at one site and on one robot elsewhere. The agent composes its own step-response
  probe, commanded against measured motion, and tells a fixed network delay from a
  controller falling behind. After paired experiments it compensates for the delay,
  rolls back an update and services the controller. On 576 reserved attempts it never
  saw, success rose from 451 to 558: 113 rescued, 6 broken.

The agents are never given the answer. The meaning of success, the reserved starts and
the ground truth of the injected faults stay with the host, outside the agents'
sandbox. Every clip in the film and every number on the page resolves to a recorded
run; these are counts from tens to hundreds of runs in one simulator.

## How it is built

- **Simulator and policy.** MuJoCo for physics and rendering, LIBERO for the scenes
  and tasks (carried in [`third_party/libero`](third_party/libero) with a port to
  robosuite 1.5 and a fix for exact resets), and X-VLA from LeRobot
  (`lerobot/xvla-libero`) as the fixed policy. Runs are exact: an episode replays bit
  for bit from its seed and start state.
- **Faults and fleets.** Physical, configuration, software and site-specific faults
  are injected on purpose into simulated fleets, with measured dose-response curves.
  An oracle establishes the ground truth by replaying episodes with and without each
  fault.
- **Agents.** The investigating agents run as sandboxed command-line agents with a
  tool server over the run records (queries, experiments, checks, job programs) and a
  library of starting skills. Statistical and ordinary robotics knowledge is in the
  skills; answers are not.
- **Jobs.** The jobs runtime chains the robot's goal skills into one job in one
  simulator state, judges delivery with a host-fixed check and keeps released jobs,
  checks and learned skills in the installation's library.
- **Film and page.** The film, its narration and the static page are built from the
  recorded runs, with every clip, count and output listed by SHA-256.

## Repository map

| Path | What |
| --- | --- |
| [`src/fleetlab/contracts`](src/fleetlab/contracts) | The interfaces of the test bed ([notes](docs/design/contracts.md)) |
| [`src/fleetlab/sim`](src/fleetlab/sim), [`src/fleetlab/policies`](src/fleetlab/policies) | The LIBERO environment and the policy adapters |
| [`src/fleetlab/faults`](src/fleetlab/faults) | The fault library and its calibration ([notes](docs/design/faults.md)) |
| [`src/fleetlab/worlds`](src/fleetlab/worlds), [`src/fleetlab/engine`](src/fleetlab/engine) | Fleet composition and the rollout engine |
| [`src/fleetlab/oracle`](src/fleetlab/oracle) | Ground truth by replay |
| [`src/fleetlab/log`](src/fleetlab/log) | The episode log |
| [`src/fleetlab/agents`](src/fleetlab/agents), [`src/fleetlab/toolserver`](src/fleetlab/toolserver), [`src/fleetlab/audit`](src/fleetlab/audit) | The agent runtime and its sandbox, the agents' tool server, the leak audit |
| [`src/fleetlab/jobs`](src/fleetlab/jobs) | The jobs runtime ([notes](docs/design/jobs.md)) |
| [`src/fleetlab/loop`](src/fleetlab/loop) | The improvement loop over a fleet ([notes](docs/design/loop.md)) |
| [`src/fleetlab/live`](src/fleetlab/live) | A live layer that stops a failing attempt while it runs |
| [`src/fleetlab/baselines`](src/fleetlab/baselines), [`src/fleetlab/scoring`](src/fleetlab/scoring) | Baselines and the scoring of development worlds |
| [`src/fleetlab/film`](src/fleetlab/film) | The film, its narration and the static page |
| [`skills`](skills) | The agents' starting skills |
| [`site`](site) | Sources of the demo page: film manifests, narration, record pages, templates |
| [`spikes`](spikes) | Development measurements, each with a README |
| [`docs`](docs) | Technical notes and measured results |
| [`tests`](tests) | Unit tests and simulator tests |
| [`tools`](tools) | The job queue for simulator and GPU work, the nightly worlds, narration |

## Setup

Four commands, on a machine as described below:

- `make setup` builds `.venv` from `uv.lock`. It proves that the environment
  can be rebuilt from what is in this repository.
- `make check` runs lint, the type check and the unit tests, without a GPU. It
  proves that the code is sound and that the installed packages are the
  combination the lock file describes.
- `make check-sim` runs the tests that build LIBERO scenes and render them,
  without a model, in about a minute and a half. It proves what an exact
  replay rests on: a reset with the same seed and start state gives the same
  simulator state and images, absolute and relative control both do what they
  are told, two scenes in one process each render their own images, and two
  scenes whose start depends on the MuJoCo version start as they should.
- `make smoke` runs one LIBERO episode headless with the baseline policy,
  X-VLA, on the GPU and writes `data/smoke/episode.mp4`. It proves that
  simulator, renderer, policy and GPU work together. It does not prove that
  the policy solves the task.

LeRobot's own evaluation runs in the same environment, as the reference for
our engine: `tools/heavy-run uv run --locked python -m fleetlab.reference_eval`
followed by the flags of `lerobot-eval`. The module's docstring has an example
and says where the launcher differs from plain `lerobot-eval`.

What they need from the machine:

- Linux on x86-64, `uv` 0.12.23 or newer and GNU Make. uv brings Python 3.12.
- For `make setup`, network access to PyPI and to AMD's ROCm index: about
  2.1 GB of downloads, 7.2 GB installed.
- A C and a C++ compiler. The lock builds seven packages from source. Five are
  pure Python (`rocm`, `robomimic`, `bddl`, `antlr4-python3-runtime`,
  `docopt`). `hf-egl-probe` compiles two C++ programs with CMake and `make`;
  uv supplies CMake, and the package carries its own EGL headers. `evdev`
  compiles a C extension against the Linux input headers (`linux/input.h`,
  `linux/input-event-codes.h`, `linux/uinput.h`).
- For `make check-sim` and `make smoke`, a Radeon that the ROCm `gfx1151`
  wheels support and an EGL library for off-screen rendering. `tools/heavy-run`
  needs `systemd-run` with a user session, and `flock`.
- The first `make smoke` downloads the checkpoint `lerobot/xvla-libero` and
  the tokenizer of `facebook/bart-large` into the Hugging Face cache
  (`~/.cache/huggingface`, about 3.3 GB). The first use of the simulator
  downloads LIBERO's meshes and textures into `~/.cache/libero/assets`
  (0.4 GB).

The demo ran on a single workstation: an AMD Ryzen AI Max+ 395 with an integrated
Radeon 8060S that shares the machine's 62 GiB of memory, rendering headless through
EGL, with PyTorch on ROCm. There is no CUDA path. [AGENTS.md](AGENTS.md) has the
working rules for the simulator and GPU job queue.

## Records and history

The run records behind the demo (videos, logs, agent transcripts) are kept outside the
repository, under `~/.cache/robots-probes`, and are not part of it. The demo page
publishes the film's clips with their SHA-256 checksums.

This repository is a snapshot of the development repository, taken on 21 October 2026.
Commit ids and frozen tags cited in the module READMEs, the spike READMEs and the run
records refer to that development history, which is not published.

## Licence

No licence is granted. The source is published to be read; all rights are reserved.
[`third_party/libero`](third_party/libero) keeps its own MIT licence.
