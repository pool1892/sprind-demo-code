# Third-party code carried in this repository

## libero

`libero/` is the package `hf-libero` 0.1.4, Hugging Face's packaging of the
LIBERO benchmark, with a port to robosuite 1.5 and two fixes of our own
applied. The environment installs it from here as an editable path dependency
(see `pyproject.toml`), so the files in this directory are the ones that run.

It is carried here because upstream pins `robosuite==1.4.0` and the port is not
released anywhere: the build must not depend on a fork staying online.

**Origin.** The source distribution `hf_libero-0.1.4.tar.gz` from PyPI, sha256
`c058d67ad5a2b589529c14d614282ef4cca3a7763dafa134f58a6c9039657e34`. Three
things the sdist build generates are left out: `PKG-INFO`, `setup.cfg` and
`hf_libero.egg-info/`.

**Changes.** Two patch files, applied in this order, are everything that
differs from the sdist. They are kept apart so that each step can be checked on
its own: the first against the upstream pull request, the second by reading it.

1. [`libero-robosuite-1.5.patch`](libero-robosuite-1.5.patch), the port:

   - Six files under `libero/libero/envs/`, taken unchanged from pull request
     3 of `shuheng-liu/LIBERO`, "feat: port LIBERO to robosuite 1.5" (head
     commit `64657da1a8b324af2a4e5ebda12ef71a1a2e912d`, merged 2026-06-03).
     The pull request also touches `requirements.txt` and two files under
     `scripts/`, which the PyPI package does not contain.
   - `pyproject.toml`: `robosuite==1.4.0` becomes `robosuite==1.5.2`.

2. [`libero-reset.patch`](libero-reset.patch), ours, fourteen added lines
   marked `fleetlab:`. Without them a reset with the same seed and the same stored
   start state does not give the same scene: the furniture
   lands up to 2 cm elsewhere, depending on what the process did before.

   - `env_wrapper.py`, `ControlEnv.seed()`: also seed NumPy's global random
     generator, from which LIBERO draws where the furniture stands. On
     robosuite 1.4 `seed()` did that. robosuite 1.5 keeps the seed in an
     attribute of the same name, which hides LIBERO's method, and the port
     then seeds a generator that LIBERO's own samplers do not use. One
     difference to 1.4 is deliberate: `seed(None)` leaves the generator
     alone, where 1.4 restarted it from the clock. Gymnasium resets a
     finished environment that way in the middle of an evaluation, and a
     whole run of `lerobot-eval` should repeat, as it did before this patch.
   - `bddl_base_domain.py`, `_setup_placement_initializer()`: start with an
     empty list of drawer and switch samplers each time the model is loaded.
     Upstream appends them again at every hard reset, and each one draws a
     random number before the furniture is placed, so the scene depended on
     how many resets an environment had seen. This is LIBERO's own defect, not
     the port's, and should exist on robosuite 1.4 as well (not measured there).

   The fixes are here and not in our own code because LeRobot's LIBERO
   wrapper, which the reference evaluation runs unmodified, calls exactly
   these two functions: fixed at the source, it is repeatable too.
   `tests/test_sim_reset.py` checks the result.

**Licence.** MIT, see [`libero/LICENSE`](libero/LICENSE). The fork the port
comes from is MIT as well.

**Checked.** With SmolVLA the port scores the same as the pinned robosuite
1.4.0 within noise, on `libero_spatial` and `libero_10`.
Other suites and policies were not compared. That was before the second patch,
which changes where the furniture stands in a given episode, not how the
furniture is placed.

Success rates measured here can be compared with published figures as rates.
Episodes cannot be compared one to one with robosuite 1.4, for two reasons:

- Where the furniture stands for a given seed differs. On robosuite 1.4,
  `Robot.reset` draws seven numbers from NumPy's global generator (the noise
  on the arm's start pose, of magnitude zero in LIBERO) before LIBERO places
  the furniture from the same generator. robosuite 1.5 takes those seven from
  a generator of its own. Read from the code of both versions, not measured.
- The second patch does two things that upstream LIBERO on 1.4 does not: the
  list of drawer and switch samplers no longer grows with every hard reset,
  and `seed(None)` leaves the generator alone.

**Not fixed here.** Absolute control: robosuite 1.5 reads target poses in the
frame of the robot's base, 1.4 read them in world coordinates. LIBERO itself
never switches the control mode; its users do, so the frame is set where the
mode is set, in `src/fleetlab/sim/libero_env.py`.

**Not included.** The meshes and textures. LIBERO downloads them from the
Hugging Face dataset `lerobot/libero-assets` into `~/.cache/libero/assets` the
first time an environment is built.

**To verify or redo.** Unpack the sdist, delete the three generated entries,
apply the two patches and compare. After the first patch the six files are the
pull request's; after the second the tree is this directory.

```sh
tar -xzf hf_libero-0.1.4.tar.gz && cd hf_libero-0.1.4
rm -r PKG-INFO setup.cfg hf_libero.egg-info
patch -p1 < /path/to/repo/third_party/libero-robosuite-1.5.patch
patch -p1 < /path/to/repo/third_party/libero-reset.patch
diff -r -x hf_libero.egg-info -x __pycache__ . /path/to/repo/third_party/libero
```
