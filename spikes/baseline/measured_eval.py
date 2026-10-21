"""lerobot-eval with measurements added. Same command-line flags as lerobot-eval.

It runs LeRobot's own evaluation unchanged and records, next to eval_info.json,
a file measure.json with: wall time, episode lengths in environment steps, time
spent stepping the simulator, number of model calls, peak process memory and
peak GPU memory.

Environment variables (all optional):
  BASELINE_NO_VIDEO=1   do not render or encode episode videos (LeRobot always
                        renders the first 10 episodes of every task otherwise,
                        which re-renders the cameras once more per step)
  BASELINE_EXPECT_GIB   GPU memory this run is expected to add (default 8). The
                        run does not start while that would push the system
                        total above BASELINE_GTT_LIMIT_GIB (default 20), and
                        PyTorch is capped at this amount.
  BASELINE_ABS_WORLD=1  robosuite 1.5 port only, for policies that command
                        absolute poses (X-VLA): interpret absolute commands in
                        the world frame, as robosuite 1.4 does. Without it the
                        port reads them in the robot-base frame and the arm
                        flies off (see probe_absolute_control.py).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

T0 = time.time()
GTT = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
VRAM = Path("/sys/class/drm/card1/device/mem_info_vram_used")


def _read(p: Path) -> int:
    try:
        return int(p.read_text())
    except OSError:
        return -1


# --- machine safety: do not load a model into a GPU memory pool that is already full ---
# The GPU shares system memory (limit 31 GiB) and the machine hangs when it
# fills. BASELINE_EXPECT_GIB is what this run is expected to add; the run waits
# up to 30 seconds for other users of the GPU to finish and otherwise gives up
# (it holds the heavy-job lock while it waits, so the wait is kept short).
_expect = float(os.environ.get("BASELINE_EXPECT_GIB", "8"))
_limit = float(os.environ.get("BASELINE_GTT_LIMIT_GIB", "20"))
for _i in range(30):
    _now = _read(GTT) / 2**30
    if _now + _expect <= _limit:
        break
    time.sleep(1)
else:
    print(
        f"MEASURE-ABORT GPU memory in use is {_now:.1f} GiB; adding the expected {_expect:.1f} GiB "
        f"would exceed {_limit:.0f} GiB. Not starting.",
        flush=True,
    )
    sys.exit(3)
T0 = time.time()  # do not count the wait above as run time

stats = {
    "gtt_start": _read(GTT),
    "gtt_peak": _read(GTT),
    "vram_start": _read(VRAM),
    "vram_peak": _read(VRAM),
    "episode_steps": [],
    "rollout_s": 0.0,
    "env_step_s": 0.0,
    "env_step_calls": 0,
    "env_reset_s": 0.0,
    "model_calls": 0,
    "model_s": 0.0,
    "select_action_calls": 0,
}
_stop = threading.Event()


def _own_gpu_bytes() -> dict:
    """GPU memory held by this process, from the kernel's per-file accounting.

    The system-wide counter above also contains whatever other jobs and the
    desktop hold, so it only bounds this process from above. Each open GPU
    device file reports its own totals; one client can appear under several
    file descriptors, so count each drm-client-id once.
    """
    seen, total = set(), {}
    try:
        for f in os.listdir("/proc/self/fdinfo"):
            try:
                text = Path(f"/proc/self/fdinfo/{f}").read_text()
            except OSError:
                continue
            if "drm-client-id" not in text:
                continue
            fields = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip()
            cid = fields.get("drm-client-id")
            if cid in seen:
                continue
            seen.add(cid)
            for k, v in fields.items():
                if k.startswith(("drm-memory-", "drm-total-", "drm-resident-")) and v.endswith(("KiB", "MiB", "GiB")):
                    num, unit = v.split()
                    total[k] = total.get(k, 0) + int(num) * {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[unit]
    except Exception:  # never let accounting break a run
        pass
    return total


stats["own_gpu_peak"] = {}


def _poll():
    while not _stop.wait(0.25):
        stats["gtt_peak"] = max(stats["gtt_peak"], _read(GTT))
        stats["vram_peak"] = max(stats["vram_peak"], _read(VRAM))
        for k, v in _own_gpu_bytes().items():
            if v > stats["own_gpu_peak"].get(k, 0):
                stats["own_gpu_peak"][k] = v


threading.Thread(target=_poll, daemon=True).start()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lerobot.scripts.lerobot_eval as le  # noqa: E402

# PyTorch may not take more GPU memory than the run was expected to add in total
# (tools/heavy-run asks model-loading processes to cap themselves).
try:
    if torch.cuda.is_available():
        _total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, _expect * 2**30 / _total))
except Exception as _e:  # a cap that cannot be set must not stop the run
    print(f"MEASURE-NOTE could not cap PyTorch GPU memory: {_e}", flush=True)

# --- episode lengths and rollout time -------------------------------------
_orig_rollout = le.rollout


def _rollout(*args, **kwargs):
    t = time.time()
    ret = _orig_rollout(*args, **kwargs)
    stats["rollout_s"] += time.time() - t
    done = ret["done"].to(int)  # (batch, steps), cumulative
    first_done = torch.argmax(done, dim=1) + 1
    never = done.sum(dim=1) == 0
    first_done[never] = done.shape[1]
    stats["episode_steps"].extend(int(x) for x in first_done)
    # Which stored start state each environment would use next. In the synchronous
    # runner this moves on by more than one round's worth when finished
    # environments were reset and re-run inside the round.
    try:
        vec = kwargs.get("env", args[0] if args else None)
        stats.setdefault("next_init_state_after_rollout", []).append([int(x) for x in vec.get_attr("init_state_id")])
    except Exception:
        pass
    return ret


le.rollout = _rollout

# --- simulator time -------------------------------------------------------
for _cls in (gym.vector.SyncVectorEnv,):
    _step, _reset = _cls.step, _cls.reset

    def step(self, actions, _step=_step):
        t = time.time()
        out = _step(self, actions)
        stats["env_step_s"] += time.time() - t
        stats["env_step_calls"] += 1
        return out

    def reset(self, *a, _reset=_reset, **k):
        t = time.time()
        out = _reset(self, *a, **k)
        stats["env_reset_s"] += time.time() - t
        return out

    _cls.step, _cls.reset = step, reset

# --- model calls ----------------------------------------------------------
_orig_make_policy = le.make_policy


def _make_policy(*args, **kwargs):
    policy = _orig_make_policy(*args, **kwargs)
    sel = policy.select_action

    def select_action(*a, **k):
        stats["select_action_calls"] += 1
        return sel(*a, **k)

    policy.select_action = select_action
    # The expensive part is the chunk prediction; its name differs per policy.
    for name in ("_get_action_chunk", "predict_action_chunk"):
        if hasattr(policy, name):
            inner = getattr(policy, name)

            def timed(*a, _inner=inner, **k):
                t = time.time()
                out = _inner(*a, **k)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stats["model_s"] += time.time() - t
                stats["model_calls"] += 1
                return out

            setattr(policy, name, timed)
            stats["model_call_fn"] = name
            break
    stats["n_action_steps"] = getattr(policy.config, "n_action_steps", None)
    stats["policy_type"] = getattr(policy.config, "type", None)
    stats["policy_params_m"] = round(sum(p.numel() for p in policy.parameters()) / 1e6, 1)
    stats["policy_dtype"] = str(next(policy.parameters()).dtype)
    return policy


le.make_policy = _make_policy

# --- optional: world-frame absolute commands on the robosuite 1.5 port ----
if os.environ.get("BASELINE_ABS_WORLD") == "1":
    import lerobot.envs.libero as ll

    _orig_set_delta = ll._set_delta

    def _set_delta(robot, use_delta: bool) -> None:
        _orig_set_delta(robot, use_delta)
        ctrl = ll._arm_controller(robot)
        if not use_delta and hasattr(ctrl, "input_ref_frame"):
            ctrl.input_ref_frame = "world"

    ll._set_delta = _set_delta

# --- optional: no videos --------------------------------------------------
if os.environ.get("BASELINE_NO_VIDEO") == "1":
    _orig_all = le.eval_policy_all

    def _eval_policy_all(*args, **kwargs):
        kwargs["max_episodes_rendered"] = 0
        kwargs["videos_dir"] = None
        return _orig_all(*args, **kwargs)

    le.eval_policy_all = _eval_policy_all


def _out_dir() -> Path | None:
    for a in sys.argv:
        if a.startswith("--output_dir="):
            return Path(a.split("=", 1)[1])
    return None


def _finish(exit_code):
    _stop.set()
    hwm = -1
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    steps = stats["episode_steps"]
    wall = time.time() - T0
    import mujoco
    import robosuite

    out = {
        "exit_code": exit_code,
        "mujoco": mujoco.__version__,
        "robosuite": robosuite.__version__,
        "abs_world": os.environ.get("BASELINE_ABS_WORLD") == "1",
        "argv": sys.argv[1:],
        "no_video": os.environ.get("BASELINE_NO_VIDEO") == "1",
        "wall_s": round(wall, 1),
        "episodes": len(steps),
        "env_steps": sum(steps),
        "rollout_s": round(stats["rollout_s"], 1),
        "s_per_episode_rollout": round(stats["rollout_s"] / max(len(steps), 1), 2),
        "s_per_episode_wall": round(wall / max(len(steps), 1), 2),
        "env_steps_per_s_rollout": round(sum(steps) / max(stats["rollout_s"], 1e-9), 1),
        "env_steps_per_s_wall": round(sum(steps) / wall, 1),
        "sim_step_s": round(stats["env_step_s"], 1),
        "sim_reset_s": round(stats["env_reset_s"], 1),
        "sim_batched_step_calls": stats["env_step_calls"],
        "model_s": round(stats["model_s"], 1),
        "model_calls": stats["model_calls"],
        "model_ms_per_call": round(1000 * stats["model_s"] / max(stats["model_calls"], 1), 1),
        "model_call_fn": stats.get("model_call_fn"),
        "select_action_calls": stats["select_action_calls"],
        "n_action_steps": stats.get("n_action_steps"),
        "policy_type": stats.get("policy_type"),
        "policy_params_m": stats.get("policy_params_m"),
        "policy_dtype": stats.get("policy_dtype"),
        "peak_rss_gib": round(hwm / 2**30, 2),
        "torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "own_gpu_peak_gib": {k: round(v / 2**30, 2) for k, v in sorted(stats["own_gpu_peak"].items())},
        "gtt_start_gib": round(stats["gtt_start"] / 2**30, 2),
        "gtt_peak_gib": round(stats["gtt_peak"] / 2**30, 2),
        "gtt_peak_delta_gib": round((stats["gtt_peak"] - stats["gtt_start"]) / 2**30, 2),
        "vram_peak_delta_gib": round((stats["vram_peak"] - stats["vram_start"]) / 2**30, 2),
        "episode_steps": steps,
        "next_init_state_after_rollout": stats.get("next_init_state_after_rollout"),
    }
    d = _out_dir()
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "measure.json").write_text(json.dumps(out, indent=1))
    short = {k: v for k, v in out.items() if k not in ("episode_steps", "argv")}
    print("MEASURE " + json.dumps(short), flush=True)


if __name__ == "__main__":
    code = 0
    try:
        le.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        raise
    except BaseException:
        code = 1
        raise
    finally:
        _finish(code)
