"""LeRobot's own evaluation, `lerobot-eval`, in this environment: the reference.

    tools/heavy-run uv run --locked python -m fleetlab.reference_eval [--no-freeze] FLAGS...

FLAGS are `lerobot-eval`'s and reach it unchanged, with one default changed:
the environments are simulated in this process (`--eval.use_async_envs=false`)
unless the flag is given. LeRobot's default is worker processes, which the
compatibility layer below does not reach and which crashed on this machine
(`spikes/baseline`). The baseline policy on one task:

    --policy.path=lerobot/xvla-libero --env.type=libero --env.control_mode=absolute \\
    --env.task=libero_spatial --env.task_ids='[0]' --eval.n_episodes=10 --eval.batch_size=5 \\
    --seed=1000 --output_dir=data/reference/libero_spatial_0

LeRobot's files are not changed. What this adds around them:

- `fleetlab.sim.lerobot_compat`, without which LeRobot's LIBERO wrapper does
  not run on robosuite 1.5.
- LeRobot's own `FreezeAfterEpisodeEnd` around every environment. LeRobot
  applies it in its default runner, the worker processes, but not in the
  in-process runner used here. With it, as in LeRobot as it comes, a finished
  environment stops simulating and the episodes of a task begin from the
  stored start states 0, 1, 2 and so on.
- `--no-freeze` leaves that wrapper out. LeRobot runs a task in rounds of
  `eval.batch_size` environments. Without the wrapper an environment that
  finishes before its round does is reset, plays on uncounted, and every such
  reset moves it on by `batch_size` start states, so from the second round on
  the start states depend on how the earlier episodes went. That
  is how `spikes/baseline` measured and how the earlier reference runs
  were made (X-VLA, 98 of 100 on `libero_spatial` and on `libero_object`).
  Repeating those needs `--no-freeze`.
- In `eval_info.json`, every task's metrics gain `init_states`: the stored
  start state each counted episode began from, beside `successes`.

Nothing on this path switches rendering contexts. The images are right only
because `lerobot-eval` has the simulators of one task alive at a time.
Different tasks in one vector environment, or LeRobot's wrapper and our
adapter alive together in one process, would render wrong images without an
error.

Loads a model onto the GPU, so run it through `tools/heavy-run`.
"""

from __future__ import annotations

import sys
from typing import Any

from fleetlab.runtime import cap_gpu_memory, configure_process


def record_start_states(lerobot_eval: Any) -> None:
    """Make `eval_info.json` say which stored start state each counted episode began from."""
    rollout, eval_one = lerobot_eval.rollout, lerobot_eval.eval_one
    started: dict[int, list[int | None]] = {}  # per vector environment, in episode order

    def rollout_noting_starts(env: Any, *args: Any, **kwargs: Any) -> Any:
        # Read before the reset that begins the round: an environment's reset loads
        # the state its counter points at and then moves the counter on.
        counters, stored = env.call("init_state_id"), env.call("_init_states")
        started.setdefault(id(env), []).extend(
            None if states is None else counter % len(states)
            for counter, states in zip(counters, stored, strict=True)
        )
        return rollout(env, *args, **kwargs)

    def eval_one_with_starts(env: Any, **kwargs: Any) -> Any:
        metrics = eval_one(env, **kwargs)
        # LeRobot counts the first `n_episodes` of the rounds it ran.
        metrics["init_states"] = started.pop(id(env))[: len(metrics["successes"])]
        return metrics

    lerobot_eval.rollout = rollout_noting_starts
    lerobot_eval.eval_one = eval_one_with_starts


def freeze_finished_environments() -> None:
    """Wrap the in-process environments as LeRobot wraps those in its worker processes."""
    from lerobot.envs import libero
    from lerobot.envs.utils import freeze_after_episode_end

    make_env_fns = libero._make_env_fns

    def make_freezing_env_fns(**kwargs: Any) -> list[Any]:
        return [freeze_after_episode_end(make) for make in make_env_fns(**kwargs)]

    libero._make_env_fns = make_freezing_env_fns


def lerobot_eval_arguments(arguments: list[str]) -> tuple[list[str], bool]:
    """Returns what goes on to `lerobot-eval`, and whether to freeze finished environments."""
    passed_on = [argument for argument in arguments if argument != "--no-freeze"]
    if not any(argument.startswith("--eval.use_async_envs") for argument in passed_on):
        passed_on.append("--eval.use_async_envs=false")
    return passed_on, "--no-freeze" not in arguments


def main() -> None:
    sys.argv[1:], freeze = lerobot_eval_arguments(sys.argv[1:])
    configure_process()  # before the imports below, which read the environment

    from lerobot.scripts import lerobot_eval

    from fleetlab.sim import lerobot_compat

    cap_gpu_memory()
    lerobot_compat.install()
    record_start_states(lerobot_eval)
    if freeze:
        freeze_finished_environments()
    lerobot_eval.main()


if __name__ == "__main__":
    main()
