"""Expected episodes per hour for the recommended layout, from component times.

A small model to recompute the throughput for another policy or setting.
Against the three measured prototype runs in the README it was 6%, 14% and 26%
too high: it is optimistic, the more so the more other jobs use the GPU.

The layout: N simulation workers, one environment each, and one inference
process that batches whatever requests are waiting.

A worker's episode costs
    reset  +  steps * step_ms  +  requests * wait
where requests = ceil(steps / actions_per_chunk) and wait is the time from
sending an observation to getting the action chunk back. The wait depends on
how busy the inference process is, which depends on how many requests all
workers send, which depends on the wait: the script solves that loop.

    python spikes/throughput/estimate.py --workers 8 --step-ms 8.5 --actions-per-chunk 50
"""

from __future__ import annotations

import argparse
import math


def solve(workers: int, step_ms: float, reset_s: float, steps: float, k: int,
          call_base_ms: float, call_per_env_ms: float, overhead_ms: float, contention: float = 1.0) -> dict:
    """Call time for a batch of b = contention * (call_base_ms + call_per_env_ms * b) + overhead_ms.

    contention: how much longer a call takes while the workers render on the same GPU.
    overhead_ms: preprocessing, postprocessing and transport per call, on the CPU.
    """
    call_base_ms, call_per_env_ms = contention * call_base_ms, contention * call_per_env_ms
    requests = math.ceil(steps / k)
    sim_s = reset_s + steps * step_ms / 1000
    wait = (call_base_ms + call_per_env_ms + overhead_ms) / 1000
    batch = 1.0
    for _ in range(500):
        episode_s = sim_s + requests * wait
        rate = workers * requests / episode_s  # requests per second arriving at the inference process
        a, c = (call_base_ms + overhead_ms) / 1000, call_per_env_ms / 1000
        if rate * c >= 0.999:  # the GPU cannot keep up at any batch size: everyone waits in one big batch
            batch = float(workers)
        else:
            # Requests that arrive while one batch is being served form the next batch.
            batch = min(max(rate * a / (1 - rate * c), 1.0), float(workers))
        service = a + c * batch
        busy = min(rate * service / batch, 1.0)
        # Wait = own batch, plus on average half of the batch in progress when the server is busy.
        new_wait = service * (1 + 0.5 * busy)
        if abs(new_wait - wait) < 1e-7:
            break
        wait = 0.5 * wait + 0.5 * new_wait
    episode_s = sim_s + requests * wait
    return {
        "episodes_per_hour": 3600 * workers / episode_s,
        "episode_s_per_worker": episode_s,
        "requests_per_episode": requests,
        "wait_ms_per_request": 1000 * wait,
        "mean_batch": batch,
        "inference_busy_share": busy,
        "worker_share_simulating": (steps * step_ms / 1000) / episode_s,
        "worker_share_waiting": requests * wait / episode_s,
        "worker_share_resetting": reset_s / episode_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--step-ms", type=float, default=8.5, help="one environment step in a worker, at this worker count")
    ap.add_argument("--reset-s", type=float, default=0.75, help="reset between episodes, including settle steps")
    ap.add_argument("--steps", type=float, default=160, help="mean environment steps per episode")
    ap.add_argument("--actions-per-chunk", type=int, default=50)
    ap.add_argument("--call-base-ms", type=float, default=65, help="inference call: fixed part")
    ap.add_argument("--call-per-env-ms", type=float, default=43, help="inference call: per environment in the batch")
    ap.add_argument("--overhead-ms", type=float, default=35, help="preprocessing, postprocessing, transport per call")
    ap.add_argument("--contention", type=float, default=1.35,
                    help="factor on the call time while workers render on the same GPU (measured 1.35 to 1.5)")
    args = ap.parse_args()
    r = solve(args.workers, args.step_ms, args.reset_s, args.steps, args.actions_per_chunk,
              args.call_base_ms, args.call_per_env_ms, args.overhead_ms, args.contention)
    print(f"{r['episodes_per_hour']:.0f} episodes/hour "
          f"({r['episode_s_per_worker']:.2f} s per episode per worker, {r['requests_per_episode']} inference requests "
          f"per episode, {r['wait_ms_per_request']:.0f} ms wait each, mean batch {r['mean_batch']:.1f}, "
          f"inference busy {100 * r['inference_busy_share']:.0f}% of the time; worker time: "
          f"{100 * r['worker_share_simulating']:.0f}% simulating, {100 * r['worker_share_waiting']:.0f}% waiting, "
          f"{100 * r['worker_share_resetting']:.0f}% resetting)")


if __name__ == "__main__":
    main()
