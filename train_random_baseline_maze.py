"""
Random policy baseline on ObstacleMazeEnv.
Evaluates performance of uniform random actions w.r.t. checkpoint adherence.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

from core.obstacle_maze_env import ObstacleMazeEnv


def calculate_first_visits(visited_sequence: List[int]) -> List[int]:
    seen = set()
    first_visits = []
    for cp in visited_sequence:
        if cp not in seen:
            seen.add(cp)
            first_visits.append(cp)
    return first_visits


def adherence_rate(first_visits: List[int], workflow: List[int]) -> float:
    matches = 0
    for i in range(min(len(first_visits), len(workflow))):
        if first_visits[i] == workflow[i]:
            matches += 1
        else:
            break
    return matches / len(workflow)


def run_episode(env: ObstacleMazeEnv, max_steps: int, workflow: List[int]) -> Tuple[float, List[int], float, bool]:
    """Run one episode with random actions and collect metrics."""
    env.reset()
    visited_sequence: List[int] = []
    ep_return = 0.0

    for _ in range(max_steps):
        action = env.action_space.sample()
        _, reward, done, _ = env.step(int(action))
        ep_return += float(reward)

        pos = tuple(env.agent_pos)
        for cp_idx in range(env.num_checkpoints):
            if env._in_checkpoint(pos, cp_idx) and cp_idx not in visited_sequence:
                visited_sequence.append(cp_idx)
        if done:
            break

    first_visits = calculate_first_visits(visited_sequence)
    adherence = adherence_rate(first_visits, workflow)
    success = first_visits == workflow
    return ep_return, first_visits, adherence, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--wall_density", type=float, default=0.15)
    parser.add_argument("--exp_name", type=str, default="random_baseline_maze")
    args = parser.parse_args()

    workflow = [0, 1, 2, 3]

    os.makedirs("logs", exist_ok=True)
    run_dir = os.path.join("logs", f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(
            {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "wall_density": args.wall_density,
                "exp_name": args.exp_name,
            },
            f,
            indent=2,
        )

    csv_path = os.path.join(run_dir, "episodes.csv")
    with open(csv_path, "w") as f:
        f.write("episode,return,adherence,success\n")

    env = ObstacleMazeEnv(max_steps=args.max_steps, wall_density=args.wall_density)

    returns = []
    adherences = []
    successes = []

    for ep in range(args.episodes):
        ep_return, visited_seq, adherence, success = run_episode(env, args.max_steps, workflow)
        returns.append(ep_return)
        adherences.append(adherence)
        successes.append(1.0 if success else 0.0)

        print(f"Episode {ep:4d} | Return {ep_return:7.2f} | Adherence {adherence:5.1%} | Sequence {visited_seq}")
        with open(csv_path, "a") as f:
            f.write(f"{ep},{ep_return:.4f},{adherence:.4f},{int(success)}\n")

    summary = {
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_adherence": float(np.mean(adherences)) if adherences else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Random baseline evaluation finished.")
    print(f"Summary: {summary}")
    print(f"Logs saved to {run_dir}")


if __name__ == "__main__":
    main()

