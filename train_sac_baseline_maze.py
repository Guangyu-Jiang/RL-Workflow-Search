"""
Parallel Soft Actor-Critic baseline on ObstacleMazeEnv using env-only rewards.
Episodes are collected in parallel worker processes, mirroring the high-throughput
setup from the CybORG baseline while targeting the maze workflow task.
"""

import argparse
import csv
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.obstacle_maze_env import ObstacleMazeEnv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Experience replay buffer backed by deque."""

    def __init__(self, max_size: int = 200_000):
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self._buffer[i] for i in idx))
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class SACActorCritic(nn.Module):
    """Actor + twin Q networks with smaller hidden layers for the maze task."""

    def __init__(self, input_dim: int = 14, n_actions: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )


def _first_visits(sequence: List[int]) -> List[int]:
    seen = set()
    visits = []
    for idx in sequence:
        if idx not in seen:
            seen.add(idx)
            visits.append(idx)
    return visits


def _adherence(first_visits: List[int], workflow: List[int]) -> float:
    matches = 0
    for i, target in enumerate(workflow):
        if i < len(first_visits) and first_visits[i] == target:
            matches += 1
        else:
            break
    return matches / len(workflow)


def collect_episode(worker_id: int,
                    policy_weights: Dict[str, torch.Tensor],
                    max_steps: int,
                    wall_density: float,
                    seed_base: int) -> Dict:
    """Collect a single episode of experience using the provided policy weights."""
    torch.set_num_threads(1)
    device_local = torch.device("cpu")

    env = ObstacleMazeEnv(max_steps=max_steps, wall_density=wall_density)
    env.reset(seed=seed_base + worker_id)

    state_dim = 2 + 4 * 2 + 4  # agent (2) + checkpoint centers (8) + visited flags (4)
    policy = SACActorCritic(state_dim, 4, hidden_dim=128).to(device_local)
    policy.load_state_dict(policy_weights)
    policy.eval()

    transitions = []
    visited_sequence = []
    total_reward = 0.0

    state = env.get_state_for_policy()
    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device_local).unsqueeze(0)
        with torch.no_grad():
            logits = policy.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs=probs).sample().item()

        _, reward, done, _ = env.step(int(action))
        next_state = env.get_state_for_policy()

        pos = tuple(env.agent_pos)
        for cp_idx in range(env.num_checkpoints):
            if env._in_checkpoint(pos, cp_idx) and cp_idx not in visited_sequence:
                visited_sequence.append(cp_idx)

        transitions.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

        total_reward += float(reward)
        state = next_state
        if done:
            break

    first = _first_visits(visited_sequence)
    return {
        "transitions": transitions,
        "total_reward": total_reward,
        "steps": len(transitions),
        "adherence": _adherence(first, [0, 1, 2, 3]),
        "success": first == [0, 1, 2, 3],
    }


class ParallelSAC:
    """SAC agent orchestrating parallel data collection and centralized updates."""

    def __init__(self,
                 state_dim: int,
                 n_actions: int,
                 hidden_dim: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 alpha: float,
                 buffer_size: int):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy = SACActorCritic(state_dim, n_actions, hidden_dim).to(DEVICE)
        self.target_q1 = SACActorCritic(state_dim, n_actions, hidden_dim).to(DEVICE)
        self.target_q2 = SACActorCritic(state_dim, n_actions, hidden_dim).to(DEVICE)
        self.target_q1.load_state_dict(self.policy.state_dict())
        self.target_q2.load_state_dict(self.policy.state_dict())

        self.actor_opt = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.policy.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.policy.q2.parameters(), lr=lr)

        self.replay = ReplayBuffer(max_size=buffer_size)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tgt, src in zip(target.parameters(), source.parameters()):
                tgt.copy_(self.tau * src + (1.0 - self.tau) * tgt)

    def update(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            next_logits = self.policy.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_q1 = self.target_q1.q1(next_states)
            next_q2 = self.target_q2.q2(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q1_preds = self.policy.q1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_preds = self.policy.q2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(q1_preds, target_q)
        q2_loss = F.mse_loss(q2_preds, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        logits = self.policy.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q_values = torch.min(self.policy.q1(states), self.policy.q2(states))
        policy_loss = (probs * (self.alpha * log_probs - q_values)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.target_q1, self.policy)
        self._soft_update(self.target_q2, self.policy)

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
        }


def train_parallel_sac(args):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", f"{args.exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    csv_path = os.path.join(run_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episodes_collected",
            "avg_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "mean_adherence",
            "success_rate",
            "collection_time",
            "update_time",
            "buffer_size",
        ])

    state_dim = 2 + 4 * 2 + 4
    agent = ParallelSAC(
        state_dim=state_dim,
        n_actions=4,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
    )

    executor = ProcessPoolExecutor(max_workers=args.n_workers)
    total_episodes = 0
    update_idx = 0

    print("\n=== Parallel SAC Maze Training ===")
    print(f"Workers: {args.n_workers}")
    print(f"Total episodes: {args.total_episodes}")
    print(f"Episodes/update: {args.episodes_per_update}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Logs: {run_dir}\n")

    while total_episodes < args.total_episodes:
        update_idx += 1
        episodes_target = min(args.episodes_per_update, args.total_episodes - total_episodes)
        print(f"[Update {update_idx:04d}] Collecting {episodes_target} episodes...")

        start_collection = time.time()
        weights_cpu = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}

        futures = [
            executor.submit(
                collect_episode,
                worker_id=total_episodes + wid,
                policy_weights=weights_cpu,
                max_steps=args.max_steps,
                wall_density=args.wall_density,
                seed_base=args.seed,
            )
            for wid in range(episodes_target)
        ]

        episodes = []
        for idx, fut in enumerate(as_completed(futures), start=1):
            episodes.append(fut.result())
            if idx % max(1, episodes_target // 5) == 0:
                elapsed = max(time.time() - start_collection, 1e-6)
                print(f"  Collected {idx}/{episodes_target} ({idx / elapsed:.2f} eps/s)")

        collection_time = time.time() - start_collection

        rewards = []
        adherences = []
        successes = []
        transitions_added = 0
        for ep in episodes:
            rewards.append(ep["total_reward"])
            adherences.append(ep["adherence"])
            successes.append(1.0 if ep["success"] else 0.0)
            for tr in ep["transitions"]:
                agent.replay.add(tr["state"], tr["action"], tr["reward"], tr["next_state"], tr["done"])
            transitions_added += len(ep["transitions"])

        total_episodes += len(episodes)
        print(f"  Added {len(episodes)} episodes, {transitions_added} transitions. Buffer={len(agent.replay)}")

        start_update = time.time()
        updates_to_run = transitions_added * args.updates_per_step
        last_stats = {}
        for _ in range(updates_to_run):
            last_stats = agent.update(args.batch_size)
        update_time = time.time() - start_update

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        min_reward = float(np.min(rewards)) if rewards else 0.0
        max_reward = float(np.max(rewards)) if rewards else 0.0
        mean_adh = float(np.mean(adherences)) if adherences else 0.0
        success_rate = float(np.mean(successes)) if successes else 0.0

        print(f"  Reward avg/std/min/max: {avg_reward:.2f}/{std_reward:.2f}/{min_reward:.2f}/{max_reward:.2f}")
        print(f"  Adherence {mean_adh:.3f} | Success {success_rate:.3f} | Updates {updates_to_run} ({update_time:.2f}s)\n")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_episodes,
                f"{avg_reward:.4f}",
                f"{std_reward:.4f}",
                f"{min_reward:.4f}",
                f"{max_reward:.4f}",
                f"{mean_adh:.4f}",
                f"{success_rate:.4f}",
                f"{collection_time:.4f}",
                f"{update_time:.4f}",
                len(agent.replay),
            ])

        if update_idx % args.checkpoint_interval == 0:
            ckpt = os.path.join(run_dir, f"sac_policy_{total_episodes}.pt")
            torch.save(agent.policy.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}\n")

    final_ckpt = os.path.join(run_dir, "sac_policy_final.pt")
    torch.save(agent.policy.state_dict(), final_ckpt)
    print(f"Training complete. Final checkpoint saved to {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Parallel SAC baseline training on ObstacleMazeEnv")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--total-episodes", type=int, default=2000)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--wall-density", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--exp-name", type=str, default="sac_parallel_maze")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    train_parallel_sac(args)


if __name__ == "__main__":
    main()

