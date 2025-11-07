"""
Parallel Soft Actor-Critic baseline for ObstacleMazeEnv.

- Collects episodes in parallel worker processes via ProcessPoolExecutor.
- No workflow conditioning; uses the canonical checkpoint order [0,1,2,3].
- Logs rewards and adherence statistics for comparison against workflow-search methods.
"""

import argparse
import csv
import json
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.obstacle_maze_env import ObstacleMazeEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class SACActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )


def calculate_first_visits(sequence: List[int]) -> List[int]:
    seen = set()
    first = []
    for cp in sequence:
        if cp not in seen:
            seen.add(cp)
            first.append(cp)
    return first


def adherence_rate(first_visits: List[int], workflow: List[int]) -> float:
    matches = 0
    for i, target in enumerate(workflow):
        if i < len(first_visits) and first_visits[i] == target:
            matches += 1
        else:
            break
    return matches / len(workflow)


def collect_episode(worker_id: int, policy_weights_cpu: Dict[str, torch.Tensor],
                    max_steps: int, wall_density: float, seed: int):
    torch.set_num_threads(1)
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    env = ObstacleMazeEnv(max_steps=max_steps, wall_density=wall_density)
    env.reset(seed=worker_seed)

    state_dim = 2 + 4 * 2 + 4
    action_dim = env.action_space.n

    policy = SACActorCritic(state_dim, action_dim, hidden=128).to("cpu")
    policy.load_state_dict(policy_weights_cpu)
    policy.eval()

    state = env.get_state_for_policy()
    transitions = []
    visited_sequence = []
    total_reward = 0.0

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = policy.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs=probs).sample().item()

        _, reward, done, _ = env.step(action)
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

        total_reward += reward
        state = next_state
        if done:
            break

    first_visits = calculate_first_visits(visited_sequence)
    adherence = adherence_rate(first_visits, [0, 1, 2, 3])

    return {
        "transitions": transitions,
        "total_reward": total_reward,
        "steps": len(transitions),
        "adherence": adherence,
        "success": first_visits == [0, 1, 2, 3],
    }


class ParallelSAC:
    def __init__(self, state_dim: int, n_actions: int,
                 lr: float, gamma: float, tau: float, alpha: float,
                 buffer_size: int):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_net = SACActorCritic(state_dim, n_actions, hidden=128).to(device)
        self.target_q1 = SACActorCritic(state_dim, n_actions, hidden=128).to(device)
        self.target_q2 = SACActorCritic(state_dim, n_actions, hidden=128).to(device)
        self.target_q1.load_state_dict(self.policy_net.state_dict())
        self.target_q2.load_state_dict(self.policy_net.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.policy_net.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.policy_net.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.policy_net.q2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_logits = self.policy_net.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_q1 = self.target_q1.q1(next_states)
            next_q2 = self.target_q2.q2(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        current_q1 = self.policy_net.q1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        current_q2 = self.policy_net.q2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        logits = self.policy_net.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q_values = torch.min(self.policy_net.q1(states), self.policy_net.q2(states))
        policy_loss = (probs * (self.alpha * log_probs - q_values)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for target_param, param in zip(self.target_q2.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)


def train_parallel_sac(n_workers: int = 50,
                      total_episodes: int = 100000,
                      episodes_per_update: int = 50,
                      batch_size: int = 256,
                      updates_per_step: int = 1,
                      max_steps: int = 100,
                      wall_density: float = 0.15,
                      seed: int = 42,
                      lr: float = 3e-4,
                      gamma: float = 0.99,
                      tau: float = 0.005,
                      alpha: float = 0.2,
                      buffer_size: int = 100000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"logs/parallel_sac_maze_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        "n_workers": n_workers,
        "total_episodes": total_episodes,
        "episodes_per_update": episodes_per_update,
        "batch_size": batch_size,
        "updates_per_step": updates_per_step,
        "max_steps": max_steps,
        "wall_density": wall_density,
        "seed": seed,
        "lr": lr,
        "gamma": gamma,
        "tau": tau,
        "alpha": alpha,
        "buffer_size": buffer_size,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(exp_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodes", "avg_reward", "std_reward", "min_reward", "max_reward",
                         "avg_adherence", "success_rate", "collection_time", "update_time",
                         "buffer_size"])

    state_dim = 2 + 4 * 2 + 4
    n_actions = 4
    agent = ParallelSAC(state_dim, n_actions, lr, gamma, tau, alpha, buffer_size)

    executor = ProcessPoolExecutor(max_workers=n_workers)

    total_episodes_collected = 0
    update_num = 0

    import time

    while total_episodes_collected < total_episodes:
        update_num += 1
        episodes_needed = min(episodes_per_update, total_episodes - total_episodes_collected)
        print(f"[SAC-Maze] Update {update_num}: collecting {episodes_needed} episodes...")

        start_collection = time.time()
        policy_weights_cpu = {k: v.detach().cpu() for k, v in agent.policy_net.state_dict().items()}

        futures = [
            executor.submit(
                collect_episode,
                worker_id=i,
                policy_weights_cpu=policy_weights_cpu,
                max_steps=max_steps,
                wall_density=wall_density,
                seed=seed,
            )
            for i in range(episodes_needed)
        ]

        episodes = []
        collected = 0
        for future in as_completed(futures):
            episodes.append(future.result())
            collected += 1
            # Report progress every 50 episodes or at milestones
            if collected % max(1, min(50, episodes_needed // 5)) == 0 or collected == episodes_needed:
                elapsed = max(time.time() - start_collection, 1e-6)
                rate = collected / elapsed
                print(f"  {collected}/{episodes_needed} episodes ({rate:.1f} eps/sec)")

        collection_time = time.time() - start_collection
        collection_rate = len(episodes) / max(collection_time, 1e-6)
        
        print(f"  Collected {len(episodes)} episodes in {collection_time:.1f}s ({collection_rate:.1f} eps/sec)")

        # Add transitions to replay buffer and collect statistics
        rewards = []
        adherences = []
        successes = []
        for ep in episodes:
            rewards.append(ep["total_reward"])
            adherences.append(ep["adherence"])
            successes.append(1.0 if ep["success"] else 0.0)
            for trans in ep["transitions"]:
                agent.replay_buffer.add(
                    trans["state"],
                    trans["action"],
                    trans["reward"],
                    trans["next_state"],
                    trans["done"],
                )

        total_episodes_collected += len(episodes)

        start_update = time.time()
        # Limit updates: one update per transition collected (or as specified)
        # This prevents excessive updates when many transitions are collected
        updates_to_run = min(len(episodes) * max_steps * updates_per_step, 
                            len(agent.replay_buffer) // batch_size)
        updates_to_run = max(1, updates_to_run)  # At least 1 update
        
        if len(agent.replay_buffer) >= batch_size:
            for _ in range(updates_to_run):
                agent.update(batch_size)
        update_time = time.time() - start_update

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        min_reward = float(np.min(rewards)) if rewards else 0.0
        max_reward = float(np.max(rewards)) if rewards else 0.0
        avg_adherence = float(np.mean(adherences)) if adherences else 0.0
        success_rate = float(np.mean(successes)) if successes else 0.0

        print(f"  SAC updates complete ({update_time:.2f}s, {updates_to_run} updates)")
        print(f"  Episodes: {total_episodes_collected}/{total_episodes}")
        print(f"  Rewards avg/std/min/max: {avg_reward:.2f}/{std_reward:.2f}/{min_reward:.2f}/{max_reward:.2f}")
        print(f"  Adherence {avg_adherence:.3f} | Success {success_rate:.3f} | Buffer {len(agent.replay_buffer)}")
        print()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                total_episodes_collected,
                f"{avg_reward:.4f}",
                f"{std_reward:.4f}",
                f"{min_reward:.4f}",
                f"{max_reward:.4f}",
                f"{avg_adherence:.4f}",
                f"{success_rate:.4f}",
                f"{collection_time:.4f}",
                f"{update_time:.4f}",
                len(agent.replay_buffer),
            ])

    final_ckpt = os.path.join(exp_dir, "sac_policy_final.pt")
    torch.save(agent.policy_net.state_dict(), final_ckpt)
    executor.shutdown(wait=True)
    print(f"Training complete. Final checkpoint saved to {final_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Parallel SAC baseline on ObstacleMazeEnv")
    parser.add_argument("--n-workers", type=int, default=50)
    parser.add_argument("--total-episodes", type=int, default=100000)
    parser.add_argument("--episodes-per-update", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--wall-density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=int, default=100000)
    args = parser.parse_args()

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    train_parallel_sac(
        n_workers=args.n_workers,
        total_episodes=args.total_episodes,
        episodes_per_update=args.episodes_per_update,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        max_steps=args.max_steps,
        wall_density=args.wall_density,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
