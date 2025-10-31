"""
Multiprocessing PPO baseline on ObstacleMazeEnv using env-only rewards.
Trains to visit 4 checkpoints in canonical order [0,1,2,3].
"""

import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.obstacle_maze_env import ObstacleMazeEnv
import os
import json
from datetime import datetime


def calculate_first_visits(visited_sequence: list) -> list:
    seen = set()
    first_visits = []
    for t in visited_sequence:
        if t not in seen:
            seen.add(t)
            first_visits.append(t)
    return first_visits


def adherence_rate(first_visits: list, workflow: list) -> float:
    matches = 0
    for i in range(min(len(first_visits), len(workflow))):
        if first_visits[i] == workflow[i]:
            matches += 1
        else:
            break
    return matches / len(workflow)


class Policy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, num_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.pi = nn.Linear(hidden, num_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v


def worker_episode(worker_id: int, policy_state_dict, max_steps: int, wall_density: float, return_queue: mp.Queue):
    """Run one episode in a separate process."""
    try:
        torch.set_num_threads(1)
        state_dim = 2 + 4 * 2 + 4  # agent(2) + checkpoint_centers(8) + visited_flags(4)
        policy = Policy(state_dim)
        policy.load_state_dict(policy_state_dict)
        policy.eval()

        env = ObstacleMazeEnv(max_steps=max_steps, wall_density=wall_density)
        env.reset()
        state = env.get_state_for_policy()

        traj = {k: [] for k in ["states", "actions", "logps", "rewards", "values", "dones"]}
        visited_sequence = []
        ep_return = 0.0

        while True:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                a, logp, v = policy.act(s)
                a_int = int(a.item())
                logp_val = float(logp.item())
                v_val = float(v.item())

            _, r, d, info = env.step(a_int)
            next_state = env.get_state_for_policy()

            # Track first checkpoint entries
            pos = tuple(env.agent_pos)
            for cp_idx in range(env.num_checkpoints):
                if env._in_checkpoint(pos, cp_idx) and cp_idx not in visited_sequence:
                    visited_sequence.append(cp_idx)

            traj["states"].append(state)
            traj["actions"].append(a_int)
            traj["logps"].append(logp_val)
            traj["rewards"].append(float(r))
            traj["values"].append(v_val)
            traj["dones"].append(bool(d))
            ep_return += r
            state = next_state
            if d:
                break

        first_visits = calculate_first_visits(visited_sequence)
        adherence = adherence_rate(first_visits, [0, 1, 2, 3])
        success = (first_visits == [0, 1, 2, 3])
        traj["ep_return"] = float(ep_return)
        traj["visited_sequence"] = first_visits
        traj["adherence"] = float(adherence)
        traj["success"] = bool(success)

        return_queue.put((worker_id, traj))
    except Exception as e:
        return_queue.put((worker_id, {"error": str(e)}))


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv, ret = [], []
    gae, next_v = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_v = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv.insert(0, gae)
        ret.insert(0, gae + values[t])
        next_v = values[t]
    return np.array(adv, dtype=np.float32), np.array(ret, dtype=np.float32)


def ppo_update(policy: Policy, optimizer, batch, clip=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, bs=128, device=None):
    states = torch.tensor(np.array(batch['states']), dtype=torch.float32, device=device)
    actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
    old_logps = torch.tensor(batch['logps'], dtype=torch.float32, device=device)
    adv = torch.tensor(batch['advantages'], dtype=torch.float32, device=device)
    ret = torch.tensor(batch['returns'], dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    num_samples = states.shape[0]
    indices = np.arange(num_samples)
    last_stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, bs):
            mb = indices[start:start+bs]
            mb_states = states[mb]
            mb_actions = actions[mb]
            mb_old_logps = old_logps[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logits, values = policy(mb_states)
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, mb_actions.view(-1, 1)).squeeze(1)
            ratio = torch.exp(logp - mb_old_logps)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, mb_ret)
            entropy = -(torch.softmax(logits, dim=-1) * logp_all).sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            last_stats = {
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.item())
            }

    return last_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--num_envs', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=800)
    parser.add_argument('--wall_density', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--exp_name', type=str, default='ppo_baseline_maze_mp')
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('logs', exist_ok=True)
    run_dir = os.path.join('logs', f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            'updates': args.updates,
            'num_envs': args.num_envs,
            'max_steps': args.max_steps,
            'wall_density': args.wall_density,
            'lr': args.lr,
            'ppo_epochs': args.ppo_epochs,
            'minibatch_size': args.minibatch_size,
            'exp_name': args.exp_name
        }, f, indent=2)
    updates_csv = os.path.join(run_dir, 'updates.csv')
    with open(updates_csv, 'w') as f:
        f.write('update,mean_return,mean_adherence,success_rate,policy_loss,value_loss,entropy\n')

    state_dim = 2 + 4 * 2 + 4
    policy = Policy(state_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for update in range(args.updates):
        policy_cpu = Policy(state_dim)
        policy_cpu.load_state_dict({k: v.detach().cpu() for k, v in policy.state_dict().items()})
        policy_state_dict = policy_cpu.state_dict()

        return_queue: mp.Queue = mp.Queue()
        workers: List[mp.Process] = []
        for wid in range(args.num_envs):
            p = mp.Process(target=worker_episode, args=(wid, policy_state_dict, args.max_steps, args.wall_density, return_queue))
            p.daemon = True
            p.start()
            workers.append(p)

        trajectories: List[Dict] = [None] * args.num_envs
        collected = 0
        while collected < args.num_envs:
            wid, data = return_queue.get()
            trajectories[wid] = data
            collected += 1

        for p in workers:
            p.join()

        for i, tr in enumerate(trajectories):
            if isinstance(tr, dict) and 'error' in tr:
                raise RuntimeError(f"Worker {i} failed: {tr['error']}")

        batch = {k: [] for k in ['states', 'actions', 'logps', 'rewards', 'values', 'dones']}
        for tr in trajectories:
            for k in batch.keys():
                batch[k].extend(tr[k])
        advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['dones'])
        batch['advantages'] = advantages
        batch['returns'] = returns

        stats = ppo_update(policy, optimizer, batch, clip=0.2, value_coef=0.5, entropy_coef=0.01, 
                          epochs=args.ppo_epochs, bs=args.minibatch_size, device=device)

        mean_return = float(np.mean([tr['ep_return'] for tr in trajectories]))
        mean_adherence = float(np.mean([tr['adherence'] for tr in trajectories]))
        success_rate = float(np.mean([1.0 if tr['success'] else 0.0 for tr in trajectories]))
        print(f"Update {update:3d} | Return {mean_return:7.2f} | Adherence {mean_adherence:5.1%} | Success {success_rate:5.1%} | pi {stats['policy_loss']:.3f} | v {stats['value_loss']:.3f}")
        with open(updates_csv, 'a') as f:
            f.write(f"{update},{mean_return:.4f},{mean_adherence:.4f},{success_rate:.4f},{stats['policy_loss']:.6f},{stats['value_loss']:.6f},{stats['entropy']:.6f}\n")

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump({
            'final_update': args.updates,
            'final_mean_return': mean_return,
            'final_mean_adherence': mean_adherence,
            'final_success_rate': success_rate
        }, f, indent=2)

    print(f"Training finished. Logs saved to {run_dir}")


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()

