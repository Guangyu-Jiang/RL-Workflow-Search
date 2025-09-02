"""
Workflow-conditioned PPO with distance-based shaping on DiagonalCornersEnv
Adds potential-based shaping towards current target in the desired workflow.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Tuple
import argparse

from core.diagonal_corners_env import DiagonalCornersEnv


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


class WorkflowPolicy(nn.Module):
    def __init__(self, state_dim: int, wf_dim: int, hidden: int = 128, num_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + wf_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.pi = nn.Linear(hidden, num_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, wf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([state, wf], dim=-1))
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, state: torch.Tensor, wf: torch.Tensor):
        logits, v = self.forward(state, wf)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v


def workflow_to_vector(order: List[int], num_targets: int = 4) -> np.ndarray:
    vec = np.zeros(num_targets, dtype=np.float32)
    for i, t in enumerate(order):
        vec[t] = (i + 1) / num_targets
    return vec


def rollout_shaped(env: DiagonalCornersEnv, policy: WorkflowPolicy, wf_order: List[int], device: torch.device, gamma: float, shaping_coef: float) -> Dict:
    env.reset(wf_order)
    state = env.get_state_for_policy()
    wf_vec = workflow_to_vector(wf_order)

    traj = {k: [] for k in ["states","workflows","actions","logps","rewards","values","dones"]}
    ep_return = 0.0

    while True:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        wf = torch.tensor(wf_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, logp, v = policy.act(s, wf)
        a_int = int(a.item())

        # Potential-based shaping toward current target
        cur_idx = env.current_target_idx
        cur_target = env.target_positions[wf_order[cur_idx]] if cur_idx < len(wf_order) else env.target_positions[wf_order[-1]]
        phi_s = -manhattan(tuple(env.agent_pos), cur_target)

        _, r_env, done, info = env.step(a_int)
        state_next = env.get_state_for_policy()

        # Update potential after step
        cur_idx2 = env.current_target_idx
        cur_target2 = env.target_positions[wf_order[cur_idx2]] if cur_idx2 < len(wf_order) else env.target_positions[wf_order[-1]]
        phi_s2 = -manhattan(tuple(env.agent_pos), cur_target2)

        shaped = r_env + shaping_coef * (gamma * phi_s2 - phi_s)

        traj["states"].append(state)
        traj["workflows"].append(wf_vec)
        traj["actions"].append(a_int)
        traj["logps"].append(float(logp.item()))
        traj["rewards"].append(float(shaped))
        traj["values"].append(float(v.item()))
        traj["dones"].append(bool(done))
        ep_return += shaped
        state = state_next
        if done:
            break

    traj["ep_return"] = float(ep_return)
    return traj


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


def ppo_update(policy: WorkflowPolicy, optimizer, batch, clip=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, bs=64, device=None):
    states = torch.tensor(np.array(batch['states']), dtype=torch.float32, device=device)
    workflows = torch.tensor(np.array(batch['workflows']), dtype=torch.float32, device=device)
    actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
    old_logps = torch.tensor(batch['logps'], dtype=torch.float32, device=device)
    adv = torch.tensor(batch['advantages'], dtype=torch.float32, device=device)
    ret = torch.tensor(batch['returns'], dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(epochs):
        logits, values = policy(states, workflows)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, actions.view(-1,1)).squeeze(1)
        ratio = torch.exp(logp - old_logps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, ret)
        entropy = -(torch.softmax(logits, dim=-1) * logp_all).sum(dim=-1).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return {
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'entropy': float(entropy.item())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--episodes_per_update', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--shaping_coef', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = DiagonalCornersEnv()
    wf_order = [0, 1, 2, 3]

    state_dim = 2 + 4*2 + 4
    wf_dim = 4
    policy = WorkflowPolicy(state_dim, wf_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    for update in range(args.updates):
        batch_trajs = [rollout_shaped(env, policy, wf_order, device, args.gamma, args.shaping_coef) for _ in range(args.episodes_per_update)]
        batch = {k: [] for k in ['states','workflows','actions','logps','rewards','values','dones']}
        for tr in batch_trajs:
            for k in batch.keys():
                batch[k].extend(tr[k])
        advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['dones'], gamma=args.gamma)
        batch['advantages'] = advantages
        batch['returns'] = returns

        stats = ppo_update(policy, optimizer, batch, epochs=args.ppo_epochs, device=device)

        if update % 10 == 0:
            mean_return = float(np.mean([tr['ep_return'] for tr in batch_trajs]))
            print(f"Update {update:3d} | Return {mean_return:6.2f} | pi {stats['policy_loss']:.3f} | v {stats['value_loss']:.3f}")

    print("Training finished.")


if __name__ == '__main__':
    main()


