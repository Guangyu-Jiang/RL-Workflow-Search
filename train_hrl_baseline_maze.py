"""
Simple hierarchical RL baseline on ObstacleMazeEnv.

Trains a two-level policy (options + low-level controller) on a single workflow
order without any GP-UCB workflow search. Uses adherence-shaped rewards and
collects rollouts from a list of parallel environments.
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.obstacle_maze_env import ObstacleMazeEnv
from core.hierarchical_policy import HierarchicalPolicy


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AdherenceShapedMazeEnv(ObstacleMazeEnv):
    """Subclass that exposes adherence-shaped reward signals."""

    def __init__(self, workflow: List[int], per_step_penalty: float,
                 adherence_bonus: float, adherence_penalty: float,
                 correct_entry_bonus: float, penalty_future: float,
                 penalty_offworkflow: float, env_reward_coef: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.workflow = list(workflow)
        self.per_step_penalty = float(per_step_penalty)
        self.adherence_bonus = float(adherence_bonus)
        self.adherence_penalty = float(adherence_penalty)
        self.correct_entry_bonus = float(correct_entry_bonus)
        self.penalty_future = float(penalty_future)
        self.penalty_offworkflow = float(penalty_offworkflow)
        self.env_reward_coef = float(env_reward_coef)
        self.visited_sequence: List[int] = []
        self._prev_adherence: float = 0.0

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.visited_sequence = []
        self._prev_adherence = 0.0
        return obs

    def step(self, action: int):
        obs, env_reward, done, info = super().step(action)

        current_pos = tuple(self.agent_pos)
        entered_idx = None
        for idx in range(self.num_checkpoints):
            if self._in_checkpoint(current_pos, idx):
                entered_idx = int(idx)
                break

        if entered_idx is not None and entered_idx not in self.visited_sequence:
            self.visited_sequence.append(entered_idx)

        shaped = self.per_step_penalty

        if entered_idx is not None:
            if entered_idx == self.workflow[len(self.visited_sequence) - 1]:
                shaped += self.correct_entry_bonus
            elif entered_idx in self.workflow:
                shaped += self.penalty_future
            else:
                shaped += self.penalty_offworkflow

        adherence = self._compute_adherence()
        delta = adherence - self._prev_adherence
        if delta > 0:
            shaped += self.adherence_bonus * delta
        elif delta < 0:
            shaped += self.adherence_penalty * abs(delta)
        self._prev_adherence = adherence

        if self.env_reward_coef != 0.0:
            shaped += self.env_reward_coef * float(env_reward)

        info = dict(info)
        info["visited_sequence"] = list(self.visited_sequence)
        info["adherence"] = float(adherence)
        info["success"] = bool(adherence == 1.0)

        return obs, float(shaped), done, info

    def _compute_adherence(self) -> float:
        prefix_ok = 0
        for i, t in enumerate(self.visited_sequence):
            if i < len(self.workflow) and t == self.workflow[i]:
                prefix_ok += 1
            else:
                break
        return prefix_ok / float(len(self.workflow))

    def get_state_for_policy(self) -> np.ndarray:
        state = np.zeros(22, dtype=np.float32)
        norm = float(self.grid_size - 1)
        state[0] = self.agent_pos[0] / norm
        state[1] = self.agent_pos[1] / norm
        for i, (r, c) in enumerate(self.checkpoint_centers):
            state[2 + i * 2] = r / norm
            state[2 + i * 2 + 1] = c / norm
        for i in range(self.num_checkpoints):
            state[10 + i] = 1.0 if i in self.visited_sequence else 0.0
        next_one_hot = np.zeros(4, dtype=np.float32)
        next_idx = 0
        for i, t in enumerate(self.workflow):
            if t in self.visited_sequence:
                next_idx = i + 1
            else:
                break
        next_cp = self.workflow[min(next_idx, len(self.workflow) - 1)]
        next_one_hot[next_cp] = 1.0
        state[14:18] = next_one_hot
        wf_vec = np.zeros(4, dtype=np.float32)
        for i, t in enumerate(self.workflow):
            wf_vec[t] = (i + 1) / 4.0
        state[18:22] = wf_vec
        return state


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


def ppo_update_hrl(high_policy,
                   low_policy,
                   optimizer_high, optimizer_low,
                   batch_high, batch_low,
                   clip=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, bs=128, device=None):
    states_high = torch.tensor(np.array(batch_high['states']), dtype=torch.float32, device=device)
    options = torch.tensor(batch_high['options'], dtype=torch.long, device=device)
    old_logps_high = torch.tensor(batch_high['logps'], dtype=torch.float32, device=device)
    adv_high = torch.tensor(batch_high['advantages'], dtype=torch.float32, device=device)
    ret_high = torch.tensor(batch_high['returns'], dtype=torch.float32, device=device)

    adv_high = (adv_high - adv_high.mean()) / (adv_high.std() + 1e-8)

    num_samples_high = states_high.shape[0]
    indices_high = np.arange(num_samples_high)
    for _ in range(epochs):
        np.random.shuffle(indices_high)
        for start in range(0, num_samples_high, bs):
            batch_idx = indices_high[start:start + bs]
            s_batch = states_high[batch_idx]
            o_batch = options[batch_idx]
            old_lp_batch = old_logps_high[batch_idx]
            adv_batch = adv_high[batch_idx]

            logits = high_policy(s_batch)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(o_batch)

            ratio = torch.exp(new_logps - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss - entropy_coef * entropy
            optimizer_high.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(high_policy.parameters(), 0.5)
            optimizer_high.step()

    states_low = torch.tensor(np.array(batch_low['states']), dtype=torch.float32, device=device)
    options_low = torch.tensor(batch_low['options'], dtype=torch.long, device=device)
    actions = torch.tensor(batch_low['actions'], dtype=torch.long, device=device)
    old_logps_low = torch.tensor(batch_low['logps'], dtype=torch.float32, device=device)
    adv_low = torch.tensor(batch_low['advantages'], dtype=torch.float32, device=device)
    ret_low = torch.tensor(batch_low['returns'], dtype=torch.float32, device=device)

    adv_low = (adv_low - adv_low.mean()) / (adv_low.std() + 1e-8)

    num_samples_low = states_low.shape[0]
    indices_low = np.arange(num_samples_low)
    for _ in range(epochs):
        np.random.shuffle(indices_low)
        for start in range(0, num_samples_low, bs):
            batch_idx = indices_low[start:start + bs]
            s_batch = states_low[batch_idx]
            o_batch = options_low[batch_idx]
            a_batch = actions[batch_idx]
            old_lp_batch = old_logps_low[batch_idx]
            adv_batch = adv_low[batch_idx]
            ret_batch = ret_low[batch_idx]

            action_logits, values = low_policy.forward(s_batch, o_batch)
            dist = torch.distributions.Categorical(logits=action_logits)
            new_logps = dist.log_prob(a_batch)

            ratio = torch.exp(new_logps - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_batch)
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer_low.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(low_policy.parameters(), 0.5)
            optimizer_low.step()


def collect_hrl_rollouts(policy: HierarchicalPolicy, envs: List[AdherenceShapedMazeEnv],
                         option_duration: int, max_steps: int, device):
    num_envs = len(envs)
    traj_high = {k: [] for k in ["states", "options", "logps", "rewards", "values", "dones"]}
    traj_low = {k: [] for k in ["states", "options", "actions", "logps", "rewards", "values", "dones"]}
    episode_stats = []

    states = []
    options = []
    option_steps = []
    option_logps = []
    ep_returns = []
    ep_visited_seqs = []

    for idx, env in enumerate(envs):
        obs = env.reset()
        states.append(env.get_state_for_policy())
        ep_returns.append(0.0)
        ep_visited_seqs.append([])
        with torch.no_grad():
            s_tensor = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0).to(device)
            opt, logp_opt = policy.select_option(s_tensor, deterministic=False)
            options.append(int(opt.item()))
            option_logps.append(float(logp_opt.item()))
            option_steps.append(0)

    step_count = 0
    while step_count < max_steps:
        # Handle option terminations (batched)
        need_new_option = [i for i in range(num_envs) if option_steps[i] >= option_duration]
        if need_new_option:
            with torch.no_grad():
                term_states = torch.tensor([states[i] for i in need_new_option], dtype=torch.float32).to(device)
                term_options = torch.tensor([options[i] for i in need_new_option], dtype=torch.long).to(device)
                _, term_values = policy.low_level.forward(term_states, term_options)
                new_options, new_logps = policy.select_option(term_states, deterministic=False)
            
            for idx, i in enumerate(need_new_option):
                traj_high["states"].append(states[i])
                traj_high["options"].append(options[i])
                traj_high["logps"].append(option_logps[i])
                traj_high["rewards"].append(0.0)
                traj_high["values"].append(float(term_values[idx].item()))
                traj_high["dones"].append(False)
                
                options[i] = int(new_options[idx].item())
                option_logps[i] = float(new_logps[idx].item())
                option_steps[i] = 0

        # Batch action selection for all environments
        with torch.no_grad():
            all_states = torch.tensor(states, dtype=torch.float32).to(device)
            all_options = torch.tensor(options, dtype=torch.long).to(device)
            actions_tensor, logps_tensor, values_tensor = policy.select_action(
                all_states, all_options, deterministic=False
            )
            actions = [int(a.item()) for a in actions_tensor]
            logps_low = [float(lp.item()) for lp in logps_tensor]
            values_low = [float(v.item()) for v in values_tensor]

        # Step all environments
        terminals = []
        terminal_infos = []
        for i in range(num_envs):
            obs, reward, done, info = envs[i].step(actions[i])
            terminal = bool(done)
            state_for_policy = envs[i].get_state_for_policy()

            if len(info.get("visited_sequence", [])) > len(ep_visited_seqs[i]):
                ep_visited_seqs[i] = list(info["visited_sequence"])

            traj_low["states"].append(states[i])
            traj_low["options"].append(options[i])
            traj_low["actions"].append(actions[i])
            traj_low["logps"].append(logps_low[i])
            traj_low["rewards"].append(float(reward))
            traj_low["values"].append(values_low[i])
            traj_low["dones"].append(terminal)

            ep_returns[i] += float(reward)
            option_steps[i] += 1

            if terminal:
                terminals.append(i)
                terminal_infos.append(info)
            else:
                states[i] = state_for_policy

        # Batch process terminal episodes
        if terminals:
            with torch.no_grad():
                term_states = torch.tensor([states[i] for i in terminals], dtype=torch.float32).to(device)
                term_options = torch.tensor([options[i] for i in terminals], dtype=torch.long).to(device)
                _, term_values = policy.low_level.forward(term_states, term_options)
            
            for idx, i in enumerate(terminals):
                traj_high["states"].append(states[i])
                traj_high["options"].append(options[i])
                traj_high["logps"].append(option_logps[i])
                traj_high["rewards"].append(ep_returns[i])
                traj_high["values"].append(float(term_values[idx].item()))
                traj_high["dones"].append(True)

                episode_stats.append({
                    "return": float(ep_returns[i]),
                    "visited_sequence": list(ep_visited_seqs[i]),
                    "adherence": float(terminal_infos[idx].get("adherence", 0.0)),
                    "success": bool(terminal_infos[idx].get("success", False)),
                })

                # Reset environment
                envs[i].reset()
                states[i] = envs[i].get_state_for_policy()
                ep_returns[i] = 0.0
                ep_visited_seqs[i] = []
                option_steps[i] = 0

            # Batch select new options for reset environments
            with torch.no_grad():
                reset_states = torch.tensor([states[i] for i in terminals], dtype=torch.float32).to(device)
                reset_options, reset_logps = policy.select_option(reset_states, deterministic=False)
            
            for idx, i in enumerate(terminals):
                options[i] = int(reset_options[idx].item())
                option_logps[i] = float(reset_logps[idx].item())

        step_count += 1

    return traj_high, traj_low, episode_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", type=str, default="0,1,2,3",
                        help="Comma-separated checkpoint order to follow.")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--option_duration", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_high", type=float, default=None)
    parser.add_argument("--lr_low", type=float, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--ppo_value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.1)
    parser.add_argument("--wall_density", type=float, default=0.15)
    parser.add_argument("--per_step_penalty", type=float, default=0.0)
    parser.add_argument("--adherence_bonus", type=float, default=0.0)
    parser.add_argument("--adherence_penalty", type=float, default=0.0)
    parser.add_argument("--correct_entry_bonus", type=float, default=0.0)
    parser.add_argument("--penalty_future", type=float, default=0.0)
    parser.add_argument("--penalty_offworkflow", type=float, default=0.0)
    parser.add_argument("--env_reward_coef", type=float, default=1.0)
    parser.add_argument("--num_options", type=int, default=8)
    parser.add_argument("--option_embed_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="hrl_baseline_maze")
    args = parser.parse_args()

    workflow = [int(x.strip()) for x in args.workflow.split(",")]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("logs", exist_ok=True)
    run_dir = os.path.join("logs", f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            **vars(args),
            "workflow": workflow,
        }, f, indent=2)

    envs: List[AdherenceShapedMazeEnv] = []
    for idx in range(args.num_envs):
        env = AdherenceShapedMazeEnv(
            workflow=workflow,
            per_step_penalty=args.per_step_penalty,
            adherence_bonus=args.adherence_bonus,
            adherence_penalty=args.adherence_penalty,
            correct_entry_bonus=args.correct_entry_bonus,
            penalty_future=args.penalty_future,
            penalty_offworkflow=args.penalty_offworkflow,
            env_reward_coef=args.env_reward_coef,
            max_steps=args.max_steps,
            wall_density=args.wall_density,
        )
        env.reset(seed=args.seed + idx)
        envs.append(env)

    policy = HierarchicalPolicy(
        state_dim=22,
        num_options=int(args.num_options),
        num_actions=4,
        hidden_dim=int(args.hidden_dim),
        option_embed_dim=int(args.option_embed_dim),
    ).to(device)

    lr_high = args.lr_high if args.lr_high is not None else args.lr
    lr_low = args.lr_low if args.lr_low is not None else args.lr
    optimizer_high = torch.optim.Adam(policy.high_level.parameters(), lr=lr_high)
    optimizer_low = torch.optim.Adam(policy.low_level.parameters(), lr=lr_low)

    updates_csv = os.path.join(run_dir, "updates.csv")
    with open(updates_csv, "w") as f:
        f.write("update,mean_return,mean_adherence,mode_sequence,mode_fraction\n")

    for update in range(args.updates):
        traj_high, traj_low, episode_stats = collect_hrl_rollouts(
            policy, envs,
            option_duration=int(args.option_duration),
            max_steps=int(args.max_steps),
            device=device,
        )

        if len(traj_high["rewards"]) > 0:
            adv_high, ret_high = compute_gae(
                traj_high["rewards"], traj_high["values"], traj_high["dones"],
                gamma=args.gae_gamma, lam=args.gae_lambda
            )
            traj_high["advantages"] = adv_high.tolist()
            traj_high["returns"] = ret_high.tolist()

        if len(traj_low["rewards"]) > 0:
            adv_low, ret_low = compute_gae(
                traj_low["rewards"], traj_low["values"], traj_low["dones"],
                gamma=args.gae_gamma, lam=args.gae_lambda
            )
            traj_low["advantages"] = adv_low.tolist()
            traj_low["returns"] = ret_low.tolist()

        if len(traj_high["states"]) > 0 and len(traj_low["states"]) > 0:
            ppo_update_hrl(
                policy.high_level, policy.low_level,
                optimizer_high, optimizer_low,
                traj_high, traj_low,
                clip=args.ppo_clip,
                value_coef=args.ppo_value_coef,
                entropy_coef=args.ppo_entropy_coef,
                epochs=args.ppo_epochs,
                bs=args.minibatch_size,
                device=device,
            )

        if episode_stats:
            mean_return = float(np.mean([s["return"] for s in episode_stats]))
            mean_adh = float(np.mean([s["adherence"] for s in episode_stats]))
            vseqs = [tuple(s["visited_sequence"]) for s in episode_stats]
            if vseqs:
                counter = Counter(vseqs)
                mode_seq, mode_count = counter.most_common(1)[0]
                mode_frac = mode_count / len(vseqs)
            else:
                mode_seq, mode_frac = (), 0.0
        else:
            mean_return = mean_adh = mode_frac = 0.0
            mode_seq = ()

        print(f"[HRL] Update {update:04d} | Return {mean_return:7.2f} | Adherence {mean_adh:5.1%} | Visit {list(mode_seq)} ({mode_frac:5.1%})")

        with open(updates_csv, "a") as f:
            f.write(f"{update},{mean_return:.4f},{mean_adh:.4f},\"{list(mode_seq)}\",{mode_frac:.4f}\n")

    torch.save(policy.state_dict(), os.path.join(run_dir, "policy_final.pt"))
    print(f"Training complete. Logs in {run_dir}")


if __name__ == "__main__":
    main()

