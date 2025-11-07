"""
Workflow Search with GP-UCB using Hierarchical RL Baseline on ObstacleMazeEnv.

Uses 2-level hierarchical policy:
- High-level: Selects options every ~option_duration steps
- Low-level: Executes actions using state + option embedding
"""

from workflow_search_gpucb import *  # reuse helpers
from core.obstacle_maze_env import ObstacleMazeEnv
from core.hierarchical_policy import HierarchicalPolicy

import multiprocessing as mp
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict, Tuple
import csv


MAX_TOTAL_UPDATES = 10000


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AdherenceShapedMazeEnv(gym.Wrapper):
    """Wrapper for maze env with potential shaping + adherence-progress rewards."""
    def __init__(self, env: gym.Env, workflow: List[int], gamma: float, shaping_coef: float,
                 penalty_revisit: float, penalty_future: float, penalty_offworkflow: float,
                 per_step_penalty: float, adherence_bonus: float = 50.0, adherence_penalty: float = -50.0,
                 correct_entry_bonus: float = 10.0, env_reward_coef: float = 0.0):
        super().__init__(env)
        self.workflow = list(workflow)
        self.gamma = float(gamma)
        self.shaping_coef = float(shaping_coef)
        self.penalty_revisit = float(penalty_revisit)
        self.penalty_future = float(penalty_future)
        self.penalty_offworkflow = float(penalty_offworkflow)
        self.per_step_penalty = float(per_step_penalty)
        self.adherence_bonus = float(adherence_bonus)
        self.adherence_penalty = float(adherence_penalty)
        self.correct_entry_bonus = float(correct_entry_bonus)
        self.env_reward_coef = float(env_reward_coef)
        self.visited_sequence: List[int] = []
        self._phi_s: float = 0.0
        self._prev_adherence: float = 0.0
        # Observation: agent(2) + checkpoint_centers(8) + visited(4) + next_cp_onehot(4) + workflow_vec(4) = 22
        import numpy as _np
        self.observation_space = gym.spaces.Box(
            low=_np.zeros(22, dtype=_np.float32),
            high=_np.ones(22, dtype=_np.float32),
            dtype=_np.float32,
        )

    def reset(self, **kwargs):
        self.visited_sequence = []
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._phi_s = self._compute_phi()
        self._prev_adherence = 0.0
        return self._augment_obs(obs), info

    def step(self, action):
        step_result = self.env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, _env_r, done, truncated, info = step_result
        else:
            obs, _env_r, done, info = step_result
            truncated = False
        pre_step_visited = set(self.visited_sequence)

        # Detect entry into any checkpoint
        entered_idx = None
        try:
            current_pos = tuple(self.env.agent_pos)
            for idx in range(self.env.num_checkpoints):
                if self.env._in_checkpoint(current_pos, idx):
                    entered_idx = int(idx)
                    break
        except Exception:
            entered_idx = None

        # Update visited sequence if first entry
        if entered_idx is not None and entered_idx not in pre_step_visited:
            self.visited_sequence.append(entered_idx)

        # Potential-based shaping
        phi_s = self._phi_s
        phi_s2 = self._compute_phi()
        shaped = self.shaping_coef * (self.gamma * phi_s2 - phi_s)

        # Penalties and bonuses
        if entered_idx is not None:
            if entered_idx in pre_step_visited:
                shaped += self.penalty_revisit
            else:
                prefix_ok = 0
                for i, t in enumerate(self.visited_sequence):
                    if i < len(self.workflow) and t == self.workflow[i]:
                        prefix_ok += 1
                    else:
                        break
                req_idx = min(prefix_ok, len(self.workflow) - 1)
                required_cp = self.workflow[req_idx]
                if int(entered_idx) == int(required_cp):
                    shaped += self.correct_entry_bonus
                elif entered_idx in self.workflow:
                    shaped += self.penalty_future
                else:
                    shaped += self.penalty_offworkflow

        shaped += self.per_step_penalty

        # Adherence-progress reward
        current_adherence = self._compute_adherence()
        adh_delta = current_adherence - self._prev_adherence
        if adh_delta > 0:
            shaped += self.adherence_bonus * adh_delta
        elif adh_delta < 0:
            shaped += self.adherence_penalty * abs(adh_delta)
        self._prev_adherence = current_adherence

        # Optionally include original environment reward
        if self.env_reward_coef != 0.0:
            shaped += self.env_reward_coef * float(_env_r)

        self._phi_s = phi_s2

        # Info
        try:
            adherence = current_adherence
            success = (int(current_adherence * len(self.workflow)) == len(self.workflow))
            info = dict(info)
            info['visited_sequence'] = list(self.visited_sequence)
            info['adherence'] = float(adherence)
            info['success'] = bool(success)
            info['adherence_delta'] = float(adh_delta)
        except Exception:
            pass

        return self._augment_obs(obs), float(shaped), done, truncated, info

    def _compute_adherence(self) -> float:
        prefix_ok = 0
        for i, t in enumerate(self.visited_sequence):
            if i < len(self.workflow) and t == self.workflow[i]:
                prefix_ok += 1
            else:
                break
        return float(prefix_ok) / float(len(self.workflow))

    def _augment_obs(self, obs):
        import numpy as _np
        grid_norm = float(self.env.grid_size - 1)
        if isinstance(obs, _np.ndarray):
            agent_r = float(obs[0]) / grid_norm
            agent_c = float(obs[1]) / grid_norm
        else:
            agent_r = float(obs[0]) / grid_norm
            agent_c = float(obs[1]) / grid_norm
        agent_norm = _np.asarray([agent_r, agent_c], dtype=_np.float32)

        # Checkpoint centers normalized
        centers = []
        for (r, c) in self.env.checkpoint_centers:
            centers.append(float(r) / grid_norm)
            centers.append(float(c) / grid_norm)
        centers_norm = _np.asarray(centers, dtype=_np.float32)

        # Visited flags
        visited_flags = _np.zeros(4, dtype=_np.float32)
        for t in self.visited_sequence:
            if 0 <= int(t) < 4:
                visited_flags[int(t)] = 1.0

        # Next-checkpoint one-hot
        one_hot = _np.zeros(4, dtype=_np.float32)
        next_idx = 0
        for i, t in enumerate(self.workflow):
            if t in self.visited_sequence:
                next_idx = i + 1
            else:
                break
        if next_idx >= len(self.workflow):
            next_cp = self.workflow[-1]
        else:
            next_cp = self.workflow[next_idx]
        if 0 <= int(next_cp) < 4:
            one_hot[int(next_cp)] = 1.0

        # Workflow encoding
        wf_vec = _np.zeros(4, dtype=_np.float32)
        for i, t in enumerate(self.workflow):
            wf_vec[t] = (i + 1) / 4.0

        return _np.concatenate([agent_norm, centers_norm, visited_flags, one_hot, wf_vec], axis=0)

    def _compute_phi(self) -> float:
        try:
            next_idx = 0
            for i, t in enumerate(self.workflow):
                if t in self.visited_sequence:
                    next_idx = i + 1
                else:
                    break
            if next_idx >= len(self.workflow):
                next_idx = len(self.workflow) - 1
            required_cp_id = self.workflow[next_idx]
            target_center = self.env.checkpoint_centers[required_cp_id]
            return -float(manhattan(tuple(self.env.agent_pos), tuple(target_center)))
        except Exception:
            return 0.0


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns."""
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
    """PPO update for hierarchical policy (both high and low levels)."""
    
    # Update high-level policy
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
            end = min(start + bs, num_samples_high)
            batch_idx = indices_high[start:end]
            
            s_batch = states_high[batch_idx]
            o_batch = options[batch_idx]
            old_lp_batch = old_logps_high[batch_idx]
            adv_batch = adv_high[batch_idx]
            ret_batch = ret_high[batch_idx]
            
            logits = high_policy(s_batch)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logps = dist.log_prob(o_batch)
            
            ratio = torch.exp(new_logps - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_batch
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            entropy = dist.entropy().mean()
            loss = policy_loss - entropy_coef * entropy
            
            optimizer_high.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(high_policy.parameters(), 0.5)
            optimizer_high.step()
    
    # Update low-level policy
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
            end = min(start + bs, num_samples_low)
            batch_idx = indices_low[start:end]
            
            s_batch = states_low[batch_idx]
            o_batch = options_low[batch_idx]
            a_batch = actions[batch_idx]
            old_lp_batch = old_logps_low[batch_idx]
            adv_batch = adv_low[batch_idx]
            ret_batch = ret_low[batch_idx]
            
            action_logits, values = low_policy.forward(s_batch, o_batch)
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logps = dist.log_prob(a_batch)
            
            ratio = torch.exp(new_logps - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_batch
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            value_loss = F.mse_loss(values, ret_batch)
            entropy = dist.entropy().mean()
            
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            optimizer_low.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(low_policy.parameters(), 0.5)
            optimizer_low.step()


def collect_hrl_rollouts(policy: HierarchicalPolicy, envs: List[gym.Env], 
                        option_duration: int = 10, max_steps: int = 800, device=None):
    """Collect rollouts using hierarchical policy.
    
    Returns:
        traj_high: High-level transitions (option selections)
        traj_low: Low-level transitions (action selections)
        episode_stats: List of dicts with episode metrics (return, adherence, visited_sequence)
    """
    num_envs = len(envs)
    traj_high = {k: [] for k in ["states", "options", "logps", "rewards", "values", "dones"]}
    traj_low = {k: [] for k in ["states", "options", "actions", "logps", "rewards", "values", "dones"]}
    episode_stats = []
    
    # Per-environment state
    states = []
    options = []
    option_steps = []
    option_logps = []  # Store high-level log probs
    ep_returns = []
    ep_visited_seqs = []
    
    # Initialize
    for env in envs:
        obs, _ = env.reset()
        states.append(obs)
        ep_returns.append(0.0)
        ep_visited_seqs.append([])
        with torch.no_grad():
            s_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            opt, logp_opt = policy.select_option(s_tensor, deterministic=False)
            options.append(int(opt.item()))
            option_logps.append(float(logp_opt.item()))
            option_steps.append(0)
    
    step_count = 0
    while step_count < max_steps:
        # Check option termination (every option_duration steps)
        for i in range(num_envs):
            if option_steps[i] >= option_duration:
                # Store high-level transition (option termination)
                with torch.no_grad():
                    s_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                    o_tensor = torch.tensor([options[i]], dtype=torch.long).to(device)
                    _, value_opt = policy.low_level.forward(s_tensor, o_tensor)
                    
                traj_high["states"].append(states[i])
                traj_high["options"].append(options[i])
                traj_high["logps"].append(option_logps[i])
                traj_high["rewards"].append(0.0)  # Option termination reward (can be shaped later)
                traj_high["values"].append(float(value_opt.item()))
                traj_high["dones"].append(False)
                
                # Select new option
                with torch.no_grad():
                    opt, logp_opt = policy.select_option(s_tensor, deterministic=False)
                    options[i] = int(opt.item())
                    option_logps[i] = float(logp_opt.item())
                    option_steps[i] = 0
        
        # Low-level: select actions
        actions = []
        logps_low = []
        values_low = []
        for i in range(num_envs):
            with torch.no_grad():
                s_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                o_tensor = torch.tensor([options[i]], dtype=torch.long).to(device)
                a, logp, v = policy.select_action(s_tensor, o_tensor, deterministic=False)
                actions.append(int(a.item()))
                logps_low.append(float(logp.item()))
                values_low.append(float(v.item()))
        
        # Step environments
        for i in range(num_envs):
            obs, reward, done, truncated, info = envs[i].step(actions[i])
            terminal = done or truncated
            
            # Track visited sequence
            vseq = info.get('visited_sequence', [])
            if len(vseq) > len(ep_visited_seqs[i]):
                ep_visited_seqs[i] = list(vseq)
            
            # Store low-level transition
            traj_low["states"].append(states[i].copy() if hasattr(states[i], 'copy') else states[i])
            traj_low["options"].append(options[i])
            traj_low["actions"].append(actions[i])
            traj_low["logps"].append(logps_low[i])
            traj_low["rewards"].append(float(reward))
            traj_low["values"].append(values_low[i])
            traj_low["dones"].append(bool(terminal))
            
            ep_returns[i] += float(reward)
            option_steps[i] += 1
            
            if terminal:
                # Store final high-level transition (episode end)
                with torch.no_grad():
                    s_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                    o_tensor = torch.tensor([options[i]], dtype=torch.long).to(device)
                    _, value_opt = policy.low_level.forward(s_tensor, o_tensor)
                
                traj_high["states"].append(states[i])
                traj_high["options"].append(options[i])
                traj_high["logps"].append(option_logps[i])
                traj_high["rewards"].append(ep_returns[i])  # Episode return as option reward
                traj_high["values"].append(float(value_opt.item()))
                traj_high["dones"].append(True)
                
                # Record episode stats
                episode_stats.append({
                    'return': float(ep_returns[i]),
                    'visited_sequence': list(ep_visited_seqs[i]),
                    'adherence': float(info.get('adherence', 0.0)),
                    'success': bool(info.get('success', False)),
                })
                
                # Reset environment
                obs, _ = envs[i].reset()
                states[i] = obs
                ep_returns[i] = 0.0
                ep_visited_seqs[i] = []
                with torch.no_grad():
                    s_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    opt, logp_opt = policy.select_option(s_tensor, deterministic=False)
                    options[i] = int(opt.item())
                    option_logps[i] = float(logp_opt.item())
                    option_steps[i] = 0
            else:
                states[i] = obs
        
        step_count += 1
    
    return traj_high, traj_low, episode_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--num_envs', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=800)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_high', type=float, default=None)  # If None, use --lr
    parser.add_argument('--lr_low', type=float, default=None)   # If None, use --lr
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--shaping_coef', type=float, default=1.0)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=250)
    parser.add_argument('--gae_gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--ppo_value_coef', type=float, default=0.5)
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.1)
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5)
    parser.add_argument('--per_step_penalty', type=float, default=-0.01)
    parser.add_argument('--wall_density', type=float, default=0.15)
    parser.add_argument('--adherence_target', type=float, default=1.0)
    parser.add_argument('--adherence_patience', type=int, default=2)
    parser.add_argument('--early_stop_on_adherence', action='store_true')
    parser.add_argument('--adh_eval_threshold', type=float, default=0.9)
    parser.add_argument('--penalty_revisit', type=float, default=0.0)
    parser.add_argument('--penalty_future', type=float, default=-10.0)
    parser.add_argument('--penalty_offworkflow', type=float, default=-5.0)
    parser.add_argument('--adherence_bonus', type=float, default=100.0)
    parser.add_argument('--adherence_penalty', type=float, default=-50.0)
    parser.add_argument('--correct_entry_bonus', type=float, default=20.0)
    parser.add_argument('--env_reward_coef', type=float, default=0.0)
    parser.add_argument('--num_options', type=int, default=8)
    parser.add_argument('--option_duration', type=int, default=10)
    parser.add_argument('--option_embed_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--length_scale', type=float, default=3.0)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--kernel_type', type=str, default='rbf_possunmatch',
                        choices=['rbf_rank','rbf_pairwise','rbf_mixed','rbf_posunmatch','rbf_possunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=1e-4)
    parser.add_argument('--kappa', type=float, default=4.0)
    parser.add_argument('--enable_ucb_early_stop', action='store_true')
    parser.add_argument('--stop_min_explored', type=int, default=5)
    parser.add_argument('--allow_revisits', dest='allow_revisits', action='store_true')
    parser.add_argument('--no_revisits', dest='allow_revisits', action='store_false')
    parser.set_defaults(allow_revisits=True)
    parser.add_argument('--continue_policy', action='store_true')
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker_seed_base', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='hrl_gpucb_maze')
    parser.add_argument('--spawn_mp', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs('logs', exist_ok=True)
    proc_name = mp.current_process().name
    pid = os.getpid()
    run_dir = os.path.join('logs', f"{args.exp_name}_{proc_name}_pid{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'gp_workflow_search.jsonl')
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            **vars(args),
            'proc_name': proc_name,
            'pid': int(pid),
            'run_dir': run_dir,
            'MAX_TOTAL_UPDATES': int(MAX_TOTAL_UPDATES),
            'num_candidates': 24,
        }, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a reference maze instance to save layout visualization
    ref_env = ObstacleMazeEnv(max_steps=int(args.max_steps), wall_density=float(args.wall_density), seed=int(args.seed))
    ref_env.reset()
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        ax.set_xlim(-0.5, ref_env.grid_size - 0.5)
        ax.set_ylim(-0.5, ref_env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        # Draw walls
        wall_positions = np.argwhere(ref_env.walls == 1)
        for (r, c) in wall_positions:
            rect = mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0, facecolor='gray', alpha=0.6)
            ax.add_patch(rect)
        # Draw checkpoints
        cp_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (r_min, r_max, c_min, c_max) in enumerate(ref_env.checkpoints):
            width = c_max - c_min + 1
            height = r_max - r_min + 1
            rect = mpatches.Rectangle((c_min - 0.5, r_min - 0.5), width, height, linewidth=2,
                                     edgecolor=cp_colors[idx], facecolor=cp_colors[idx], alpha=0.4)
            ax.add_patch(rect)
            center_r = (r_min + r_max) / 2.0
            center_c = (c_min + c_max) / 2.0
            ax.text(center_c, center_r, f'CP{idx}', fontsize=12, fontweight='bold', ha='center', va='center', color=cp_colors[idx])
        # Mark start
        sr, sc = ref_env.start_pos
        ax.plot(sc, sr, marker='o', markersize=10, color='black', markeredgewidth=2, markerfacecolor='white', zorder=10)
        ax.text(sc, sr, 'S', fontsize=8, fontweight='bold', ha='center', va='center', zorder=11)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(f'Maze Layout (seed={args.seed}, walls={ref_env.walls.sum()}, density={args.wall_density:.2f})', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, ref_env.grid_size, 5))
        ax.set_yticks(range(0, ref_env.grid_size, 5))
        ax.grid(True, alpha=0.2, linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'maze_layout.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved maze layout to {run_dir}/maze_layout.png")
    except Exception as e:
        print(f"Could not save maze visualization: {e}")

    # Candidate workflows (4! = 24)
    candidates: List[List[int]] = [list(p) for p in itertools.permutations([0, 1, 2, 3])]
    candidate_embeddings = [
        build_workflow_embedding(
            w,
            kernel_type=args.kernel_type,
            num_targets=4,
            rank_scale=args.rank_scale,
            pairwise_scale=args.pairwise_scale,
        ) for w in candidates
    ]

    observed_indices: List[int] = []
    observed_embeddings: List[np.ndarray] = []
    observed_scores: List[float] = []
    visit_counts: Dict[Tuple, int] = {}
    current_policy_state = None
    current_lr = float(args.lr)
    current_lr_high = float(args.lr_high) if args.lr_high is not None else float(args.lr)
    current_lr_low = float(args.lr_low) if args.lr_low is not None else float(args.lr)

    it = 0
    global_update_count = 0
    hit_global_limit = False
    worker_seed_base = int(args.worker_seed_base) if args.worker_seed_base is not None else int(args.seed)

    while True:
        mu, std = gp_posterior(
            observed_embeddings,
            observed_scores,
            candidate_embeddings,
            length_scale=args.length_scale,
            noise=args.noise,
            signal_variance=args.signal_variance,
        )
        std = np.maximum(std, 1e-2)
        ucb = mu + args.kappa * std

        if args.enable_ucb_early_stop and len(observed_indices) >= int(args.stop_min_explored):
            unexplored = [i for i in range(len(candidates)) if i not in observed_indices]
            if len(unexplored) > 0:
                best_score = float(np.max(observed_scores)) if len(observed_scores) > 0 else -np.inf
                max_ucb_unexplored = float(np.max(ucb[unexplored]))
                if max_ucb_unexplored <= best_score:
                    print(f"[GP-UCB] Early stop: max UCB(unexplored) {max_ucb_unexplored:.2f} <= best_score {best_score:.2f}")
                    break

        ucb_for_choice = ucb.copy()
        if not bool(args.allow_revisits):
            ucb_for_choice[observed_indices] = -np.inf
        max_val = float(np.max(ucb_for_choice))
        tie_indices = [i for i, val in enumerate(ucb_for_choice) if val == max_val]
        next_idx = int(np.random.choice(tie_indices))
        print(f"[GP-UCB] Iter {it} selecting workflow {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")

        wf = tuple(candidates[next_idx])

        continued = False
        effective_lr = float(current_lr) if bool(args.continue_policy) and current_policy_state is not None else float(args.lr)
        effective_lr_high = float(current_lr_high) if bool(args.continue_policy) and current_policy_state is not None else (float(args.lr_high) if args.lr_high is not None else float(args.lr))
        effective_lr_low = float(current_lr_low) if bool(args.continue_policy) and current_policy_state is not None else (float(args.lr_low) if args.lr_low is not None else float(args.lr))
        if bool(args.continue_policy) and current_policy_state is not None:
            continued = True

        vc = visit_counts.get(wf, 0) + 1
        visit_counts[wf] = vc
        env_eval_history = []
        updates_csv = os.path.join(run_dir, 'updates.csv')
        consecutive_target_meets = 0
        last_mean_adherence = 0.0
        score_on_adh_thresh = None
        env_eval_when_adh_thresh = []

        # Build envs
        def make_env_fn(rank: int):
            def _init():
                base = ObstacleMazeEnv(max_steps=int(args.max_steps), wall_density=float(args.wall_density))
                env = AdherenceShapedMazeEnv(
                    base,
                    list(wf),
                    gamma=float(args.gamma),
                    shaping_coef=float(args.shaping_coef),
                    penalty_revisit=float(args.penalty_revisit),
                    penalty_future=float(args.penalty_future),
                    penalty_offworkflow=float(args.penalty_offworkflow),
                    per_step_penalty=float(args.per_step_penalty),
                    adherence_bonus=float(args.adherence_bonus),
                    adherence_penalty=float(args.adherence_penalty),
                    correct_entry_bonus=float(args.correct_entry_bonus),
                    env_reward_coef=float(args.env_reward_coef),
                )
                try:
                    env.reset(seed=int(worker_seed_base) + int(rank))
                except TypeError:
                    try:
                        env.seed(int(worker_seed_base) + int(rank))
                    except Exception:
                        pass
                try:
                    env.action_space.seed(int(worker_seed_base) + int(rank))
                    env.observation_space.seed(int(worker_seed_base) + int(rank))
                except Exception:
                    pass
                return env
            return _init

        # Create environments list (not using VecEnv for custom rollout collection)
        envs = [make_env_fn(i)() for i in range(int(args.num_envs))]

        # Create hierarchical policy
        state_dim = 22  # From AdherenceShapedMazeEnv observation space
        num_actions = 4
        policy = HierarchicalPolicy(
            state_dim=state_dim,
            num_options=int(args.num_options),
            num_actions=num_actions,
            hidden_dim=int(args.hidden_dim),
            option_embed_dim=int(args.option_embed_dim),
        ).to(device)

        if continued:
            try:
                policy.load_state_dict(current_policy_state)
            except Exception:
                pass

        # Optimizers
        optimizer_high = optim.Adam(policy.high_level.parameters(), lr=effective_lr_high)
        optimizer_low = optim.Adam(policy.low_level.parameters(), lr=effective_lr_low)

        # Training loop
        for update in range(int(args.updates)):
            global_update_count += 1
            total_update_idx = int(global_update_count)

            # Collect rollouts
            traj_high, traj_low, episode_stats = collect_hrl_rollouts(
                policy, envs, 
                option_duration=int(args.option_duration),
                max_steps=int(args.max_steps),
                device=device
            )

            # Compute advantages for high-level
            if len(traj_high['rewards']) > 0:
                # Use stored values from rollout collection
                values_high = traj_high['values']
                adv_high, ret_high = compute_gae(
                    traj_high['rewards'], values_high, traj_high['dones'],
                    gamma=float(args.gae_gamma), lam=float(args.gae_lambda)
                )
                traj_high['advantages'] = adv_high.tolist()
                traj_high['returns'] = ret_high.tolist()

            # Compute advantages for low-level
            if len(traj_low['rewards']) > 0:
                adv_low, ret_low = compute_gae(
                    traj_low['rewards'], traj_low['values'], traj_low['dones'],
                    gamma=float(args.gae_gamma), lam=float(args.gae_lambda)
                )
                traj_low['advantages'] = adv_low.tolist()
                traj_low['returns'] = ret_low.tolist()

            # PPO update
            if len(traj_high['states']) > 0 and len(traj_low['states']) > 0:
                ppo_update_hrl(
                    policy.high_level, policy.low_level,
                    optimizer_high, optimizer_low,
                    traj_high, traj_low,
                    clip=float(args.ppo_clip),
                    value_coef=float(args.ppo_value_coef),
                    entropy_coef=float(args.ppo_entropy_coef),
                    epochs=int(args.ppo_epochs),
                    bs=int(args.minibatch_size),
                    device=device,
                )

            # Evaluation metrics from episode stats
            if len(episode_stats) > 0:
                mean_return = float(np.mean([s['return'] for s in episode_stats]))
                mean_adh = float(np.mean([s['adherence'] for s in episode_stats]))
                
                # Compute canonical score (positional match vs [0,1,2,3])
                canonical_scores = []
                ref = [0, 1, 2, 3]
                for s in episode_stats:
                    vseq = s['visited_sequence']
                    _, weight, _ = positional_match_metrics(vseq, ref)
                    canonical_scores.append(float(weight) + float(args.per_step_penalty) * float(len(vseq)))
                mean_canonical = float(np.mean(canonical_scores)) if canonical_scores else 0.0
                
                # Mode sequence
                from collections import Counter
                vseqs = [tuple(s['visited_sequence']) for s in episode_stats]
                if len(vseqs) > 0:
                    c = Counter(vseqs)
                    mode_seq, mode_count = c.most_common(1)[0]
                    mode_frac = float(mode_count) / float(len(vseqs))
                else:
                    mode_seq, mode_frac = [], 0.0
                
                env_eval_history.append(mean_canonical)
                last_mean_adherence = float(mean_adh)
                
                print(f"  [HRL Maze wf {list(wf)}] Update {update:3d} | Shaped {mean_return:7.2f} | Canonical {mean_canonical:7.2f} | Adh {mean_adh:5.1%} | Visit {list(mode_seq)} ({mode_frac:5.1%})", flush=True)
                
                # Early stopping on adherence
                if mean_adh >= float(args.adh_eval_threshold):
                    env_eval_when_adh_thresh.append(mean_canonical)
                    if score_on_adh_thresh is None:
                        score_on_adh_thresh = float(mean_canonical)
                    if len(env_eval_when_adh_thresh) >= 3 and env_eval_when_adh_thresh[-1] <= env_eval_when_adh_thresh[-2] <= env_eval_when_adh_thresh[-3]:
                        print(f"  [HRL Maze wf {list(wf)}] Early stop on stability")
                        break

                if bool(args.early_stop_on_adherence):
                    if mean_adh >= float(args.adherence_target):
                        consecutive_target_meets += 1
                    else:
                        consecutive_target_meets = 0
                    if consecutive_target_meets >= int(args.adherence_patience):
                        print(f"  [HRL Maze wf {list(wf)}] Early stop on adherence: {mean_adh:.3f}")
                        break
                
                # CSV logging
                try:
                    header = [
                        'workflow', 'visit', 'update', 'total_update',
                        'mean_return_shaped', 'mean_env_return', 'mean_adherence', 'mode_seq', 'mode_frac'
                    ]
                    row = {
                        'workflow': '-'.join(map(str, wf)),
                        'visit': int(vc),
                        'update': int(update),
                        'total_update': int(total_update_idx),
                        'mean_return_shaped': float(mean_return),
                        'mean_env_return': float(mean_canonical),
                        'mean_adherence': float(mean_adh),
                        'mode_seq': ' '.join(map(str, mode_seq)),
                        'mode_frac': float(mode_frac),
                    }
                    write_header = not os.path.exists(updates_csv)
                    with open(updates_csv, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=header)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception:
                    pass

            if global_update_count >= int(MAX_TOTAL_UPDATES):
                print(f"[Search] Stopping: MAX_TOTAL_UPDATES", flush=True)
                hit_global_limit = True
                break

        # Score
        if score_on_adh_thresh is not None:
            score = float(score_on_adh_thresh)
            score_source = f"adh{int(100*args.adh_eval_threshold)}"
        elif len(env_eval_history) > 0:
            score = float(np.max(env_eval_history))
            score_source = 'max_canonical'
        else:
            score = 0.0
            score_source = 'fallback'

        if next_idx not in observed_indices:
            observed_indices.append(next_idx)
            observed_embeddings.append(candidate_embeddings[next_idx])
            observed_scores.append(score)
        else:
            pos = observed_indices.index(next_idx)
            observed_scores[pos] = max(observed_scores[pos], score)

        try:
            current_policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        except Exception:
            current_policy_state = policy.state_dict()
        current_lr = max(float(args.min_lr), float(effective_lr) * float(args.lr_decay))
        current_lr_high = max(float(args.min_lr), float(effective_lr_high) * float(args.lr_decay))
        current_lr_low = max(float(args.min_lr), float(effective_lr_low) * float(args.lr_decay))

        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'iteration': int(it),
                    'workflow': list(wf),
                    'continued': bool(continued),
                    'effective_lr': float(effective_lr),
                    'score': float(score),
                    'score_source': score_source,
                    'mu': float(mu[next_idx]),
                    'std': float(std[next_idx]),
                    'ucb': float(ucb[next_idx]),
                }) + "\n")
        except Exception:
            pass

        it += 1
        if hit_global_limit:
            break
        if int(args.iterations) > 0 and it >= int(args.iterations):
            break

    if len(observed_scores) > 0:
        best_idx = int(np.argmax(observed_scores))
        print("\n=== Hierarchical RL Maze Workflow Search Completed ===")
        print(f"Observed {len(observed_indices)} unique workflows (out of 24 total)")
        print(f"Best workflow: {candidates[observed_indices[best_idx]]} with score {observed_scores[best_idx]:.2f}")
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump({
                'observed_count': int(len(observed_indices)),
                'total_candidates': 24,
                'best_workflow': candidates[observed_indices[best_idx]],
                'best_score': float(observed_scores[best_idx])
            }, f, indent=2)


if __name__ == '__main__':
    main()
