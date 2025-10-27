"""
Workflow Search with GP-UCB (Allow-Revisit, Diverse MP) on DiagonalCornersEnv.

- Always argmax UCB (no epsilon-greedy)
- Allows revisits by default
- Global-only continuation with LR decay and floor
- Multiprocessing rollouts with per-worker RNG seeding and optional 'spawn'
- Early switch/eval when adherence crosses configurable threshold
"""

from workflow_search_gpucb import *  # reuse all helpers, env wrappers, PPO, GP utils

import multiprocessing as mp
import gym
import torch.nn as nn
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Hard cap on total PPO updates across the whole search (replacing CLI flag)
MAX_TOTAL_UPDATES = 10000


def _child_vectorized_rollout(worker_id,
                              policy_state_dict,
                              workflow,
                              num_envs,
                              max_steps,
                              device_str,
                              gamma,
                              shaping_coef,
                              penalty_revisit,
                              penalty_future,
                              penalty_offworkflow,
                              per_step_penalty,
                              debug_rewards,
                              debug_rewards_env,
                              debug_rewards_steps,
                              deterministic_rollouts,
                              worker_seed):
    import numpy as _np
    import torch as _torch

    _np.random.seed(int(worker_seed))
    _torch.manual_seed(int(worker_seed))

    device = _torch.device(device_str)
    # Reconstruct policy
    state_dim = 2 + 4 * 2 + 4
    wf_dim = 4
    policy = WorkflowPolicy(state_dim, wf_dim).to(device)
    if policy_state_dict is not None:
        try:
            policy.load_state_dict(policy_state_dict)
        except Exception:
            pass

    trajs = rollout_shaped_vectorized(
        policy,
        list(workflow),
        int(num_envs),
        int(max_steps),
        device,
        float(gamma),
        float(shaping_coef),
        float(penalty_revisit),
        float(penalty_future),
        float(penalty_offworkflow),
        float(per_step_penalty),
        bool(debug_rewards),
        int(debug_rewards_env),
        int(debug_rewards_steps),
        bool(deterministic_rollouts),
    )
    return trajs


class ShapedWorkflowEnv(gym.Wrapper):
    """Gym wrapper that:
    - Appends next-target one-hot (size 4) for the selected workflow to the observation
    - Replaces reward with potential-based shaping and penalties used in our vectorized rollouts
    """
    def __init__(self, env: gym.Env, workflow: List[int], gamma: float, shaping_coef: float,
                 penalty_revisit: float, penalty_future: float, penalty_offworkflow: float,
                 per_step_penalty: float):
        super().__init__(env)
        self.workflow = list(workflow)
        self.gamma = float(gamma)
        self.shaping_coef = float(shaping_coef)
        self.penalty_revisit = float(penalty_revisit)
        self.penalty_future = float(penalty_future)
        self.penalty_offworkflow = float(penalty_offworkflow)
        self.per_step_penalty = float(per_step_penalty)
        self.visited_sequence: List[int] = []
        self._phi_s: float = 0.0
        # Expand observation to match custom policy: [agent_norm(2), targets_norm(8), visited_flags(4), next_target_one_hot(4), workflow_encoding(4)]
        import numpy as _np
        self.observation_space = gym.spaces.Box(
            low=_np.zeros(2 + 4 * 2 + 4 + 4 + 4, dtype=_np.float32),
            high=_np.ones(2 + 4 * 2 + 4 + 4 + 4, dtype=_np.float32),
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
        return self._augment_obs(obs), info

    def step(self, action):
        step_result = self.env.step(action)
        # Support gym (4-tuple) and gymnasium (5-tuple)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, _env_r, done, truncated, info = step_result
        else:
            obs, _env_r, done, info = step_result
            truncated = False
        pre_step_visited = set(self.visited_sequence)
        # Detect landing on any target
        landed_idx = None
        try:
            for idx, pos in enumerate(self.env.target_positions):
                if tuple(self.env.agent_pos) == tuple(pos):
                    landed_idx = int(idx)
                    break
        except Exception:
            landed_idx = None

        # Update visited sequence if first visit
        if landed_idx is not None and landed_idx not in pre_step_visited:
            self.visited_sequence.append(landed_idx)

        # Potential-based shaping (exclude env reward)
        phi_s = self._phi_s
        phi_s2 = self._compute_phi()
        shaped = self.shaping_coef * (self.gamma * phi_s2 - phi_s)

        # Additional penalties and bonuses
        if landed_idx is not None:
            if landed_idx in pre_step_visited:
                shaped += self.penalty_revisit
            else:
                # Determine immediate next required target by correct prefix
                prefix_ok = 0
                for i, t in enumerate(self.visited_sequence):
                    if i < len(self.workflow) and t == self.workflow[i]:
                        prefix_ok += 1
                    else:
                        break
                req_idx = min(prefix_ok, len(self.workflow) - 1)
                required_target = self.workflow[req_idx]
                if int(landed_idx) == int(required_target):
                    # Bonus for correct-order target hit
                    shaped += 10.0
                elif landed_idx in self.workflow:
                    shaped += self.penalty_future
                else:
                    shaped += self.penalty_offworkflow

        shaped += self.per_step_penalty

        # Update stored potential
        self._phi_s = phi_s2

        # Info for adherence / diagnostics
        try:
            prefix_ok = 0
            for i, t in enumerate(self.visited_sequence):
                if i < len(self.workflow) and t == self.workflow[i]:
                    prefix_ok += 1
                else:
                    break
            adherence = float(prefix_ok) / float(len(self.workflow))
            success = (prefix_ok == len(self.workflow))
            info = dict(info)
            info['visited_sequence'] = list(self.visited_sequence)
            info['adherence'] = float(adherence)
            info['success'] = bool(success)
        except Exception:
            pass

        return self._augment_obs(obs), float(shaped), done, truncated, info

    def _augment_obs(self, obs):
        import numpy as _np
        # Normalize agent position
        grid_norm = float(getattr(self.env, 'grid_size', 11) - 1)
        if isinstance(obs, _np.ndarray):
            agent_r = float(obs[0]) / grid_norm
            agent_c = float(obs[1]) / grid_norm
        else:
            agent_r = float(_to_numpy(obs)[0]) / grid_norm
            agent_c = float(_to_numpy(obs)[1]) / grid_norm
        agent_norm = _np.asarray([agent_r, agent_c], dtype=_np.float32)

        # Targets normalized
        targets = []
        try:
            for (r, c) in self.env.target_positions:
                targets.append(float(r) / grid_norm)
                targets.append(float(c) / grid_norm)
        except Exception:
            targets = [0.0] * (4 * 2)
        targets_norm = _np.asarray(targets, dtype=_np.float32)

        # Visited flags from wrapper sequence
        visited_flags = _np.zeros(4, dtype=_np.float32)
        for t in self.visited_sequence:
            if 0 <= int(t) < 4:
                visited_flags[int(t)] = 1.0

        # Next-target one-hot
        one_hot = _np.zeros(4, dtype=_np.float32)
        next_idx = 0
        for i, t in enumerate(self.workflow):
            if t in self.visited_sequence:
                next_idx = i + 1
            else:
                break
        if next_idx >= len(self.workflow):
            next_target = self.workflow[-1]
        else:
            next_target = self.workflow[next_idx]
        if 0 <= int(next_target) < 4:
            one_hot[int(next_target)] = 1.0

        # Workflow encoding vector
        try:
            wf_vec = _to_numpy(workflow_to_vector(self.workflow, num_targets=4))
        except Exception:
            wf_vec = _np.zeros(4, dtype=_np.float32)

        return _np.concatenate([agent_norm, targets_norm, visited_flags, one_hot, wf_vec], axis=0)

    def _compute_phi(self) -> float:
        try:
            # Current target by visited prefix
            next_idx = 0
            for i, t in enumerate(self.workflow):
                if t in self.visited_sequence:
                    next_idx = i + 1
                else:
                    break
            if next_idx >= len(self.workflow):
                next_idx = len(self.workflow) - 1
            required_target_id = self.workflow[next_idx]
            cur_target = self.env.target_positions[required_target_id]
            return -float(manhattan(tuple(self.env.agent_pos), tuple(cur_target)))
        except Exception:
            return 0.0


def _to_numpy(x):
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x.astype(_np.float32)
    try:
        return _np.asarray(x, dtype=_np.float32)
    except Exception:
        return _np.array(x, dtype=_np.float32)


def evaluate_model_canonical_sb3(model, workflow: List[int], episodes: int, max_steps: int) -> Tuple[float, float]:
    """Run deterministic eval episodes and return (canonical_mean_return, mean_adherence)."""
    returns = []
    adherences = []
    for _ in range(int(episodes)):
        env = DiagonalCornersEnv(max_steps=int(max_steps))
        # Use same observation augmentation as training for the policy input
        _reset = env.reset()
        if isinstance(_reset, tuple) and len(_reset) == 2:
            obs, _info = _reset
        else:
            obs, _info = _reset, {}
        visited_sequence = []
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            # Build next-target one-hot for policy input
            import numpy as _np
            one_hot = _np.zeros(4, dtype=_np.float32)
            next_idx = 0
            for i, t in enumerate(workflow):
                if t in visited_sequence:
                    next_idx = i + 1
                else:
                    break
            if next_idx >= len(workflow):
                next_idx = len(workflow) - 1
            one_hot[int(workflow[next_idx])] = 1.0
            obs_in = _np.concatenate([_to_numpy(obs), one_hot], axis=0)
            action, _ = model.predict(obs_in, deterministic=True)
            _step = env.step(int(action))
            if isinstance(_step, tuple) and len(_step) == 5:
                obs, _r, done, truncated, _info = _step
            else:
                obs, _r, done, _info = _step
                truncated = False
            # record visits
            try:
                for idx, pos in enumerate(env.target_positions):
                    if tuple(env.agent_pos) == tuple(pos):
                        if idx not in visited_sequence:
                            visited_sequence.append(idx)
                        break
            except Exception:
                pass
            steps += 1
            if steps >= int(max_steps):
                break

        # Canonical positional metric vs [0,1,2,3]
        ref = [0, 1, 2, 3]
        _, weight, frac = positional_match_metrics(visited_sequence, ref)
        step_pen = -0.01
        returns.append(float(weight) + step_pen * float(steps))
        # Adherence to the selected workflow
        prefix_ok = 0
        for i, t in enumerate(visited_sequence):
            if i < len(workflow) and t == workflow[i]:
                prefix_ok += 1
            else:
                break
        adherences.append(float(prefix_ok) / float(len(workflow)))

    import numpy as _np
    mean_return = float(_np.mean(returns)) if len(returns) > 0 else 0.0
    mean_adh = float(_np.mean(adherences)) if len(adherences) > 0 else 0.0
    return mean_return, mean_adh

def evaluate_model_shaped_sb3(model,
                              workflow: List[int],
                              episodes: int,
                              max_steps: int,
                              gamma: float,
                              shaping_coef: float,
                              penalty_revisit: float,
                              penalty_future: float,
                              penalty_offworkflow: float,
                              per_step_penalty: float) -> float:
    """Run deterministic eval episodes on the shaped wrapper and return mean shaped return."""
    returns = []
    for _ in range(int(episodes)):
        base = DiagonalCornersEnv(max_steps=int(max_steps))
        env = ShapedWorkflowEnv(
            base,
            list(workflow),
            gamma=float(gamma),
            shaping_coef=float(shaping_coef),
            penalty_revisit=float(penalty_revisit),
            penalty_future=float(penalty_future),
            penalty_offworkflow=float(penalty_offworkflow),
            per_step_penalty=float(per_step_penalty),
        )
        _reset = env.reset()
        if isinstance(_reset, tuple) and len(_reset) == 2:
            obs, _info = _reset
        else:
            obs, _info = _reset, {}
        total_r = 0.0
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            _step = env.step(int(action))
            if isinstance(_step, tuple) and len(_step) == 5:
                obs, r, done, truncated, _info = _step
            else:
                obs, r, done, _info = _step
                truncated = False
            total_r += float(r)
            steps += 1
            if steps >= int(max_steps):
                break
        returns.append(total_r)
    import numpy as _np
    return float(_np.mean(returns)) if len(returns) > 0 else 0.0

def rollout_shaped_multiprocessing_seeded(policy,
                                          workflow,
                                          num_envs,
                                          max_steps,
                                          device,
                                          gamma,
                                          shaping_coef,
                                          penalty_revisit,
                                          penalty_future,
                                          penalty_offworkflow,
                                          per_step_penalty,
                                          debug_rewards,
                                          debug_rewards_env,
                                          debug_rewards_steps,
                                          deterministic_rollouts,
                                          worker_seed_base,
                                          spawn):
    # Configure start method
    ctx = mp.get_context('spawn' if bool(spawn) else 'fork')

    workers = int(num_envs)
    # Split envs across workers (1 env per worker)
    envs_per_worker = [1 for _ in range(workers)]

    policy_state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    # To avoid CUDA use in forked/spawned workers unintentionally, place children on CPU
    device_str = 'cpu'

    args_list = []
    for wid, nev in enumerate(envs_per_worker):
        seed = int(worker_seed_base) + int(wid)
        args_list.append((
            int(wid),
            policy_state_dict,
            list(workflow),
            int(nev),
            int(max_steps),
            device_str,
            float(gamma),
            float(shaping_coef),
            float(penalty_revisit),
            float(penalty_future),
            float(penalty_offworkflow),
            float(per_step_penalty),
            bool(debug_rewards),
            int(debug_rewards_env),
            int(debug_rewards_steps),
            bool(deterministic_rollouts),
            int(seed),
        ))

    results = []
    with ctx.Pool(processes=workers) as pool:
        for trajs in pool.starmap(_child_vectorized_rollout, args_list):
            results.extend(trajs)
    return results


def main():
    parser = argparse.ArgumentParser()
    # Base search options
    parser.add_argument('--iterations', type=int, default=0, help='Optional max selections (0 = no max; stop by UCB criterion if enabled)')
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--num_envs', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--shaping_coef', type=float, default=1.0)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--gae_gamma', type=float, default=0.99)
    # PPO loss hyperparameters
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--ppo_value_coef', type=float, default=0.5)
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.0)
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5)
    parser.add_argument('--per_step_penalty', type=float, default=-0.01)
    parser.add_argument('--eval_episodes_per_update', type=int, default=1)
    parser.add_argument('--final_eval_episodes', type=int, default=5)
    parser.add_argument('--eval_parallel', action='store_true')
    parser.add_argument('--eval_parallel_num_envs', type=int, default=25)
    parser.add_argument('--eval_use_canonical', dest='eval_use_canonical', action='store_true', default=True)
    parser.add_argument('--no_eval_use_canonical', dest='eval_use_canonical', action='store_false')
    # Training behavior
    parser.add_argument('--deterministic_rollouts', action='store_true')
    parser.add_argument('--adherence_target', type=float, default=1.0)
    parser.add_argument('--adherence_patience', type=int, default=1)
    parser.add_argument('--early_stop_on_adherence', action='store_true')
    parser.add_argument('--adh_eval_threshold', type=float, default=0.9, help='Trigger early evaluation/switch when mean adherence >= this threshold')
    # Penalties
    parser.add_argument('--penalty_revisit', type=float, default=-2.0)
    parser.add_argument('--penalty_future', type=float, default=-100.0)
    parser.add_argument('--penalty_offworkflow', type=float, default=-50.0)
    # GP / kernel
    parser.add_argument('--length_scale', type=float, default=0.75)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--kernel_type', type=str, default='rbf_possunmatch', choices=['rbf_rank','rbf_pairwise','rbf_mixed','rbf_posunmatch','rbf_possunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=1e-4)
    parser.add_argument('--kappa', type=float, default=4.0)
    # Early stop
    parser.add_argument('--enable_ucb_early_stop', action='store_true', help='Enable UCB-based early stopping')
    parser.add_argument('--stop_min_explored', type=int, default=2)
    # Revisits default enabled
    parser.add_argument('--allow_revisits', dest='allow_revisits', action='store_true', help='Allow selecting previously observed workflows (default)')
    parser.add_argument('--no_revisits', dest='allow_revisits', action='store_false', help='Disallow revisiting already observed workflows')
    parser.set_defaults(allow_revisits=True)
    # Stability: continuation and LR decay (global only)
    parser.add_argument('--continue_policy', action='store_true', help='Continue training from existing global policy state')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Multiplicative LR decay applied after each workflow selection')
    parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum LR floor when decaying')
    # MP diversity controls
    parser.add_argument('--spawn_mp', action='store_true', help='Use spawn start method to avoid RNG inheritance')
    parser.add_argument('--worker_seed_base', type=int, default=None, help='Base for per-worker seeds (defaults to --seed)')
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='gpucb_diagonal_allowrevisit_diverse')
    parser.add_argument('--use_sb3', action='store_true', help='Use Stable-Baselines3 PPO for training instead of custom PPO')
    parser.add_argument('--use_mp', action='store_true')
    parser.set_defaults(use_mp=True)
    parser.add_argument('--debug_rewards', action='store_true')
    parser.add_argument('--debug_rewards_env', type=int, default=0)
    parser.add_argument('--debug_rewards_steps', type=int, default=200)

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
        }, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Enforce spawn on CUDA to avoid fork+Cuda initialization errors
    try:
        if device.type == 'cuda':
            mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    spawn_mp_effective = bool(args.spawn_mp or (device.type == 'cuda'))
    if device.type == 'cuda' and not bool(args.spawn_mp):
        print("[MP] CUDA detected; forcing spawn start method to avoid fork/CUDA errors. Use --spawn_mp to make this explicit.", flush=True)

    # Candidate workflows
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

    # GP observations
    observed_indices: List[int] = []
    observed_embeddings: List[np.ndarray] = []
    observed_scores: List[float] = []

    # For revisits: track per-workflow visit counts (continuation is global-only)
    visit_counts: Dict[Tuple[int,int,int,int], int] = {}
    # Global continuation state
    current_policy_state = None
    # Separate continuation state for SB3 policy
    current_sb3_policy_state = None
    current_lr = float(args.lr)

    it = 0
    global_update_count = 0
    hit_global_limit = False

    while True:
        # Posterior on all
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

        # Early stop (optional)
        if args.enable_ucb_early_stop and len(observed_indices) >= int(args.stop_min_explored):
            unexplored = [i for i in range(len(candidates)) if i not in observed_indices]
            if len(unexplored) > 0:
                best_score = float(np.max(observed_scores)) if len(observed_scores) > 0 else -np.inf
                max_ucb_unexplored = float(np.max(ucb[unexplored]))
                if max_ucb_unexplored <= best_score:
                    print(f"[GP-UCB] Early stop: max UCB(unexplored) {max_ucb_unexplored:.2f} <= best_score {best_score:.2f}")
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            'iteration': it,
                            'event': 'early_stop',
                            'best_score': best_score,
                            'max_ucb_unexplored': max_ucb_unexplored,
                            'stop_min_explored': int(args.stop_min_explored)
                        }) + "\n")
                    break

        # Selection: pure GP-UCB argmax; if revisits disallowed, mask explored
        ucb_for_choice = ucb.copy()
        if not bool(args.allow_revisits):
            ucb_for_choice[observed_indices] = -np.inf
        max_val = float(np.max(ucb_for_choice))
        tie_indices = [i for i, val in enumerate(ucb_for_choice) if val == max_val]
        next_idx = int(np.random.choice(tie_indices))
        print(f"[GP-UCB] Iter {it} selecting workflow {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")

        wf = tuple(candidates[next_idx])

        # Policy initialization or continuation (global-only)
        start_lr = float(args.lr)
        continued = False
        effective_lr = start_lr

        # Build learner depending on branch
        if bool(args.use_sb3):
            # SB3 continuation is handled after model creation
            pass
        else:
            state_dict = None
            if bool(args.continue_policy) and current_policy_state is not None:
                state_dict = current_policy_state
                effective_lr = float(current_lr)
                continued = True
            # Build custom policy and optimizer
            state_dim = 2 + 4 * 2 + 4
            wf_dim = 4
            policy = WorkflowPolicy(state_dim, wf_dim).to(device)
            if state_dict is not None:
                try:
                    policy.load_state_dict(state_dict)
                except Exception:
                    pass
            optimizer = optim.Adam(policy.parameters(), lr=effective_lr)

        updates_run = 0
        env_eval_history = []
        updates_csv = os.path.join(run_dir, 'updates.csv')
        adherence_history = []
        success_history = []
        env_eval_when_adh_thresh = []
        consecutive_target_meets = 0
        last_mean_adherence = 0.0
        vc = visit_counts.get(wf, 0) + 1
        visit_counts[wf] = vc
        score_on_adh_thresh = None

        worker_seed_base = int(args.worker_seed_base) if args.worker_seed_base is not None else int(args.seed)

        # SB3 PPO branch
        if bool(args.use_sb3):
            # Build vectorized shaped env(s) (seeding per sub-env for diversity)
            def make_env_fn(rank: int):
                def _init():
                    base = DiagonalCornersEnv(max_steps=int(args.max_steps))
                    env = ShapedWorkflowEnv(
                        base,
                        list(wf),
                        gamma=float(args.gamma),
                        shaping_coef=float(args.shaping_coef),
                        penalty_revisit=float(args.penalty_revisit),
                        penalty_future=float(args.penalty_future),
                        penalty_offworkflow=float(args.penalty_offworkflow),
                        per_step_penalty=float(args.per_step_penalty),
                    )
                    # Per-env seed for diversity
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
            vec_env = DummyVecEnv([make_env_fn(i) for i in range(int(args.num_envs))])
            # Vec-level monitor: writes one CSV with episode rewards/lengths across envs
            mon_dir = os.path.join(run_dir, 'monitor')
            os.makedirs(mon_dir, exist_ok=True)
            vec_env = VecMonitor(vec_env, filename=os.path.join(mon_dir, 'monitor.csv'))
            # SB3 policy architecture with LayerNorm-like features extractor
            class LayerNormMLP(BaseFeaturesExtractor):
                def __init__(self, observation_space, features_dim=128):
                    super().__init__(observation_space, features_dim)
                    n_obs = int(observation_space.shape[0])
                    self.net = nn.Sequential(
                        nn.Linear(n_obs, 128),
                        nn.LayerNorm(128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.LayerNorm(128),
                        nn.ReLU(),
                    )
                    for m in self.net:
                        if isinstance(m, nn.Linear):
                            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                            nn.init.zeros_(m.bias)
                    self._features_dim = 128

                def forward(self, x):
                    return self.net(x)

            policy_kwargs = dict(
                features_extractor_class=LayerNormMLP,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=[],          # direct heads on features
                activation_fn=nn.ReLU,
                ortho_init=False,
            )
            # Map our hyperparams
            sb3_model = SB3PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=float(effective_lr),
                n_steps=int(args.max_steps),
                batch_size=int(args.minibatch_size),
                n_epochs=int(args.ppo_epochs),
                gamma=float(args.gae_gamma),
                clip_range=float(args.ppo_clip),
                ent_coef=float(args.ppo_entropy_coef),
                vf_coef=float(args.ppo_value_coef),
                max_grad_norm=float(args.ppo_max_grad_norm),
                policy_kwargs=policy_kwargs,
                seed=int(args.seed),
                verbose=0,
                device=device,
            )

            # SB3 continuation
            if bool(args.continue_policy) and current_sb3_policy_state is not None:
                try:
                    sb3_model.policy.load_state_dict(current_sb3_policy_state)
                    continued = True
                except Exception:
                    pass

            # Callback to aggregate shaped/canonical/adherence from training rollouts
            class TrainingMetricsCallback(BaseCallback):
                def __init__(self, n_envs: int, step_penalty: float, verbose: int = 0):
                    super().__init__(verbose)
                    self.n_envs = int(n_envs)
                    self.step_penalty = float(step_penalty)
                    self.env_shaped = [0.0 for _ in range(self.n_envs)]
                    self.env_steps = [0 for _ in range(self.n_envs)]
                    self.env_vseq = [[] for _ in range(self.n_envs)]
                    self.ep_shaped = []
                    self.ep_canonical = []
                    self.ep_adh = []
                    self.ep_vseqs = []

                def _on_step(self) -> bool:
                    rewards = self.locals.get('rewards', None)
                    infos = self.locals.get('infos', None)
                    dones = self.locals.get('dones', None)
                    if rewards is None or infos is None or dones is None:
                        return True
                    for i in range(self.training_env.num_envs):
                        r = float(rewards[i])
                        self.env_shaped[i] += r
                        self.env_steps[i] += 1
                        info = infos[i]
                        vseq = info.get('visited_sequence', None)
                        if vseq is not None:
                            self.env_vseq[i] = list(vseq)
                        if bool(dones[i]):
                            ref = [0, 1, 2, 3]
                            _, weight, _ = positional_match_metrics(self.env_vseq[i], ref)
                            canonical = float(weight) + float(self.step_penalty) * float(self.env_steps[i])
                            adh = float(info.get('adherence', 0.0))
                            self.ep_shaped.append(self.env_shaped[i])
                            self.ep_canonical.append(canonical)
                            self.ep_adh.append(adh)
                            try:
                                self.ep_vseqs.append(list(self.env_vseq[i]))
                            except Exception:
                                pass
                            self.env_shaped[i] = 0.0
                            self.env_steps[i] = 0
                            self.env_vseq[i] = []
                    return True

                def get_means(self):
                    if len(self.ep_shaped) == 0:
                        return 0.0, 0.0, 0.0
                    import numpy as _np
                    return (
                        float(_np.mean(self.ep_shaped)),
                        float(_np.mean(self.ep_canonical)),
                        float(_np.mean(self.ep_adh)),
                    )

                def get_mode_seq(self):
                    try:
                        from collections import Counter
                        if len(self.ep_vseqs) == 0:
                            return [], 0.0
                        c = Counter([tuple(s) for s in self.ep_vseqs])
                        (mode_t, count) = c.most_common(1)[0]
                        frac = float(count) / float(len(self.ep_vseqs))
                        return list(mode_t), float(frac)
                    except Exception:
                        return [], 0.0

            # Train per-update chunks to mirror custom PPO and enable early stopping
            steps_per_update = int(args.num_envs) * int(args.max_steps)
            for update in range(int(args.updates)):
                global_update_count += 1
                total_update_idx = int(global_update_count)
                cb = TrainingMetricsCallback(n_envs=vec_env.num_envs, step_penalty=float(args.per_step_penalty))
                sb3_model.learn(total_timesteps=steps_per_update, reset_num_timesteps=False, progress_bar=False, callback=cb)

                shaped_mean, eval_env_return, mean_adh = cb.get_means()
                mode_seq, mode_frac = cb.get_mode_seq()
                env_eval_history.append(eval_env_return)
                last_mean_adherence = float(mean_adh)
                print(f"  [SB3 Train wf {list(wf)}] Update {update:3d} | Shaped {shaped_mean:7.2f} | Canonical {eval_env_return:7.2f} | Adh {mean_adh:5.1%} | Visit {mode_seq} ({mode_frac:5.1%})", flush=True)
                updates_run = update + 1

                # Threshold-triggered early switch score and stability early stop
                if mean_adh >= float(args.adh_eval_threshold):
                    env_eval_when_adh_thresh.append(eval_env_return)
                    if score_on_adh_thresh is None:
                        score_on_adh_thresh = float(eval_env_return)
                    if len(env_eval_when_adh_thresh) >= 3 and env_eval_when_adh_thresh[-1] <= env_eval_when_adh_thresh[-2] <= env_eval_when_adh_thresh[-3]:
                        print(f"  [SB3 Train wf {list(wf)}] Early stop on canonical-eval stability (adh>={args.adh_eval_threshold:.2f}): last3={env_eval_when_adh_thresh[-3:]} (non-increasing)")
                        break

                # Adherence-based early stop (optional)
                if bool(args.early_stop_on_adherence):
                    if mean_adh >= float(args.adherence_target):
                        consecutive_target_meets += 1
                    else:
                        consecutive_target_meets = 0
                    if consecutive_target_meets >= int(args.adherence_patience):
                        print(f"  [SB3 Train wf {list(wf)}] Early stop on adherence: mean_adherence={mean_adh:.3f} for {consecutive_target_meets} consecutive updates")
                        break

                if global_update_count >= int(MAX_TOTAL_UPDATES):
                    print(f"[Search] Stopping: reached MAX_TOTAL_UPDATES={int(MAX_TOTAL_UPDATES)}", flush=True)
                    hit_global_limit = True
                    break

                # Logging with visit count and total_update index (approximate fields for SB3)
                try:
                    header = [
                        'workflow', 'visit', 'update', 'total_update',
                        'mean_return_shaped', 'mean_env_return', 'mean_adherence', 'success_rate', 'avg_ep_len', 'mode_seq', 'mode_frac',
                        'policy_loss', 'value_loss', 'entropy'
                    ]
                    row = {
                        'workflow': '-'.join(map(str, wf)),
                        'visit': int(vc),
                        'update': int(update),
                        'total_update': int(total_update_idx),
                        'mean_return_shaped': float(shaped_mean),
                        'mean_env_return': float(eval_env_return),
                        'mean_adherence': float(mean_adh),
                        'success_rate': float(0.0),
                        'avg_ep_len': float(0.0),
                        'mode_seq': ' '.join(map(str, mode_seq)),
                        'mode_frac': float(mode_frac),
                        'policy_loss': float(0.0),
                        'value_loss': float(0.0),
                        'entropy': float(0.0),
                    }
                    write_header = not os.path.exists(updates_csv)
                    with open(updates_csv, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=header)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception:
                    pass

        else:
            for update in range(int(args.updates)):
                global_update_count += 1
            total_update_idx = int(global_update_count)

            if bool(args.use_mp):
                batch_trajs = rollout_shaped_multiprocessing_seeded(
                    policy,
                    list(wf),
                    int(args.num_envs),
                    int(args.max_steps),
                    device,
                    float(args.gamma),
                    float(args.shaping_coef),
                    float(args.penalty_revisit),
                    float(args.penalty_future),
                    float(args.penalty_offworkflow),
                    float(args.per_step_penalty),
                    bool(args.debug_rewards),
                    int(args.debug_rewards_env),
                    int(args.debug_rewards_steps),
                    bool(args.deterministic_rollouts),
                    int(worker_seed_base),
                    bool(spawn_mp_effective),
                )
            else:
                batch_trajs = rollout_shaped_vectorized(
                    policy,
                    list(wf),
                    int(args.num_envs),
                    int(args.max_steps),
                    device,
                    float(args.gamma),
                    float(args.shaping_coef),
                    float(args.penalty_revisit),
                    float(args.penalty_future),
                    float(args.penalty_offworkflow),
                    float(args.per_step_penalty),
                    bool(args.debug_rewards),
                    int(args.debug_rewards_env),
                    int(args.debug_rewards_steps),
                    bool(args.deterministic_rollouts),
                )

            batch = {k: [] for k in ['states', 'workflows', 'actions', 'logps', 'rewards', 'values', 'dones']}
            for tr in batch_trajs:
                for k in batch.keys():
                    batch[k].extend(tr[k])
            advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['dones'], gamma=float(args.gae_gamma))
            batch['advantages'] = advantages
            batch['returns'] = returns

            stats = ppo_update(
                policy,
                optimizer,
                batch,
                clip=float(args.ppo_clip),
                value_coef=float(args.ppo_value_coef),
                entropy_coef=float(args.ppo_entropy_coef),
                epochs=int(args.ppo_epochs),
                bs=int(args.minibatch_size),
                max_grad_norm=float(args.ppo_max_grad_norm),
                device=device,
            )

            mean_return = float(np.mean([tr['ep_return'] for tr in batch_trajs]))
            # Canonical env-only metric during training
            if bool(args.eval_use_canonical):
                ref = [0, 1, 2, 3]
                step_pen = -0.01
                pos_rewards = []
                for tr in batch_trajs:
                    seq = tr.get('visited_sequence', [])
                    steps = len(tr.get('dones', []))
                    _, weight, _ = positional_match_metrics(seq, ref)
                    pos_rewards.append(float(weight) + step_pen * float(steps))
                eval_env_return = float(np.mean(pos_rewards)) if len(pos_rewards) > 0 else 0.0
            else:
                eval_env_return = float(np.mean([tr.get('env_ep_return', 0.0) for tr in batch_trajs]))
            env_eval_history.append(eval_env_return)
            # Adherence metrics and early stopping
            mean_adherence = float(np.mean([tr.get('adherence', 0.0) for tr in batch_trajs]))
            success_rate = float(np.mean([1.0 if tr.get('success', False) else 0.0 for tr in batch_trajs]))
            ep_lengths = [len(tr.get('dones', [])) for tr in batch_trajs]
            avg_ep_len = float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else 0.0
            try:
                from collections import Counter
                seqs = [tuple(tr.get('visited_sequence', [])) for tr in batch_trajs]
                mode_seq = []
                mode_frac = 0.0
                if len(seqs) > 0:
                    c = Counter(seqs)
                    mode_seq_t, mode_count = c.most_common(1)[0]
                    mode_seq = list(mode_seq_t)
                    mode_frac = float(mode_count) / float(len(seqs))
            except Exception:
                mode_seq = []
                mode_frac = 0.0
            adherence_history.append(mean_adherence)
            success_history.append(success_rate)
            last_mean_adherence = mean_adherence
            # Real-time console print
            current_lr_now = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else effective_lr
            print(f"  [Train wf {list(wf)}] Update {update:3d} | Return {mean_return:7.2f} | Adh {mean_adherence:5.1%} | Succ {success_rate:5.1%} | Len {avg_ep_len:5.1f} | LR {current_lr_now:.2e} | Visit {mode_seq} ({mode_frac:5.1%})", flush=True)
            if bool(args.eval_use_canonical):
                print(f"    EvalEnv canonical (mean): {eval_env_return:7.2f}", flush=True)
            else:
                print(f"    RolloutEnv (mean env-only over batch): {eval_env_return:7.2f}", flush=True)

            # Threshold-triggered early switch
            if mean_adherence >= float(args.adh_eval_threshold) and score_on_adh_thresh is None:
                env_eval_when_adh_thresh.append(eval_env_return)
                score_on_adh_thresh = float(eval_env_return)

            # Early stop on adherence target (optional legacy)
            if bool(args.early_stop_on_adherence):
                if mean_adherence >= float(args.adherence_target):
                    consecutive_target_meets += 1
                else:
                    consecutive_target_meets = 0
                if consecutive_target_meets >= int(args.adherence_patience):
                    print(f"  [Train wf {list(wf)}] Early stop on adherence: mean_adherence={mean_adherence:.3f} for {consecutive_target_meets} consecutive updates")
                    updates_run = update + 1
                    break

            # Canonical-eval stability early stop when threshold crossed
            if len(env_eval_when_adh_thresh) >= 3 and env_eval_when_adh_thresh[-1] <= env_eval_when_adh_thresh[-2] <= env_eval_when_adh_thresh[-3]:
                print(f"  [Train wf {list(wf)}] Early stop on canonical-eval stability (adh>={args.adh_eval_threshold:.2f}): last3={env_eval_when_adh_thresh[-3:]} (non-increasing)")
                updates_run = update + 1
                pass

            # Logging with visit count and total_update index
            try:
                header = [
                    'workflow', 'visit', 'update', 'total_update',
                    'mean_return_shaped', 'mean_env_return', 'mean_adherence', 'success_rate', 'avg_ep_len', 'mode_seq', 'mode_frac',
                    'policy_loss', 'value_loss', 'entropy'
                ]
                row = {
                    'workflow': '-'.join(map(str, wf)),
                    'visit': int(vc),
                    'update': int(update),
                    'total_update': int(total_update_idx),
                    'mean_return_shaped': float(mean_return),
                    'mean_env_return': float(eval_env_return),
                    'mean_adherence': float(mean_adherence),
                    'success_rate': float(success_rate),
                    'avg_ep_len': float(avg_ep_len),
                    'mode_seq': ' '.join(map(str, mode_seq)),
                    'mode_frac': float(mode_frac),
                    'policy_loss': float(stats.get('policy_loss', 0.0)),
                    'value_loss': float(stats.get('value_loss', 0.0)),
                    'entropy': float(stats.get('entropy', 0.0)),
                }
                write_header = not os.path.exists(updates_csv)
                with open(updates_csv, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=header)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception:
                pass

                updates_run = update + 1
                if int(args.eval_episodes_per_update) <= 0:
                    pass
                if len(env_eval_when_adh_thresh) >= 3 and env_eval_when_adh_thresh[-1] <= env_eval_when_adh_thresh[-2] <= env_eval_when_adh_thresh[-3]:
                    break
                if score_on_adh_thresh is not None:
                    updates_run = update + 1
                    break
                if global_update_count >= int(MAX_TOTAL_UPDATES):
                    print(f"[Search] Stopping: reached MAX_TOTAL_UPDATES={int(MAX_TOTAL_UPDATES)}", flush=True)
                    hit_global_limit = True
                    break

        # Score for GP update
        if score_on_adh_thresh is not None:
            score = float(score_on_adh_thresh)
            score_source = f"adh{int(100*args.adh_eval_threshold)}_canonical_mean"
        elif bool(args.eval_use_canonical) and len(env_eval_history) > 0:
            score = float(np.max(env_eval_history))
            score_source = 'max_canonical_mean_during_training'
        else:
            env = DiagonalCornersEnv(max_steps=int(args.max_steps))
            eval_returns = [rollout_env_only(env, policy, list(wf), device, deterministic=True) for _ in range(int(args.final_eval_episodes))]
            score = float(np.mean(eval_returns))
            score_source = 'final_eval_mean_proposed'

        # Update GP observations (best-of if revisiting)
        if next_idx not in observed_indices:
            observed_indices.append(next_idx)
            observed_embeddings.append(candidate_embeddings[next_idx])
            observed_scores.append(score)
        else:
            pos = observed_indices.index(next_idx)
            observed_scores[pos] = max(observed_scores[pos], score)

        # Update continuation state: global only with LR decay and floor
        if bool(args.use_sb3):
            try:
                current_sb3_policy_state = {k: v.detach().cpu() for k, v in sb3_model.policy.state_dict().items()}
            except Exception:
                current_sb3_policy_state = sb3_model.policy.state_dict()
        else:
            current_policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        current_lr = max(float(args.min_lr), float(effective_lr) * float(args.lr_decay))

        # Log GP state and selection
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'iteration': int(it),
                    'workflow': list(wf),
                    'continued': bool(continued),
                    'effective_lr': float(effective_lr),
                    'score_env_only': float(score),
                    'score_source': score_source,
                    'adherence_last': float(last_mean_adherence),
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

    # Summary
    if len(observed_scores) > 0:
        best_idx = int(np.argmax(observed_scores))
        print("\n=== Allow-Revisit Diverse Workflow Search Completed ===")
        print(f"Observed {len(observed_indices)} unique workflows")
        print(f"Best workflow: {candidates[observed_indices[best_idx]]} with score {observed_scores[best_idx]:.2f}")
        with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
            json.dump({
                'observed_count': int(len(observed_indices)),
                'best_workflow': candidates[observed_indices[best_idx]],
                'best_score': float(observed_scores[best_idx])
            }, f, indent=2)


if __name__ == '__main__':
    main()


