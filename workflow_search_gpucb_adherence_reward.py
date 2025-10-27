"""
Workflow Search with GP-UCB + Adherence-Based Reward Shaping.

Adds explicit adherence-progress rewards:
- Bonus when adherence increases step-to-step
- Penalty when adherence drops
- Keeps all other shaping (potential, target bonuses, penalties)
"""

from workflow_search_gpucb import *  # reuse all helpers, env wrappers, PPO, GP utils

import multiprocessing as mp
import gym
import torch.nn as nn
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Hard cap on total PPO updates across the whole search
MAX_TOTAL_UPDATES = 10000


class AdherenceShapedWorkflowEnv(gym.Wrapper):
    """Gym wrapper with adherence-progress rewards in addition to potential shaping."""
    def __init__(self, env: gym.Env, workflow: List[int], gamma: float, shaping_coef: float,
                 penalty_revisit: float, penalty_future: float, penalty_offworkflow: float,
                 per_step_penalty: float, adherence_bonus: float = 50.0, adherence_penalty: float = -50.0,
                 env_reward_coef: float = 0.0):
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
        self.env_reward_coef = float(env_reward_coef)
        self.visited_sequence: List[int] = []
        self._phi_s: float = 0.0
        self._prev_adherence: float = 0.0
        # Expand observation to match custom policy
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

        # Potential-based shaping
        phi_s = self._phi_s
        phi_s2 = self._compute_phi()
        shaped = self.shaping_coef * (self.gamma * phi_s2 - phi_s)

        # Additional penalties and bonuses
        if landed_idx is not None:
            if landed_idx in pre_step_visited:
                shaped += self.penalty_revisit
            else:
                prefix_ok = 0
                for i, t in enumerate(self.visited_sequence):
                    if i < len(self.workflow) and t == self.workflow[i]:
                        prefix_ok += 1
                    else:
                        break
                req_idx = min(prefix_ok, len(self.workflow) - 1)
                required_target = self.workflow[req_idx]
                if int(landed_idx) == int(required_target):
                    shaped += 10.0  # correct target hit bonus
                elif landed_idx in self.workflow:
                    shaped += self.penalty_future
                else:
                    shaped += self.penalty_offworkflow

        shaped += self.per_step_penalty

        # Adherence-progress reward
        current_adherence = self._compute_adherence()
        adh_delta = current_adherence - self._prev_adherence
        if adh_delta > 0:
            shaped += self.adherence_bonus * adh_delta  # scale by improvement
        elif adh_delta < 0:
            shaped += self.adherence_penalty * abs(adh_delta)  # scale by drop
        self._prev_adherence = current_adherence

        # Optionally include original environment reward
        if self.env_reward_coef != 0.0:
            shaped += self.env_reward_coef * float(_env_r)

        # Update stored potential
        self._phi_s = phi_s2

        # Info for diagnostics
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

        # Visited flags
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

        # Workflow encoding
        try:
            wf_vec = _to_numpy(workflow_to_vector(self.workflow, num_targets=4))
        except Exception:
            wf_vec = _np.zeros(4, dtype=_np.float32)

        return _np.concatenate([agent_norm, targets_norm, visited_flags, one_hot, wf_vec], axis=0)

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


def main():
    parser = argparse.ArgumentParser()
    # Base search options
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--num_envs', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--shaping_coef', type=float, default=1.0)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--gae_gamma', type=float, default=0.99)
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--ppo_value_coef', type=float, default=0.5)
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.05)
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5)
    parser.add_argument('--per_step_penalty', type=float, default=-0.01)
    parser.add_argument('--adherence_target', type=float, default=1.0)
    parser.add_argument('--adherence_patience', type=int, default=2)
    parser.add_argument('--early_stop_on_adherence', action='store_true')
    parser.add_argument('--adh_eval_threshold', type=float, default=0.9)
    # Penalties
    parser.add_argument('--penalty_revisit', type=float, default=0.0)
    parser.add_argument('--penalty_future', type=float, default=-10.0)
    parser.add_argument('--penalty_offworkflow', type=float, default=-5.0)
    # Adherence-progress rewards
    parser.add_argument('--adherence_bonus', type=float, default=50.0, help='Bonus per unit adherence increase')
    parser.add_argument('--adherence_penalty', type=float, default=-50.0, help='Penalty per unit adherence decrease')
    parser.add_argument('--env_reward_coef', type=float, default=0.0, help='Coefficient for original environment reward (0=ignore, 1.0=full weight)')
    # GP / kernel
    parser.add_argument('--length_scale', type=float, default=3.0)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--kernel_type', type=str, default='rbf_possunmatch', 
                        choices=['rbf_rank','rbf_pairwise','rbf_mixed','rbf_posunmatch','rbf_possunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=1e-4)
    parser.add_argument('--kappa', type=float, default=4.0)
    # Early stop
    parser.add_argument('--enable_ucb_early_stop', action='store_true')
    parser.add_argument('--stop_min_explored', type=int, default=2)
    # Revisits
    parser.add_argument('--allow_revisits', dest='allow_revisits', action='store_true')
    parser.add_argument('--no_revisits', dest='allow_revisits', action='store_false')
    parser.set_defaults(allow_revisits=True)
    # Continuation
    parser.add_argument('--continue_policy', action='store_true')
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker_seed_base', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='gpucb_adherence_reward')
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
        }, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    visit_counts: Dict[Tuple[int,int,int,int], int] = {}
    current_sb3_policy_state = None
    current_lr = float(args.lr)

    it = 0
    global_update_count = 0
    hit_global_limit = False
    worker_seed_base = int(args.worker_seed_base) if args.worker_seed_base is not None else int(args.seed)

    while True:
        # GP posterior
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
                    break

        # Selection
        ucb_for_choice = ucb.copy()
        if not bool(args.allow_revisits):
            ucb_for_choice[observed_indices] = -np.inf
        max_val = float(np.max(ucb_for_choice))
        tie_indices = [i for i, val in enumerate(ucb_for_choice) if val == max_val]
        next_idx = int(np.random.choice(tie_indices))
        print(f"[GP-UCB] Iter {it} selecting workflow {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")

        wf = tuple(candidates[next_idx])

        # Continuation
        continued = False
        effective_lr = float(current_lr) if bool(args.continue_policy) and current_sb3_policy_state is not None else float(args.lr)
        if bool(args.continue_policy) and current_sb3_policy_state is not None:
            continued = True

        vc = visit_counts.get(wf, 0) + 1
        visit_counts[wf] = vc
        updates_run = 0
        env_eval_history = []
        updates_csv = os.path.join(run_dir, 'updates.csv')
        consecutive_target_meets = 0
        last_mean_adherence = 0.0
        score_on_adh_thresh = None
        env_eval_when_adh_thresh = []

        # Build SB3 envs with adherence-progress reward
        def make_env_fn(rank: int):
            def _init():
                base = DiagonalCornersEnv(max_steps=int(args.max_steps))
                env = AdherenceShapedWorkflowEnv(
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

        vec_env = DummyVecEnv([make_env_fn(i) for i in range(int(args.num_envs))])
        mon_dir = os.path.join(run_dir, 'monitor')
        os.makedirs(mon_dir, exist_ok=True)
        vec_env = VecMonitor(vec_env, filename=os.path.join(mon_dir, 'monitor.csv'))

        # LayerNorm MLP features extractor
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
            net_arch=[],
            activation_fn=nn.ReLU,
            ortho_init=False,
        )

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

        # Continuation
        if continued:
            try:
                sb3_model.policy.load_state_dict(current_sb3_policy_state)
            except Exception:
                pass

        # Training callback
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
                self.ep_adh_delta = []

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
                        adh_delta = float(info.get('adherence_delta', 0.0))
                        self.ep_shaped.append(self.env_shaped[i])
                        self.ep_canonical.append(canonical)
                        self.ep_adh.append(adh)
                        self.ep_adh_delta.append(adh_delta)
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
                    return 0.0, 0.0, 0.0, 0.0
                import numpy as _np
                return (
                    float(_np.mean(self.ep_shaped)),
                    float(_np.mean(self.ep_canonical)),
                    float(_np.mean(self.ep_adh)),
                    float(_np.mean(self.ep_adh_delta)),
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

        # Train per-update with timing
        import time
        steps_per_update = int(args.num_envs) * int(args.max_steps)
        timing_log = os.path.join(run_dir, 'timing.csv')
        for update in range(int(args.updates)):
            global_update_count += 1
            total_update_idx = int(global_update_count)
            cb = TrainingMetricsCallback(n_envs=vec_env.num_envs, step_penalty=float(args.per_step_penalty))
            
            t_start = time.time()
            sb3_model.learn(total_timesteps=steps_per_update, reset_num_timesteps=False, progress_bar=False, callback=cb)
            t_end = time.time()
            update_time = t_end - t_start

            shaped_mean, eval_env_return, mean_adh, mean_adh_delta = cb.get_means()
            mode_seq, mode_frac = cb.get_mode_seq()
            env_eval_history.append(eval_env_return)
            last_mean_adherence = float(mean_adh)
            
            # Compute speeds
            steps_collected = steps_per_update
            sampling_speed = steps_collected / update_time if update_time > 0 else 0.0
            
            print(f"  [SB3 Adh-Reward wf {list(wf)}] Update {update:3d} | Shaped {shaped_mean:7.2f} | Canonical {eval_env_return:7.2f} | Adh {mean_adh:5.1%} | ΔAdh {mean_adh_delta:+.3f} | Visit {mode_seq} ({mode_frac:5.1%}) | Time {update_time:.2f}s ({sampling_speed:.0f} steps/s)", flush=True)
            updates_run = update + 1
            
            # Log timing
            try:
                timing_header = ['workflow', 'visit', 'update', 'total_update', 'update_time_s', 'steps_collected', 'steps_per_sec']
                timing_row = {
                    'workflow': '-'.join(map(str, wf)),
                    'visit': int(vc),
                    'update': int(update),
                    'total_update': int(total_update_idx),
                    'update_time_s': float(update_time),
                    'steps_collected': int(steps_collected),
                    'steps_per_sec': float(sampling_speed),
                }
                write_timing_header = not os.path.exists(timing_log)
                with open(timing_log, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=timing_header)
                    if write_timing_header:
                        writer.writeheader()
                    writer.writerow(timing_row)
            except Exception:
                pass

            # Threshold-triggered early switch
            if mean_adh >= float(args.adh_eval_threshold):
                env_eval_when_adh_thresh.append(eval_env_return)
                if score_on_adh_thresh is None:
                    score_on_adh_thresh = float(eval_env_return)
                if len(env_eval_when_adh_thresh) >= 3 and env_eval_when_adh_thresh[-1] <= env_eval_when_adh_thresh[-2] <= env_eval_when_adh_thresh[-3]:
                    print(f"  [SB3 Adh-Reward wf {list(wf)}] Early stop on stability (adh>={args.adh_eval_threshold:.2f})")
                    break

            # Adherence-based early stop
            if bool(args.early_stop_on_adherence):
                if mean_adh >= float(args.adherence_target):
                    consecutive_target_meets += 1
                else:
                    consecutive_target_meets = 0
                if consecutive_target_meets >= int(args.adherence_patience):
                    print(f"  [SB3 Adh-Reward wf {list(wf)}] Early stop on adherence: {mean_adh:.3f} for {consecutive_target_meets} updates")
                    break

            if global_update_count >= int(MAX_TOTAL_UPDATES):
                print(f"[Search] Stopping: reached MAX_TOTAL_UPDATES={int(MAX_TOTAL_UPDATES)}", flush=True)
                hit_global_limit = True
                break

            # CSV logging
            try:
                header = [
                    'workflow', 'visit', 'update', 'total_update',
                    'mean_return_shaped', 'mean_env_return', 'mean_adherence', 'mean_adh_delta', 'mode_seq', 'mode_frac'
                ]
                row = {
                    'workflow': '-'.join(map(str, wf)),
                    'visit': int(vc),
                    'update': int(update),
                    'total_update': int(total_update_idx),
                    'mean_return_shaped': float(shaped_mean),
                    'mean_env_return': float(eval_env_return),
                    'mean_adherence': float(mean_adh),
                    'mean_adh_delta': float(mean_adh_delta),
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

        # Score for GP
        if score_on_adh_thresh is not None:
            score = float(score_on_adh_thresh)
            score_source = f"adh{int(100*args.adh_eval_threshold)}_canonical"
        elif len(env_eval_history) > 0:
            score = float(np.max(env_eval_history))
            score_source = 'max_canonical_during_training'
        else:
            score = 0.0
            score_source = 'fallback'

        # Update GP observations
        if next_idx not in observed_indices:
            observed_indices.append(next_idx)
            observed_embeddings.append(candidate_embeddings[next_idx])
            observed_scores.append(score)
        else:
            pos = observed_indices.index(next_idx)
            observed_scores[pos] = max(observed_scores[pos], score)

        # Update continuation state
        try:
            current_sb3_policy_state = {k: v.detach().cpu() for k, v in sb3_model.policy.state_dict().items()}
        except Exception:
            current_sb3_policy_state = sb3_model.policy.state_dict()
        current_lr = max(float(args.min_lr), float(effective_lr) * float(args.lr_decay))

        # Log GP
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
        print("\n=== Adherence-Reward Workflow Search Completed ===")
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

