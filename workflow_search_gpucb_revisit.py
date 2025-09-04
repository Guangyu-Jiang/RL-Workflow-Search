"""
Workflow Search with GP-UCB (Revisit-enabled) on DiagonalCornersEnv.

- Allows selecting workflows that have already been explored to observe convergence.
- Adds stability options (policy continuation, learning-rate decay) and richer logging.

This script is adapted from workflow_search_gpucb.py.
"""

from workflow_search_gpucb import *  # reuse all helpers, env wrappers, PPO, GP utils


def main():
    parser = argparse.ArgumentParser()
    # Base search options
    parser.add_argument('--iterations', type=int, default=0, help='Optional max selections (0 = no max; stop by UCB criterion if enabled)')
    parser.add_argument('--updates', type=int, default=5000)
    parser.add_argument('--num_envs', type=int, default=25)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-5)
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
    parser.add_argument('--adherence_patience', type=int, default=2)
    parser.add_argument('--early_stop_on_adherence', action='store_true')
    # Penalties
    parser.add_argument('--penalty_revisit', type=float, default=-2.0)
    parser.add_argument('--penalty_future', type=float, default=-100.0)
    parser.add_argument('--penalty_offworkflow', type=float, default=-50.0)
    # GP / kernel
    parser.add_argument('--length_scale', type=float, default=0.75)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--kernel_type', type=str, default='rbf_rank', choices=['rbf_rank','rbf_pairwise','rbf_mixed','rbf_posunmatch','rbf_possunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=10.0)
    parser.add_argument('--kappa', type=float, default=2.0)
    # Early stop
    parser.add_argument('--enable_ucb_early_stop', action='store_true', help='Enable UCB-based early stopping')
    parser.add_argument('--stop_min_explored', type=int, default=2)
    # Epsilon-greedy
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=1.0)
    parser.add_argument('--min_epsilon', type=float, default=0.0)
    parser.add_argument('--epsilon_scope', type=str, default='all', choices=['all','unexplored'], help='Whether epsilon samples from all workflows or only unexplored')
    # Revisits toggle
    parser.add_argument('--allow_revisits', dest='allow_revisits', action='store_true', help='Allow selecting previously observed workflows (default)')
    parser.add_argument('--no_revisits', dest='allow_revisits', action='store_false', help='Disallow revisiting already observed workflows')
    parser.set_defaults(allow_revisits=True)
    # Stability: continuation and LR decay
    parser.add_argument('--continue_policy', action='store_true', help='Continue training from last policy for a revisited workflow')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Multiplicative LR decay applied after each revisit of the same workflow')
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='gpucb_diagonal_revisit')
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

    # For revisits: track per-workflow policy state and LR
    policy_cache: Dict[Tuple[int,int,int,int], Tuple[dict, float]] = {}
    visit_counts: Dict[Tuple[int,int,int,int], int] = {}

    it = 0
    current_epsilon = float(args.epsilon)
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

        # Early stop (optional, uses same rule as main script but without masking since revisits allowed)
        if args.enable_ucb_early_stop and len(observed_indices) >= int(args.stop_min_explored):
            # Consider unexplored UCB for stopping; if all explored, allow selecting best to continue
            unexplored = [i for i in range(len(candidates)) if i not in observed_indices]
            if len(unexplored) == 0:
                # All explored at least once; continue until iterations cap
                pass
            else:
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

        # Selection: epsilon over scope, then GP-UCB. If revisits disallowed, mask explored for GP phase.
        scope_indices = list(range(len(candidates))) if args.epsilon_scope == 'all' else [i for i in range(len(candidates)) if i not in observed_indices]
        if len(scope_indices) == 0:
            scope_indices = list(range(len(candidates)))

        if np.random.rand() < current_epsilon:
            next_idx = int(np.random.choice(scope_indices))
            phase = 'epsilon'
            print(f"[Epsilon] Iter {it} selecting workflow {candidates[next_idx]} from scope={args.epsilon_scope} (eps={current_epsilon:.3f})")
        else:
            # Choose the highest UCB; optionally mask explored if revisits not allowed
            ucb_for_choice = ucb.copy()
            if not bool(args.allow_revisits):
                ucb_for_choice[observed_indices] = -np.inf
            max_val = float(np.max(ucb_for_choice))
            tie_indices = [i for i, val in enumerate(ucb_for_choice) if val == max_val]
            next_idx = int(np.random.choice(tie_indices))
            phase = 'gp'
            print(f"[GP-UCB] Iter {it} selecting workflow {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")

        wf = tuple(candidates[next_idx])

        # Policy initialization or continuation
        start_lr = float(args.lr)
        continued = False
        if bool(args.continue_policy) and wf in policy_cache:
            # Restore policy and decay LR for stability
            state_dict, prev_lr = policy_cache[wf]
            effective_lr = max(1e-6, prev_lr * float(args.lr_decay))
            continued = True
        else:
            state_dict = None
            effective_lr = start_lr

        # Train
        # We reuse train_for_workflow but optionally inject initial policy and LR
        state_dim = 2 + 4 * 2 + 4
        wf_dim = 4
        policy = WorkflowPolicy(state_dim, wf_dim).to(device)
        if state_dict is not None:
            try:
                policy.load_state_dict(state_dict)
            except Exception:
                pass
        optimizer = optim.Adam(policy.parameters(), lr=effective_lr)

        # Temporarily wrap a one-iteration training by calling rollout + ppo_update in a loop that mirrors train_for_workflow
        # to keep behavior consistent but allow continuation and custom LR.
        # We mirror essential parts of train_for_workflow for simplicity here.
        updates_run = 0
        env_eval_history = []
        updates_csv = os.path.join(run_dir, 'updates.csv')
        # Track visit count once per selection
        vc = visit_counts.get(wf, 0) + 1
        visit_counts[wf] = vc
        for update in range(int(args.updates)):
            if bool(args.use_mp):
                batch_trajs = rollout_shaped_multiprocessing(
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

            # Logging with visit count and total_update index for convergence plots
            try:
                header = [
                    'workflow', 'visit', 'update', 'total_update',
                    'mean_return_shaped', 'mean_env_return', 'policy_loss', 'value_loss', 'entropy'
                ]
                row = {
                    'workflow': '-'.join(map(str, wf)),
                    'visit': int(vc),
                    'update': int(update),
                    'total_update': int(it * int(args.updates) + update),
                    'mean_return_shaped': float(mean_return),
                    'mean_env_return': float(eval_env_return),
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

        # Score: use max canonical mean during this selection
        if bool(args.eval_use_canonical) and len(env_eval_history) > 0:
            score = float(np.max(env_eval_history))
            score_source = 'max_canonical_mean_during_training'
        else:
            # Fallback: quick eval over proposed
            env = DiagonalCornersEnv(max_steps=int(args.max_steps))
            eval_returns = [rollout_env_only(env, policy, list(wf), device, deterministic=True) for _ in range(int(args.final_eval_episodes))]
            score = float(np.mean(eval_returns))
            score_source = 'final_eval_mean_proposed'

        # Update GP observations only on the first visit of a workflow (to keep GP simple)
        if next_idx not in observed_indices:
            observed_indices.append(next_idx)
            observed_embeddings.append(candidate_embeddings[next_idx])
            observed_scores.append(score)
        else:
            # Optionally could average or keep max; we keep the best score to reflect improvement
            pos = observed_indices.index(next_idx)
            observed_scores[pos] = max(observed_scores[pos], score)

        # Update policy cache with latest state dict and LR for potential continuation
        policy_cache[wf] = (policy.state_dict(), float(effective_lr))

        # Log GP state and selection
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'iteration': int(it),
                    'workflow': list(wf),
                    'phase': phase,
                    'continued': bool(continued),
                    'effective_lr': float(effective_lr),
                    'score_env_only': float(score),
                    'score_source': score_source,
                    'mu': float(mu[next_idx]),
                    'std': float(std[next_idx]),
                    'ucb': float(ucb[next_idx]),
                }) + "\n")
        except Exception:
            pass

        it += 1
        current_epsilon = max(float(args.min_epsilon), float(current_epsilon) * float(args.epsilon_decay))
        if int(args.iterations) > 0 and it >= int(args.iterations):
            break

    # Summary
    if len(observed_scores) > 0:
        best_idx = int(np.argmax(observed_scores))
        print("\n=== Revisit-enabled Workflow Search Completed ===")
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


