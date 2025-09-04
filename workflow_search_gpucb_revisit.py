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
    # Global stopping based on total updates across all workflows
    parser.add_argument('--max_total_updates', type=int, default=10000, help='Stop the entire search once this many PPO updates have been run in total')
    # Stability: continuation and LR decay
    parser.add_argument('--continue_policy', action='store_true', help='Continue training from existing policy across workflows (global) or per-workflow')
    parser.add_argument('--continuation_scope', type=str, default='global', choices=['global','per_workflow'], help='Use global policy continuation (default) or per-workflow cache')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Multiplicative LR decay applied after each workflow selection')
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
    # Global continuation state
    current_policy_state = None
    current_lr = float(args.lr)

    it = 0
    global_update_count = 0
    hit_global_limit = False
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
        state_dict = None
        effective_lr = start_lr
        if bool(args.continue_policy):
            if args.continuation_scope == 'global' and current_policy_state is not None:
                state_dict = current_policy_state
                effective_lr = float(current_lr)
                continued = True
            elif args.continuation_scope == 'per_workflow' and wf in policy_cache:
                state_dict, prev_lr = policy_cache[wf]
                effective_lr = max(1e-6, prev_lr * float(args.lr_decay))
                continued = True

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
        adherence_history = []
        success_history = []
        env_eval_when_adh1 = []
        consecutive_target_meets = 0
        last_mean_adherence = 0.0
        # Track visit count once per selection
        vc = visit_counts.get(wf, 0) + 1
        visit_counts[wf] = vc
        score_on_adh1 = None
        for update in range(int(args.updates)):
            # Increment global update counter and compute total_update index for logging
            global_update_count += 1
            total_update_idx = int(global_update_count)
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
            # Real-time console print (mirrors original script)
            print(f"  [Train wf {list(wf)}] Update {update:3d} | Return {mean_return:7.2f} | Adh {mean_adherence:5.1%} | Succ {success_rate:5.1%} | Len {avg_ep_len:5.1f} | Visit {mode_seq} ({mode_frac:5.1%})", flush=True)
            if bool(args.eval_use_canonical):
                print(f"    EvalEnv canonical (mean): {eval_env_return:7.2f}", flush=True)
            else:
                print(f"    RolloutEnv (mean env-only over batch): {eval_env_return:7.2f}", flush=True)
            # When adherence hits 100%, record score and switch workflow immediately (GP update next)
            if mean_adherence >= 1.0 and score_on_adh1 is None:
                env_eval_when_adh1.append(eval_env_return)
                score_on_adh1 = float(eval_env_return)
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
            # Canonical-eval stability early stop when adh=100%
            if len(env_eval_when_adh1) >= 3:
                if env_eval_when_adh1[-1] <= env_eval_when_adh1[-2] <= env_eval_when_adh1[-3]:
                    print(f"  [Train wf {list(wf)}] Early stop on canonical-eval stability (adh=100%): last3={env_eval_when_adh1[-3:]} (non-increasing)")
                    updates_run = update + 1
                    # fall through to logging row and then break
                    # (we will still write the row below)
                    pass

            # Logging with visit count and total_update index for convergence plots
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
            # If stability early stop triggered above (env_eval_when_adh1 last3 non-increasing), break here
            if len(env_eval_when_adh1) >= 3 and env_eval_when_adh1[-1] <= env_eval_when_adh1[-2] <= env_eval_when_adh1[-3]:
                break
            # If we just hit 100% adherence, switch workflows after logging this row
            if score_on_adh1 is not None:
                updates_run = update + 1
                break
            # Stop globally if max total updates reached
            if global_update_count >= int(args.max_total_updates):
                print(f"[Search] Stopping: reached max_total_updates={int(args.max_total_updates)}", flush=True)
                hit_global_limit = True
                break

        # Score for GP update
        if score_on_adh1 is not None:
            score = float(score_on_adh1)
            score_source = 'adh1_canonical_mean'
        elif bool(args.eval_use_canonical) and len(env_eval_history) > 0:
            score = float(np.max(env_eval_history))
            score_source = 'max_canonical_mean_during_training'
        else:
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

        # Update continuation state: global and optional per-workflow
        current_policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        current_lr = max(1e-6, float(effective_lr) * float(args.lr_decay))
        policy_cache[wf] = (current_policy_state, float(current_lr))

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
                    'adherence_last': float(last_mean_adherence),
                    'mu': float(mu[next_idx]),
                    'std': float(std[next_idx]),
                    'ucb': float(ucb[next_idx]),
                }) + "\n")
        except Exception:
            pass

        it += 1
        current_epsilon = max(float(args.min_epsilon), float(current_epsilon) * float(args.epsilon_decay))
        if hit_global_limit:
            break
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


