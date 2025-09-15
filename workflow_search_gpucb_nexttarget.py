"""
Workflow Search with GP-UCB on DiagonalCornersEnv (Next-Target Conditioning).

- Change vs original: The policy conditions ONLY on the next target one-hot (size 4),
  while the full workflow order is still used to compute what the next target is at each step.
- Benefit: Effectively yields 4 subpolicies that are naturally reused across workflows.

We reuse PPO, GP, kernel, and utilities from workflow_search_gpucb.py, and override
the rollout/training to feed next-target one-hot inputs.
"""

import argparse
import csv
import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse helpers, GP, PPO, env wrappers from the base implementation
from workflow_search_gpucb import *  # noqa: F401,F403


# ----------------------------- Next-target utilities ----------------------------- #

def next_target_one_hot(visited_sequence: List[int], workflow: List[int], num_targets: int = 4) -> np.ndarray:
    """Compute one-hot for the next required target given first-visit sequence and workflow order.

    If all workflow targets have been achieved in order, we keep using the last workflow target
    as a stable input (episode will end promptly anyway).
    """
    progress = correct_prefix_length(visited_sequence, workflow)
    if progress < len(workflow):
        tgt_idx = int(workflow[progress])
    else:
        tgt_idx = int(workflow[-1])
    vec = np.zeros(num_targets, dtype=np.float32)
    vec[tgt_idx] = 1.0
    return vec


# ----------------------------- Rollouts (next-target input) ----------------------------- #

def rollout_shaped_vectorized_next(
    policy: WorkflowPolicy,
    wf_order: List[int],
    num_envs: int,
    max_steps: int,
    device: torch.device,
    gamma: float,
    shaping_coef: float,
    penalty_revisit: float,
    penalty_future: float,
    penalty_offworkflow: float,
    per_step_penalty: float = -0.01,
    debug_rewards: bool = False,
    debug_rewards_env: int = 0,
    debug_rewards_steps: int = 200,
    deterministic_rollouts: bool = False,
) -> List[Dict]:
    envs = [DiagonalCornersEnv(max_steps=max_steps) for _ in range(num_envs)]
    for env in envs:
        env.reset(workflow=wf_order)
    states = [env.get_state_for_policy() for env in envs]
    done = [False] * num_envs
    trajectories: List[Dict] = []
    visited_sequences: List[List[int]] = [[] for _ in range(num_envs)]
    env_ep_returns = [0.0 for _ in range(num_envs)]

    for _ in range(num_envs):
        trajectories.append({k: [] for k in ["states","workflows","actions","logps","rewards","values","dones"]})

    debug_steps_left = int(debug_rewards_steps)

    while not all(done):
        batch_indices = [i for i, d in enumerate(done) if not d]
        batch_states = torch.tensor(np.array([states[i] for i in batch_indices]), dtype=torch.float32, device=device)
        # Build dynamic next-target one-hots per env
        batch_wf_np = np.stack([
            next_target_one_hot(visited_sequences[i], wf_order) for i in batch_indices
        ], axis=0)
        batch_wf = torch.tensor(batch_wf_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            if deterministic_rollouts:
                logits, v = policy.forward(batch_states, batch_wf)
                probs = F.softmax(logits, dim=-1)
                a = torch.argmax(probs, dim=-1)
                logp_all = torch.log(probs + 1e-8)
                logp = logp_all.gather(1, a.view(-1, 1)).squeeze(1)
            else:
                a, logp, v = policy.act(batch_states, batch_wf)

        for bpos, i in enumerate(batch_indices):
            a_int = int(a[bpos].item())

            # Potential-based shaping: use correct-prefix progress from first visits so far
            progress_pre = correct_prefix_length(visited_sequences[i], wf_order)
            next_tgt_idx_pre = wf_order[progress_pre] if progress_pre < len(wf_order) else wf_order[-1]
            cur_target = envs[i].target_positions[next_tgt_idx_pre]
            phi_s = -manhattan(tuple(envs[i].agent_pos), cur_target)
            if debug_rewards and (i == int(debug_rewards_env)) and (debug_steps_left > 0):
                print(f"[DBG pre] env={i} progress_pre={progress_pre} next_target_idx={next_tgt_idx_pre} pos={tuple(envs[i].agent_pos)} phi={phi_s}")
            pre_step_visited = set(visited_sequences[i])

            _, r_env, d, info = envs[i].step(a_int)
            env_ep_returns[i] += float(r_env)

            # Track first visits immediately after step
            pos = tuple(envs[i].agent_pos)
            for t_idx, t_pos in enumerate(envs[i].target_positions):
                if pos == t_pos and t_idx not in visited_sequences[i]:
                    visited_sequences[i].append(t_idx)

            next_state = envs[i].get_state_for_policy()

            progress_post = correct_prefix_length(visited_sequences[i], wf_order)
            next_tgt_idx_post = wf_order[progress_post] if progress_post < len(wf_order) else wf_order[-1]
            cur_target2 = envs[i].target_positions[next_tgt_idx_post]
            phi_s2 = -manhattan(tuple(envs[i].agent_pos), cur_target2)
            # Potential-only customized reward (exclude environment reward)
            shaped = shaping_coef * (gamma * phi_s2 - phi_s)
            # Additional penalties
            if (penalty_revisit != 0.0) or (penalty_future != 0.0) or (penalty_offworkflow != 0.0):
                landed_idx = None
                pos_after = tuple(envs[i].agent_pos)
                for t_idx, t_pos in enumerate(envs[i].target_positions):
                    if pos_after == t_pos:
                        landed_idx = t_idx
                        break
                if landed_idx is not None:
                    if landed_idx in pre_step_visited:
                        shaped += float(penalty_revisit)
                    else:
                        if landed_idx != next_tgt_idx_pre:
                            if landed_idx in wf_order:
                                shaped += float(penalty_future)
                            else:
                                shaped += float(penalty_offworkflow)

            # Per-step penalty
            shaped += float(per_step_penalty)

            if debug_rewards and (i == int(debug_rewards_env)) and (debug_steps_left > 0):
                dbg_landed = None
                pos_after = tuple(envs[i].agent_pos)
                for t_idx, t_pos in enumerate(envs[i].target_positions):
                    if pos_after == t_pos:
                        dbg_landed = t_idx
                        break
                print(f"[DBG post] env={i} action={a_int} pos={pos_after} landed={dbg_landed} progress_post={progress_post} next_target_post={next_tgt_idx_post} shaped={shaped:.2f} visited={visited_sequences[i]}")
                debug_steps_left -= 1

            # Record transition (use the one-hot used for this step)
            trajectories[i]["states"].append(states[i])
            trajectories[i]["workflows"].append(batch_wf_np[bpos])
            trajectories[i]["actions"].append(a_int)
            trajectories[i]["logps"].append(float(logp[bpos].item()))
            trajectories[i]["rewards"].append(float(shaped))
            trajectories[i]["values"].append(float(v[bpos].item()))
            trajectories[i]["dones"].append(bool(d))

            states[i] = next_state
            done[i] = d

    # Finalize
    for i in range(num_envs):
        trajectories[i]["ep_return"] = float(sum(trajectories[i]["rewards"]))
        trajectories[i]["visited_sequence"] = calculate_first_visits(visited_sequences[i])
        trajectories[i]["adherence"] = adherence_rate(trajectories[i]["visited_sequence"], wf_order)
        trajectories[i]["success"] = (trajectories[i]["visited_sequence"] == wf_order)
        trajectories[i]["env_ep_return"] = float(env_ep_returns[i])
    return trajectories


def _worker_shaped_episode_next(
    worker_id: int,
    policy_state_dict,
    wf_order: List[int],
    max_steps: int,
    gamma: float,
    shaping_coef: float,
    penalty_revisit: float,
    penalty_future: float,
    penalty_offworkflow: float,
    per_step_penalty: float,
    debug_rewards: bool,
    debug_rewards_worker: int,
    debug_rewards_steps: int,
    deterministic_rollouts: bool,
    return_queue: mp.Queue,
):
    """Multiprocessing worker to run one shaped episode with next-target conditioning."""
    try:
        torch.set_num_threads(1)
        state_dim = 2 + 4 * 2 + 4
        wf_dim = 4
        policy = WorkflowPolicy(state_dim, wf_dim)
        policy.load_state_dict(policy_state_dict)
        policy.eval()

        env = DiagonalCornersEnv(max_steps=max_steps)
        env.reset(workflow=wf_order)
        state = env.get_state_for_policy()

        traj = {k: [] for k in ["states","workflows","actions","logps","rewards","values","dones"]}
        visited_sequence: List[int] = []
        ep_return = 0.0
        env_ep_return = 0.0
        debug_steps_left = int(debug_rewards_steps)
        while True:
            wf_vec_np = next_target_one_hot(visited_sequence, wf_order)
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            w = torch.tensor(wf_vec_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if deterministic_rollouts:
                    logits, v_t = policy.forward(s, w)
                    probs = F.softmax(logits, dim=-1)
                    a_t = torch.argmax(probs, dim=-1)
                    logp_all = torch.log(probs + 1e-8)
                    logp_t = logp_all.gather(1, a_t.view(-1, 1)).squeeze(1)
                    a_int = int(a_t.item())
                    logp_val = float(logp_t.item())
                    v_val = float(v_t.item())
                else:
                    a_t, logp_t, v_t = policy.act(s, w)
                    a_int = int(a_t.item())
                    logp_val = float(logp_t.item())
                    v_val = float(v_t.item())

            # Shaping potential before step
            progress_pre = correct_prefix_length(visited_sequence, wf_order)
            next_tgt_idx_pre = wf_order[progress_pre] if progress_pre < len(wf_order) else wf_order[-1]
            cur_target = env.target_positions[next_tgt_idx_pre]
            phi_s = -manhattan(tuple(env.agent_pos), cur_target)
            if debug_rewards and (worker_id == int(debug_rewards_worker)) and (debug_steps_left > 0):
                print(f"[DBG pre] worker={worker_id} progress_pre={progress_pre} next_target_idx={next_tgt_idx_pre} pos={tuple(env.agent_pos)} phi={phi_s}")
            pre_step_visited = set(visited_sequence)

            _, r_env, done, _ = env.step(a_int)
            env_ep_return += float(r_env)

            # Track visits after step
            pos = tuple(env.agent_pos)
            for t_idx, t_pos in enumerate(env.target_positions):
                if pos == t_pos and t_idx not in visited_sequence:
                    visited_sequence.append(t_idx)

            state_next = env.get_state_for_policy()

            progress_post = correct_prefix_length(visited_sequence, wf_order)
            next_tgt_idx_post = wf_order[progress_post] if progress_post < len(wf_order) else wf_order[-1]
            cur_target2 = env.target_positions[next_tgt_idx_post]
            phi_s2 = -manhattan(tuple(env.agent_pos), cur_target2)
            shaped = shaping_coef * (gamma * phi_s2 - phi_s)

            landed_idx = None
            pos_after = tuple(env.agent_pos)
            for t_idx, t_pos in enumerate(env.target_positions):
                if pos_after == t_pos:
                    landed_idx = t_idx
                    break
            if landed_idx is not None:
                if landed_idx in pre_step_visited:
                    shaped += float(penalty_revisit)
                else:
                    if landed_idx != next_tgt_idx_pre:
                        if landed_idx in wf_order:
                            shaped += float(penalty_future)
                        else:
                            shaped += float(penalty_offworkflow)

            shaped += float(per_step_penalty)

            if debug_rewards and (worker_id == int(debug_rewards_worker)) and (debug_steps_left > 0):
                dbg_landed = None
                pos_after = tuple(env.agent_pos)
                for t_idx, t_pos in enumerate(env.target_positions):
                    if pos_after == t_pos:
                        dbg_landed = t_idx
                        break
                print(f"[DBG post] worker={worker_id} action={a_int} pos={pos_after} landed={dbg_landed} progress_post={progress_post} next_target_post={next_tgt_idx_post} shaped={shaped:.2f} visited={visited_sequence}")
                debug_steps_left -= 1

            traj["states"].append(state)
            traj["workflows"].append(wf_vec_np)
            traj["actions"].append(a_int)
            traj["logps"].append(logp_val)
            traj["rewards"].append(float(shaped))
            traj["values"].append(v_val)
            traj["dones"].append(bool(done))
            ep_return += float(shaped)
            state = state_next
            if done:
                break

        first_visits = calculate_first_visits(visited_sequence)
        adh = adherence_rate(first_visits, wf_order)
        succ = (first_visits == wf_order)
        traj["ep_return"] = float(ep_return)
        traj["env_ep_return"] = float(env_ep_return)
        traj["visited_sequence"] = first_visits
        traj["adherence"] = float(adh)
        traj["success"] = bool(succ)
        return_queue.put((worker_id, traj))
    except Exception as e:
        return_queue.put((worker_id, {"error": str(e)}))


def rollout_shaped_multiprocessing_next(
    policy: WorkflowPolicy,
    wf_order: List[int],
    num_envs: int,
    max_steps: int,
    device: torch.device,
    gamma: float,
    shaping_coef: float,
    penalty_revisit: float,
    penalty_future: float,
    penalty_offworkflow: float,
    per_step_penalty: float,
    debug_rewards: bool,
    debug_rewards_env: int,
    debug_rewards_steps: int,
    deterministic_rollouts: bool,
) -> List[Dict]:
    """Collect one shaped episode per worker using multiprocessing (next-target input)."""
    state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    return_queue: mp.Queue = mp.Queue()
    workers: List[mp.Process] = []
    for wid in range(num_envs):
        p = mp.Process(
            target=_worker_shaped_episode_next,
            args=(
                wid,
                state_dict_cpu,
                wf_order,
                max_steps,
                gamma,
                shaping_coef,
                penalty_revisit,
                penalty_future,
                penalty_offworkflow,
                per_step_penalty,
                bool(debug_rewards),
                int(debug_rewards_env),
                int(debug_rewards_steps),
                bool(deterministic_rollouts),
                return_queue,
            ),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    trajectories: List[Dict] = [None] * num_envs
    collected = 0
    while collected < num_envs:
        wid, data = return_queue.get()
        trajectories[wid] = data
        collected += 1
    for p in workers:
        p.join()

    for i, tr in enumerate(trajectories):
        if isinstance(tr, dict) and 'error' in tr:
            raise RuntimeError(f"Worker {i} failed: {tr['error']}")
    return trajectories


# ----------------------------- PPO training (next-target) ----------------------------- #

def train_for_workflow_next(
    wf_order: List[int],
    updates: int,
    num_envs: int,
    max_steps: int,
    lr: float,
    gamma: float,
    shaping_coef: float,
    ppo_epochs: int,
    device: torch.device,
    use_mp: bool,
    penalty_revisit: float,
    penalty_future: float,
    penalty_offworkflow: float,
    per_step_penalty: float,
    eval_episodes_per_update: int,
    eval_parallel: bool,
    eval_parallel_num_envs: int,
    eval_use_canonical: bool,
    gae_gamma: float,
    minibatch_size: int,
    debug_rewards: bool,
    debug_rewards_env: int,
    debug_rewards_steps: int,
    ppo_clip: float,
    ppo_vcoef: float,
    ppo_entcoef: float,
    ppo_max_grad_norm: float,
    deterministic_rollouts: bool,
    adherence_target: float,
    adherence_patience: int,
    early_stop_on_adherence: bool,
    run_dir: str,
) -> Tuple[WorkflowPolicy, Dict]:
    state_dim = 2 + 4 * 2 + 4
    wf_dim = 4  # next-target one-hot
    policy = WorkflowPolicy(state_dim, wf_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    adherence_history: List[float] = []
    success_history: List[float] = []
    env_eval_history: List[float] = []
    env_eval_when_adh1: List[float] = []

    consecutive_target_meets = 0
    updates_run = 0
    updates_csv = os.path.join(run_dir, 'updates.csv')
    for update in range(updates):
        if use_mp:
            batch_trajs = rollout_shaped_multiprocessing_next(
                policy,
                wf_order,
                num_envs,
                max_steps,
                device,
                gamma,
                shaping_coef,
                penalty_revisit=penalty_revisit,
                penalty_future=penalty_future,
                penalty_offworkflow=penalty_offworkflow,
                per_step_penalty=per_step_penalty,
                debug_rewards=debug_rewards,
                debug_rewards_env=debug_rewards_env,
                debug_rewards_steps=debug_rewards_steps,
                deterministic_rollouts=bool(deterministic_rollouts),
            )
        else:
            batch_trajs = rollout_shaped_vectorized_next(
                policy,
                wf_order,
                num_envs,
                max_steps,
                device,
                gamma,
                shaping_coef,
                penalty_revisit=penalty_revisit,
                penalty_future=penalty_future,
                penalty_offworkflow=penalty_offworkflow,
                per_step_penalty=per_step_penalty,
                debug_rewards=debug_rewards,
                debug_rewards_env=debug_rewards_env,
                debug_rewards_steps=debug_rewards_steps,
                deterministic_rollouts=bool(deterministic_rollouts),
            )

        batch = {k: [] for k in ['states', 'workflows', 'actions', 'logps', 'rewards', 'values', 'dones']}
        for tr in batch_trajs:
            for k in batch.keys():
                batch[k].extend(tr[k])
        advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['dones'], gamma=gae_gamma)
        batch['advantages'] = advantages
        batch['returns'] = returns

        stats = ppo_update(
            policy,
            optimizer,
            batch,
            clip=ppo_clip,
            value_coef=ppo_vcoef,
            entropy_coef=ppo_entcoef,
            epochs=ppo_epochs,
            bs=minibatch_size,
            max_grad_norm=ppo_max_grad_norm,
            device=device,
        )
        mean_return = float(np.mean([tr['ep_return'] for tr in batch_trajs]))
        mean_env_ep_return = float(np.mean([tr.get('env_ep_return', 0.0) for tr in batch_trajs]))
        mean_adherence = float(np.mean([tr['adherence'] for tr in batch_trajs]))
        success_rate = float(np.mean([1.0 if tr['success'] else 0.0 for tr in batch_trajs]))
        # Average episode length and typical visit order
        ep_lengths = [len(tr['dones']) for tr in batch_trajs]
        avg_ep_len = float(np.mean(ep_lengths)) if len(ep_lengths) > 0 else 0.0
        try:
            from collections import Counter
            seqs = [tuple(tr['visited_sequence']) for tr in batch_trajs]
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

        # Canonical original reward computed from current batch rollouts (positional matches vs canonical)
        if bool(eval_use_canonical):
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
            eval_env_return = float(mean_env_ep_return)
        env_eval_history.append(eval_env_return)

        # If adherence is perfect, track env-eval for stability early stopping
        if mean_adherence >= 1.0:
            env_eval_when_adh1.append(eval_env_return)

        print(f"  [Train wf {wf_order}] Update {update:3d} | Return {mean_return:7.2f} | Adh {mean_adherence:5.1%} | Succ {success_rate:5.1%} | Len {avg_ep_len:5.1f} | Visit {mode_seq} ({mode_frac:5.1%})")
        if bool(eval_use_canonical):
            print(f"    EvalEnv canonical (mean): {eval_env_return:7.2f}")
        else:
            print(f"    RolloutEnv (mean env-only over batch): {eval_env_return:7.2f}")

        # Append detailed per-update metrics to CSV
        try:
            header = [
                'workflow', 'update', 'mean_return_shaped', 'mean_env_return', 'mean_adherence',
                'success_rate', 'avg_ep_len', 'mode_seq', 'mode_frac',
                'policy_loss', 'value_loss', 'entropy'
            ]
            row = {
                'workflow': '-'.join(map(str, wf_order)),
                'update': int(update),
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

        # Adherence-target early stopping
        if early_stop_on_adherence:
            if mean_adherence >= float(adherence_target):
                consecutive_target_meets += 1
            else:
                consecutive_target_meets = 0
            if consecutive_target_meets >= int(adherence_patience):
                print(f"  [Train wf {wf_order}] Early stop on adherence: mean_adherence={mean_adherence:.3f} for {consecutive_target_meets} consecutive updates")
                break

        # Env-reward stability early stop (adh=100%)
        if len(env_eval_when_adh1) >= 3:
            if env_eval_when_adh1[-1] <= env_eval_when_adh1[-2] <= env_eval_when_adh1[-3]:
                print(f"  [Train wf {wf_order}] Early stop on canonical-eval stability (adh=100%): last3={env_eval_when_adh1[-3:]} (non-increasing)")
                break

    return policy, {
        'last_mean_adherence': float(adherence_history[-1] if adherence_history else 0.0),
        'last_success_rate': float(success_history[-1] if success_history else 0.0),
        'avg_episode_length_last': float(avg_ep_len if len(adherence_history) > 0 else 0.0),
        'mode_visit_seq_last': mode_seq if len(adherence_history) > 0 else [],
        'mode_fraction_last': float(mode_frac if len(adherence_history) > 0 else 0.0),
        'updates_run': int(updates_run),
        'env_eval_last': float(env_eval_history[-1] if env_eval_history else 0.0),
        'env_eval_history': list(env_eval_history),
        'env_eval_when_adh1': list(env_eval_when_adh1),
    }


# ----------------------------- Env-only evaluation (next-target) ----------------------------- #

def rollout_env_only_next(
    env: DiagonalCornersEnv,
    policy: WorkflowPolicy,
    wf_order: List[int],
    device: torch.device,
    deterministic: bool = True,
    use_canonical_order: bool = False,
) -> float:
    if bool(use_canonical_order):
        env.reset()
    else:
        env.reset(workflow=wf_order)
    state = env.get_state_for_policy()
    ep_return = 0.0
    visited_sequence: List[int] = []
    while True:
        wf_vec_np = next_target_one_hot(visited_sequence, wf_order)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        wf = torch.tensor(wf_vec_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy.forward(s, wf)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = int(torch.argmax(probs, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
        _, r_env, done, _ = env.step(action)
        state = env.get_state_for_policy()
        ep_return += float(r_env)
        # Track first visits
        pos = tuple(env.agent_pos)
        for t_idx, t_pos in enumerate(env.target_positions):
            if pos == t_pos and t_idx not in visited_sequence:
                visited_sequence.append(t_idx)
        if done:
            break
    return ep_return


# ----------------------------- Search Loop (next-target) ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=0, help='Optional max workflows to explore (0 = no max; stop by UCB criterion)')
    parser.add_argument('--initial_random', type=int, default=0, help='Deprecated; first selection is random via GP prior')
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
    parser.add_argument('--per_step_penalty', type=float, default=-0.01, help='Additive penalty applied to shaped reward every step')
    parser.add_argument('--eval_episodes_per_update', type=int, default=1, help='Env-only eval episodes to run each update (0 to disable)')
    parser.add_argument('--final_eval_episodes', type=int, default=5, help='Number of env-only episodes for final evaluation per workflow')
    parser.add_argument('--eval_parallel', action='store_true', help='Use vectorized env-only eval instead of sequential episodes')
    parser.add_argument('--eval_parallel_num_envs', type=int, default=25, help='Number of parallel envs for env-only eval when --eval_parallel is set')
    parser.add_argument('--eval_use_canonical', dest='eval_use_canonical', action='store_true', default=True, help='Evaluate env-only rewards using canonical environment workflow [0,1,2,3] (default)')
    parser.add_argument('--no_eval_use_canonical', dest='eval_use_canonical', action='store_false', help='Evaluate env-only rewards using the proposed workflow order')
    # Training behavior
    parser.add_argument('--deterministic_rollouts', action='store_true', help='Use argmax actions during data collection to enforce adherence quickly')
    parser.add_argument('--adherence_target', type=float, default=1.0, help='Target mean adherence to trigger early stopping')
    parser.add_argument('--adherence_patience', type=int, default=2, help='Number of consecutive updates meeting target before early stop')
    parser.add_argument('--early_stop_on_adherence', action='store_true', help='Enable adherence-based early stopping')
    # Penalties inspired by train_parallel_episodes
    parser.add_argument('--penalty_revisit', type=float, default=-2.0)
    parser.add_argument('--penalty_future', type=float, default=-100.0)
    parser.add_argument('--penalty_offworkflow', type=float, default=-50.0)
    parser.add_argument('--length_scale', type=float, default=0.75)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--kernel_type', type=str, default='rbf_rank', choices=['rbf_rank','rbf_pairwise','rbf_mixed','rbf_posunmatch','rbf_possunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=10.0)
    parser.add_argument('--kappa', type=float, default=2.0, help='UCB exploration parameter')
    parser.add_argument('--stop_margin', type=float, default=0.0, help='Early stop if max UCB <= best_score + margin (logged)')
    parser.add_argument('--stop_min_explored', type=int, default=2, help='Require at least this many explored workflows before UCB early stopping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='gpucb_nexttarget')
    parser.add_argument('--use_mp', action='store_true')
    parser.set_defaults(use_mp=True)
    # Epsilon-greedy exploration over workflows
    parser.add_argument('--epsilon', type=float, default=0.05, help='Probability of choosing a random unexplored workflow each iteration')
    parser.add_argument('--epsilon_decay', type=float, default=1.0, help='Multiplicative decay for epsilon per iteration')
    parser.add_argument('--min_epsilon', type=float, default=0.0, help='Lower bound for epsilon after decay')
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
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            'iterations': args.iterations,
            'initial_random': args.initial_random,
            'updates': args.updates,
            'num_envs': args.num_envs,
            'max_steps': args.max_steps,
            'lr': args.lr,
            'gamma': args.gamma,
            'shaping_coef': args.shaping_coef,
            'ppo_epochs': args.ppo_epochs,
            'gae_gamma': args.gae_gamma,
            'minibatch_size': args.minibatch_size,
            'ppo_clip': args.ppo_clip,
            'ppo_value_coef': args.ppo_value_coef,
            'ppo_entropy_coef': args.ppo_entropy_coef,
            'ppo_max_grad_norm': args.ppo_max_grad_norm,
            'per_step_penalty': args.per_step_penalty,
            'eval_episodes_per_update': args.eval_episodes_per_update,
            'final_eval_episodes': args.final_eval_episodes,
            'eval_parallel': bool(args.eval_parallel),
            'eval_parallel_num_envs': int(args.eval_parallel_num_envs),
            'eval_use_canonical': bool(args.eval_use_canonical),
            'deterministic_rollouts': bool(args.deterministic_rollouts),
            'adherence_target': float(args.adherence_target),
            'adherence_patience': int(args.adherence_patience),
            'early_stop_on_adherence': bool(args.early_stop_on_adherence),
            'length_scale': args.length_scale,
            'signal_variance': args.signal_variance,
            'kernel_type': args.kernel_type,
            'rank_scale': args.rank_scale,
            'pairwise_scale': args.pairwise_scale,
            'noise': args.noise,
            'kappa': args.kappa,
            'stop_min_explored': int(args.stop_min_explored),
            'stop_margin': args.stop_margin,
            'debug_rewards': bool(args.debug_rewards),
            'debug_rewards_env': int(args.debug_rewards_env),
            'debug_rewards_steps': int(args.debug_rewards_steps),
            'epsilon': args.epsilon,
            'epsilon_decay': args.epsilon_decay,
            'min_epsilon': args.min_epsilon,
            'seed': args.seed,
            'exp_name': args.exp_name,
            'proc_name': proc_name,
            'pid': int(pid),
            'run_dir': run_dir
        }, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # All candidate workflows (permutations of 0..3)
    candidates: List[List[int]] = [list(p) for p in itertools.permutations([0, 1, 2, 3])]
    candidate_embeddings = [
        build_workflow_embedding(
            w,
            kernel_type=args.kernel_type,
            num_targets=4,
            rank_scale=args.rank_scale,
            pairwise_scale=args.pairwise_scale,
        )
        for w in candidates
    ]

    explored_indices: List[int] = []
    explored_embeddings: List[np.ndarray] = []
    explored_scores: List[float] = []  # env-only returns

    it = 0
    current_epsilon = float(args.epsilon)
    while True:
        if len(explored_indices) >= len(candidates):
            break
        # Compute GP posterior over all candidates (handles empty prior case too)
        mu, std = gp_posterior(
            explored_embeddings,
            explored_scores,
            candidate_embeddings,
            length_scale=args.length_scale,
            noise=args.noise,
            signal_variance=args.signal_variance,
        )
        std = np.maximum(std, 1e-2)
        ucb = mu + args.kappa * std

        # Mask already explored indices
        ucb_masked = ucb.copy()
        ucb_masked[explored_indices] = -np.inf

        # Early stop if no promising workflows remain
        best_score = float(np.max(explored_scores)) if len(explored_scores) > 0 else -np.inf
        max_ucb_unexplored = float(np.max(ucb_masked)) if np.any(np.isfinite(ucb_masked)) else -np.inf
        min_explored_before_stop = int(getattr(args, 'stop_min_explored', 2))
        if len(explored_indices) >= min_explored_before_stop and max_ucb_unexplored <= best_score:
            print(f"[GP-UCB] Stopping early: max UCB among unexplored ({max_ucb_unexplored:.2f}) <= best_score ({best_score:.2f})")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'iteration': it,
                    'event': 'early_stop',
                    'best_score': best_score,
                    'max_ucb_unexplored': max_ucb_unexplored,
                    'stop_margin': args.stop_margin,
                    'min_explored_before_stop': min_explored_before_stop
                }) + "\n")
            break

        unexplored = [i for i in range(len(candidates)) if i not in explored_indices]
        eps_for_iter = current_epsilon
        if len(unexplored) > 0 and np.random.rand() < eps_for_iter:
            next_idx = int(np.random.choice(unexplored))
            phase = 'epsilon'
            print(f"[Epsilon] Iter {it} exploring workflow {candidates[next_idx]} (epsilon={eps_for_iter:.3f})")
        else:
            max_ucb_val = float(np.max(ucb_masked))
            tie_indices = [i for i, val in enumerate(ucb_masked) if np.isfinite(val) and val == max_ucb_val]
            if len(tie_indices) > 1:
                next_idx = int(np.random.choice(tie_indices))
                phase = 'gp_tie'
                print(f"[GP-UCB] Iter {it} tie among {len(tie_indices)} workflows at ucb={max_ucb_val:.2f}; selected {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")
            else:
                next_idx = int(tie_indices[0])
                phase = 'gp'
                print(f"[GP-UCB] Iter {it} selecting workflow {candidates[next_idx]} | mu={mu[next_idx]:.2f}, std={std[next_idx]:.2f}, ucb={ucb[next_idx]:.2f}")

        wf = candidates[next_idx]

        policy, train_stats = train_for_workflow_next(
            wf,
            args.updates,
            args.num_envs,
            args.max_steps,
            args.lr,
            args.gamma,
            args.shaping_coef,
            args.ppo_epochs,
            device,
            args.use_mp,
            args.penalty_revisit,
            args.penalty_future,
            args.penalty_offworkflow,
            args.per_step_penalty,
            args.eval_episodes_per_update,
            bool(args.eval_parallel),
            int(args.eval_parallel_num_envs),
            bool(args.eval_use_canonical),
            args.gae_gamma,
            args.minibatch_size,
            args.debug_rewards,
            args.debug_rewards_env,
            args.debug_rewards_steps,
            args.ppo_clip,
            args.ppo_value_coef,
            args.ppo_entropy_coef,
            args.ppo_max_grad_norm,
            args.deterministic_rollouts,
            args.adherence_target,
            args.adherence_patience,
            args.early_stop_on_adherence,
            run_dir,
        )
        # Use the highest canonical mean observed during training as the workflow performance
        if bool(args.eval_use_canonical) and len(train_stats.get('env_eval_history', [])) > 0:
            score = float(np.max(train_stats['env_eval_history']))
            score_source = 'max_canonical_mean_during_training'
        else:
            # Fallback to deterministic env-only evaluation against proposed workflow
            env = DiagonalCornersEnv(max_steps=args.max_steps)
            eval_returns = [rollout_env_only_next(env, policy, wf, device, deterministic=True) for _ in range(int(args.final_eval_episodes))]
            score = float(np.mean(eval_returns))
            score_source = 'final_eval_mean_proposed'

        explored_indices.append(next_idx)
        explored_embeddings.append(candidate_embeddings[next_idx])
        explored_scores.append(score)

        # After updating with a new calibrated performance, recompute GP estimates for unexplored
        mu_upd, std_upd = gp_posterior(
            explored_embeddings,
            explored_scores,
            candidate_embeddings,
            length_scale=args.length_scale,
            noise=args.noise,
        )
        ucb_upd = mu_upd + args.kappa * std_upd
        unexplored_after = [i for i in range(len(candidates)) if i not in explored_indices]
        if len(unexplored_after) > 0:
            ranked = sorted(unexplored_after, key=lambda i: ucb_upd[i], reverse=True)
            topk = ranked[:5]
            print("[GP-Update] Top unexplored after update:")
            for i_rank, idx_u in enumerate(topk, 1):
                print(f"  {i_rank:2d}. wf {candidates[idx_u]} | mu={mu_upd[idx_u]:.2f}, std={std_upd[idx_u]:.2f}, ucb={ucb_upd[idx_u]:.2f}")

        record = {
            'iteration': it,
            'workflow': wf,
            'score_env_only': score,
            'score_source': score_source,
            'adherence_last': float(train_stats.get('last_mean_adherence', 0.0)),
            'phase': phase,
            'epsilon': float(eps_for_iter)
        }
        try:
            top_before = [
                {'idx': int(i), 'workflow': candidates[i], 'mu': float(mu[i]), 'std': float(std[i]), 'ucb': float(ucb[i])}
                for i in list(np.argsort(ucb)[::-1][:10])
            ]
            top_after = [
                {'idx': int(i), 'workflow': candidates[i], 'mu': float(mu_upd[i]), 'std': float(std_upd[i]), 'ucb': float(ucb_upd[i])}
                for i in list(np.argsort(ucb_upd)[::-1][:10])
            ]
            record.update({'gp_top_before': top_before, 'gp_top_after': top_after})
        except Exception:
            pass
        record.update({
            'env_eval_last': float(train_stats.get('env_eval_last', 0.0)),
            'env_eval_history_len': int(len(train_stats.get('env_eval_history', []))),
            'env_eval_when_adh1_len': int(len(train_stats.get('env_eval_when_adh1', [])))
        })
        if phase.startswith('gp'):
            record.update({'mu': float(mu[next_idx]), 'std': float(std[next_idx]), 'ucb': float(ucb[next_idx])})
        if len(unexplored_after) > 0:
            top5 = [{
                'workflow': candidates[idx_u],
                'mu': float(mu_upd[idx_u]),
                'std': float(std_upd[idx_u]),
                'ucb': float(ucb_upd[idx_u])
            } for idx_u in (sorted(unexplored_after, key=lambda i: ucb_upd[i], reverse=True)[:5])]
            record.update({'gp_update_top5': top5})
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + "\n")

        it += 1
        current_epsilon = max(float(args.min_epsilon), float(current_epsilon) * float(args.epsilon_decay))
        if args.iterations > 0 and it >= args.iterations:
            break

    # Summary
    best_idx = int(np.argmax(explored_scores))
    print("\n=== Workflow Search (Next-Target) Completed ===")
    print(f"Explored {len(explored_indices)} workflows")
    print(f"Best workflow: {candidates[explored_indices[best_idx]]} with score {explored_scores[best_idx]:.2f}")
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump({
            'explored_count': int(len(explored_indices)),
            'best_workflow': candidates[explored_indices[best_idx]],
            'best_score': float(explored_scores[best_idx])
        }, f, indent=2)


if __name__ == '__main__':
    main()


