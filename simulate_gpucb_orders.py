import argparse
import itertools
from typing import List

import numpy as np

# Reuse kernel/GP from the main implementation for fidelity
from workflow_search_gpucb import (
    build_workflow_embedding,
    gp_posterior,
)


def positional_weighted_matches(order: List[int], canonical: List[int]) -> int:
    weighted = 0
    for i in range(min(len(order), len(canonical))):
        if order[i] == canonical[i]:
            weighted += (i + 1)
    return weighted


def theoretical_return(order: List[int], canonical: List[int], step_penalty: float = 0.01, steps: int = 500) -> float:
    # Weighted positional matches minus per-step penalty
    bonus = positional_weighted_matches(order, canonical)
    return float(bonus - step_penalty * steps)


def main():
    parser = argparse.ArgumentParser("Simulate GP-UCB over 24 workflow orders using theoretical returns")
    parser.add_argument('--kernel_type', type=str, default='rbf_rank', choices=['rbf_rank', 'rbf_pairwise', 'rbf_mixed', 'rbf_posunmatch'])
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--pairwise_scale', type=float, default=1.0)
    parser.add_argument('--length_scale', type=float, default=0.75)
    parser.add_argument('--signal_variance', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=10.0)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--select', type=int, default=5, help='Number of GP-UCB selections to simulate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_one_based', action='store_true', help='Print workflows as 1-based indices')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    canonical = [0, 1, 2, 3]

    # All 24 candidates
    candidates: List[List[int]] = [list(p) for p in itertools.permutations([0, 1, 2, 3])]

    # Build embeddings consistent with main script
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

    # Theoretical returns per candidate
    true_returns = np.array([
        theoretical_return(w, canonical) for w in candidates
    ], dtype=np.float64)

    explored_idx: List[int] = []
    explored_embeddings: List[np.ndarray] = []
    explored_scores: List[float] = []

    print("Kernel:", args.kernel_type, "| length_scale=", args.length_scale, "signal_variance=", args.signal_variance, "noise=", args.noise)
    print("Rank scale=", args.rank_scale, "Pairwise scale=", args.pairwise_scale, "Kappa=", args.kappa)
    print()

    for it in range(max(1, int(args.select))):
        # Posterior before selecting next
        mu, std = gp_posterior(
            explored_embeddings,
            explored_scores,
            candidate_embeddings,
            length_scale=args.length_scale,
            noise=args.noise,
            signal_variance=args.signal_variance,
        )
        std = np.maximum(std, 1e-2)
        ucb = mu + float(args.kappa) * std

        # Mask explored
        ucb_masked = ucb.copy()
        if len(explored_idx) > 0:
            ucb_masked[explored_idx] = -np.inf

        # Report top-10 before update
        order_before = list(np.argsort(ucb_masked)[::-1])
        top_before = [i for i in order_before if np.isfinite(ucb_masked[i])][:10]
        print(f"[Iter {it}] Top-10 BEFORE:")
        for rank, i in enumerate(top_before, 1):
            wf_print = [x + 1 for x in candidates[i]] if args.print_one_based else candidates[i]
            print(f"  {rank:2d}. wf={wf_print} | mu={mu[i]:.3f} std={std[i]:.3f} ucb={ucb[i]:.3f} true={true_returns[i]:.2f}")

        # Select next by UCB (tie-break randomly)
        max_val = float(np.max(ucb_masked)) if np.any(np.isfinite(ucb_masked)) else -np.inf
        tie = [i for i, v in enumerate(ucb_masked) if np.isfinite(v) and v == max_val]
        next_i = int(rng.choice(tie)) if len(tie) > 1 else int(tie[0])

        wf_sel = candidates[next_i]
        wf_sel_print = [x + 1 for x in wf_sel] if args.print_one_based else wf_sel
        y_sel = float(true_returns[next_i])
        print(f"\n[Select] wf={wf_sel_print} | mu={mu[next_i]:.3f} std={std[next_i]:.3f} ucb={ucb[next_i]:.3f} -> observe y={y_sel:.2f}\n")

        # Update GP with observed true return
        explored_idx.append(next_i)
        explored_embeddings.append(candidate_embeddings[next_i])
        explored_scores.append(y_sel)

        # Posterior after update
        mu2, std2 = gp_posterior(
            explored_embeddings,
            explored_scores,
            candidate_embeddings,
            length_scale=args.length_scale,
            noise=args.noise,
            signal_variance=args.signal_variance,
        )
        std2 = np.maximum(std2, 1e-2)
        ucb2 = mu2 + float(args.kappa) * std2
        ucb2_masked = ucb2.copy()
        ucb2_masked[explored_idx] = -np.inf

        order_after = list(np.argsort(ucb2_masked)[::-1])
        top_after = [i for i in order_after if np.isfinite(ucb2_masked[i])][:10]
        print(f"[Iter {it}] Top-10 AFTER:")
        for rank, i in enumerate(top_after, 1):
            wf_print = [x + 1 for x in candidates[i]] if args.print_one_based else candidates[i]
            print(f"  {rank:2d}. wf={wf_print} | mu={mu2[i]:.3f} std={std2[i]:.3f} ucb={ucb2[i]:.3f} true={true_returns[i]:.2f}")
        print()

    print("Done. Explored:")
    for i in explored_idx:
        wf_print = [x + 1 for x in candidates[i]] if args.print_one_based else candidates[i]
        print(f"  wf={wf_print} true={true_returns[i]:.2f}")


if __name__ == '__main__':
    main()


