"""
Training script for improved layout environment
Uses PPO with strong reward shaping to enforce workflow adherence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict
import argparse
import json
from datetime import datetime
import os

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.improved_layout_env import ImprovedLayoutEnv


class WorkflowConditionedPolicy(nn.Module):
    """Policy network conditioned on workflow"""
    
    def __init__(self, state_dim: int, workflow_dim: int, hidden_dim: int = 256, num_actions: int = 4):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + workflow_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, workflow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value"""
        # Concatenate state and workflow
        x = torch.cat([state, workflow], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Get outputs
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action(self, state: torch.Tensor, workflow: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        action_logits, value = self.forward(state, workflow)
        
        # Get action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Get log probability
        log_prob = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob, action_probs, value


class PPOTrainer:
    """PPO trainer for workflow-conditioned policy"""
    
    def __init__(self, state_dim: int, workflow_dim: int, num_actions: int = 4,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create policy
        self.policy = WorkflowConditionedPolicy(state_dim, workflow_dim, num_actions=num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Training stats
        self.update_count = 0
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns"""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_value = values[t]
        
        return advantages, returns
    
    def update(self, trajectories: List[Dict], epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Update policy using collected trajectories"""
        # Prepare data
        all_states = []
        all_workflows = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            # Compute advantages and returns
            advantages, returns = self.compute_gae(traj['rewards'], traj['values'], traj['dones'])
            
            # Add to buffers
            all_states.extend(traj['states'])
            all_workflows.extend(traj['workflows'])
            all_actions.extend(traj['actions'])
            all_old_log_probs.extend(traj['log_probs'])
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        # Convert to tensors
        states = torch.FloatTensor(all_states).to(self.device)
        workflows = torch.FloatTensor(all_workflows).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        n_samples = len(states)
        n_updates = 0
        
        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_workflows = workflows[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_logits, values = self.policy(batch_states, batch_workflows)
                
                # Get log probabilities
                log_probs = F.log_softmax(action_logits, dim=-1)
                batch_log_probs = log_probs.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                probs = F.softmax(action_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Track stats
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        self.update_count += 1
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


def workflow_to_vector(workflow: List[int], num_targets: int) -> np.ndarray:
    """Convert workflow to vector representation"""
    vector = np.zeros(num_targets, dtype=np.float32)
    for i, target in enumerate(workflow):
        vector[target] = (i + 1) / num_targets  # Normalized position in sequence
    return vector


def calculate_adherence(visited_sequence: List[int], workflow: List[int]) -> float:
    """Calculate adherence considering only first visits"""
    # Get first visits only
    seen = set()
    first_visits = []
    for target in visited_sequence:
        if target not in seen and target >= 0:  # Valid target
            seen.add(target)
            first_visits.append(target)
    
    if not first_visits:
        return 0.0
    
    # Check how many match the workflow prefix
    matches = 0
    for i in range(min(len(first_visits), len(workflow))):
        if first_visits[i] == workflow[i]:
            matches += 1
        else:
            break  # Stop at first mismatch
    
    return matches / len(workflow)


def rollout_episode(env: ImprovedLayoutEnv, policy: WorkflowConditionedPolicy, 
                   workflow: List[int], device: torch.device, 
                   deterministic: bool = False) -> Dict:
    """Collect trajectory for one episode"""
    obs = env.reset(workflow)
    state = env.get_state_for_policy()
    workflow_vec = workflow_to_vector(workflow, env.num_targets)
    
    trajectory = {
        'states': [],
        'workflows': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'dones': []
    }
    
    visited_sequence = []
    episode_return = 0
    
    while True:
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        workflow_tensor = torch.FloatTensor(workflow_vec).unsqueeze(0).to(device)
        
        # Get action
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action(state_tensor, workflow_tensor, deterministic)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        state_next = env.get_state_for_policy()
        
        # Track visits
        for idx, pos in enumerate(env.target_positions):
            if tuple(env.agent_pos) == pos and idx not in visited_sequence:
                visited_sequence.append(idx)
        
        # Store transition
        trajectory['states'].append(state)
        trajectory['workflows'].append(workflow_vec)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['log_probs'].append(log_prob.item())
        trajectory['values'].append(value.item())
        trajectory['dones'].append(done)
        
        episode_return += reward
        state = state_next
        
        if done:
            break
    
    # Calculate adherence
    adherence = calculate_adherence(visited_sequence, workflow)
    success = len(visited_sequence) == len(workflow) and visited_sequence == workflow
    
    return {
        'trajectory': trajectory,
        'return': episode_return,
        'adherence': adherence,
        'success': success,
        'visited_sequence': visited_sequence,
        'steps': len(trajectory['rewards'])
    }


def main(args):
    """Main training loop"""
    # Create environment
    env = ImprovedLayoutEnv(grid_size=args.grid_size, num_targets=args.num_targets, 
                            layout=args.layout, seed=args.seed)
    
    # Parse workflow
    workflow = [int(x) for x in args.workflow.split(',')]
    assert len(workflow) == args.num_targets
    
    # Get dimensions
    state_dim = 2 + args.num_targets * 2 + args.num_targets  # pos + target_pos + visited
    workflow_dim = args.num_targets
    
    # Create trainer
    trainer = PPOTrainer(state_dim, workflow_dim, num_actions=4, lr=args.lr)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training on {args.layout} layout with workflow {workflow}")
    print(f"Grid size: {args.grid_size}, Targets: {args.num_targets}")
    print(f"Target positions: {env.target_positions}")
    print(f"Start position: {env.start_pos}")
    print(f"{'='*60}\n")
    
    # Visualize initial layout
    test_env = ImprovedLayoutEnv(grid_size=args.grid_size, num_targets=args.num_targets,
                                 layout=args.layout, seed=args.seed)
    test_env.reset(workflow)
    print("Initial Layout:")
    test_env.render()
    print()
    
    best_success_rate = 0
    best_adherence = 0
    
    for update in range(args.updates):
        # Collect trajectories
        trajectories = []
        episode_returns = []
        episode_adherences = []
        episode_successes = []
        
        for _ in range(args.episodes_per_update):
            result = rollout_episode(env, trainer.policy, workflow, trainer.device)
            trajectories.append(result['trajectory'])
            episode_returns.append(result['return'])
            episode_adherences.append(result['adherence'])
            episode_successes.append(result['success'])
        
        # Update policy
        update_stats = trainer.update(trajectories, epochs=args.ppo_epochs)
        
        # Calculate statistics
        mean_return = np.mean(episode_returns)
        mean_adherence = np.mean(episode_adherences)
        success_rate = np.mean(episode_successes)
        
        # Track best
        if success_rate > best_success_rate or (success_rate == best_success_rate and mean_adherence > best_adherence):
            best_success_rate = success_rate
            best_adherence = mean_adherence
            print(f"New best! Success: {best_success_rate:.1%}, Adherence: {best_adherence:.1%}")
        
        # Print progress
        if update % 10 == 0:
            print(f"Update {update:3d}: Return={mean_return:6.1f}, "
                  f"Adherence={mean_adherence:.1%}, Success={success_rate:.1%}, "
                  f"Loss={update_stats['loss']:.4f}")
            
            # Show sample trajectory
            if update % 50 == 0:
                print("\nSample trajectory:")
                result = rollout_episode(env, trainer.policy, workflow, trainer.device, deterministic=True)
                print(f"  Workflow: {workflow}")
                print(f"  Visited: {result['visited_sequence']}")
                print(f"  Adherence: {result['adherence']:.1%}")
                print(f"  Success: {result['success']}")
                print()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation:")
    print(f"{'='*60}")
    
    eval_returns = []
    eval_adherences = []
    eval_successes = []
    eval_sequences = []
    
    for _ in range(20):
        result = rollout_episode(env, trainer.policy, workflow, trainer.device, deterministic=True)
        eval_returns.append(result['return'])
        eval_adherences.append(result['adherence'])
        eval_successes.append(result['success'])
        eval_sequences.append(result['visited_sequence'])
    
    print(f"Mean Return: {np.mean(eval_returns):.1f} ± {np.std(eval_returns):.1f}")
    print(f"Mean Adherence: {np.mean(eval_adherences):.1%}")
    print(f"Success Rate: {np.mean(eval_successes):.1%}")
    print(f"\nMost common sequences:")
    from collections import Counter
    seq_counts = Counter(tuple(seq) for seq in eval_sequences)
    for seq, count in seq_counts.most_common(3):
        print(f"  {list(seq)}: {count}/20")
    
    # Save model if successful
    if np.mean(eval_successes) > 0.8:
        model_path = f"models/improved_layout_{args.layout}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(trainer.policy.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default="progressive", 
                       choices=["progressive", "diagonal", "zigzag"],
                       help="Layout strategy for target placement")
    parser.add_argument("--workflow", type=str, default="0,1,2,3",
                       help="Workflow order (comma-separated)")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--num_targets", type=int, default=4)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes_per_update", type=int, default=20)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    main(args)
