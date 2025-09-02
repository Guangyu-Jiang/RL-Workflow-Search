"""
Integrated Workflow-based RL with Sparse Milestones

This demonstrates how to integrate the alignment function into 
the RL training loop with PPO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import gym
import gym_maze
from collections import deque
import matplotlib.pyplot as plt

from workflow_alignment_system import (
    Milestone, Workflow, WorkflowAlignmentSystem, WorkflowGenerator
)


class WorkflowConditionedPolicy(nn.Module):
    """
    Policy network conditioned on workflow embedding
    """
    def __init__(self, state_dim: int, workflow_embedding_dim: int, 
                 hidden_dim: int = 128, action_dim: int = 4):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Workflow encoder (simple version - encode milestone positions)
        self.workflow_encoder = nn.Sequential(
            nn.Linear(workflow_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combined policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, workflow_embedding: torch.Tensor):
        state_features = self.state_encoder(state)
        workflow_features = self.workflow_encoder(workflow_embedding)
        combined = torch.cat([state_features, workflow_features], dim=-1)
        
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value
    
    def get_action(self, state: torch.Tensor, workflow_embedding: torch.Tensor,
                   deterministic: bool = False):
        action_logits, value = self.forward(state, workflow_embedding)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action, action_probs, value


class WorkflowRLAgent:
    """
    Complete RL agent with workflow-based exploration and alignment
    """
    
    def __init__(self, env_name: str = 'maze-sample-5x5-v0',
                 milestone_groups: Optional[Dict] = None,
                 alignment_weight: float = 0.5):
        
        # Environment
        self.env = gym.make(env_name)
        self.maze_size = self.env.maze_view.maze_size
        self.start = tuple(self.env.maze_view.entrance)
        self.goal = tuple(self.env.maze_view.goal)
        
        # Milestones and workflows
        if milestone_groups is None:
            # Use your specified milestones
            self.milestone_groups = {
                0: [(0, 4), (2, 4)],  # First choice
                1: [(4, 1), (4, 3)]   # Second choice
            }
        else:
            self.milestone_groups = milestone_groups
        
        # Systems
        self.alignment_system = WorkflowAlignmentSystem(
            self.maze_size, self.start, self.goal
        )
        self.workflow_generator = WorkflowGenerator(self.milestone_groups)
        self.workflows = self.workflow_generator.generate_all_workflows()
        
        # RL components
        self.state_dim = 2  # (row, col) position
        self.workflow_embedding_dim = len(self.milestone_groups) * 2  # Simple encoding
        self.policy = WorkflowConditionedPolicy(
            self.state_dim, self.workflow_embedding_dim
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Training parameters
        self.alignment_weight = alignment_weight
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Tracking
        self.workflow_performance = {i: [] for i in range(len(self.workflows))}
        self.explored_workflows = set()
    
    def encode_workflow(self, workflow: Workflow) -> torch.Tensor:
        """
        Simple workflow encoding - concatenate milestone positions
        """
        positions = workflow.get_ordered_positions()
        # Flatten positions and normalize
        encoding = []
        for pos in positions:
            encoding.extend([pos[0] / self.maze_size[0], 
                           pos[1] / self.maze_size[1]])
        return torch.FloatTensor(encoding)
    
    def compute_augmented_reward(self, 
                                state: Tuple[int, int],
                                next_state: Tuple[int, int],
                                env_reward: float,
                                workflow: Workflow,
                                trajectory: List[Tuple[int, int]]) -> float:
        """
        Combine environment reward with alignment-based reward shaping
        """
        # Get alignment-based reward
        alignment_reward = self.alignment_system.compute_reward_shaping(
            state, next_state, workflow, trajectory
        )
        
        # Combine with environment reward
        total_reward = env_reward + self.alignment_weight * alignment_reward
        
        return total_reward
    
    def collect_rollout(self, workflow: Workflow, num_steps: int = 100):
        """
        Collect trajectory following a workflow
        """
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        trajectory = []
        
        # Reset environment
        state = self.env.reset()
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        
        workflow_embedding = self.encode_workflow(workflow)
        
        for step in range(num_steps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            workflow_tensor = workflow_embedding.unsqueeze(0)
            
            with torch.no_grad():
                action, action_probs, value = self.policy.get_action(
                    state_tensor, workflow_tensor, deterministic=False
                )
            
            # Take action in environment
            next_state, env_reward, done, info = self.env.step(action.item())
            if isinstance(next_state, np.ndarray):
                next_state = tuple(next_state.astype(int))
            
            # Add to trajectory
            trajectory.append(state)
            
            # Compute augmented reward
            aug_reward = self.compute_augmented_reward(
                state, next_state, env_reward, workflow, trajectory
            )
            
            # Store transition
            states.append(state)
            actions.append(action.item())
            rewards.append(aug_reward)
            values.append(value.item())
            dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        # Compute final alignment loss
        alignment_losses = self.alignment_system.compute_alignment_loss(
            trajectory, workflow
        )
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'dones': dones,
            'trajectory': trajectory,
            'alignment_loss': alignment_losses['total'],
            'workflow': workflow
        }
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update_policy(self, rollout_data: Dict):
        """
        Update policy using PPO with alignment loss
        """
        states = torch.FloatTensor(rollout_data['states'])
        actions = torch.LongTensor(rollout_data['actions'])
        old_values = torch.FloatTensor(rollout_data['values'])
        
        workflow_embedding = self.encode_workflow(rollout_data['workflow'])
        workflow_tensor = workflow_embedding.unsqueeze(0).repeat(len(states), 1)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rollout_data['rewards'], 
            rollout_data['values'],
            rollout_data['dones']
        )
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy predictions
        action_logits, values = self.policy(states, workflow_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Compute policy loss (PPO clipped objective)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        old_action_log_probs = action_log_probs.detach()
        
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages.unsqueeze(1),
            clipped_ratio * advantages.unsqueeze(1)
        ).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_coef * value_loss - 
                     self.entropy_coef * entropy)
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def select_workflow(self, acquisition_function: str = 'ucb') -> Workflow:
        """
        Select next workflow to explore based on acquisition function
        """
        if acquisition_function == 'ucb':
            # Upper Confidence Bound
            best_score = -float('inf')
            best_workflow = None
            
            for i, workflow in enumerate(self.workflows):
                # Get performance history
                performances = self.workflow_performance[i]
                
                if len(performances) == 0:
                    # Unexplored workflow - high priority
                    score = float('inf')
                else:
                    mean_perf = np.mean(performances)
                    std_perf = np.std(performances) if len(performances) > 1 else 1.0
                    exploration_bonus = 2.0 * std_perf / np.sqrt(len(performances))
                    score = mean_perf + exploration_bonus
                
                if score > best_score:
                    best_score = score
                    best_workflow = workflow
            
            return best_workflow
        
        elif acquisition_function == 'random':
            return np.random.choice(self.workflows)
        
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_function}")
    
    def train(self, num_episodes: int = 100, steps_per_episode: int = 100):
        """
        Main training loop
        """
        episode_rewards = []
        alignment_losses = []
        
        for episode in range(num_episodes):
            # Select workflow for this episode
            workflow = self.select_workflow(acquisition_function='ucb')
            workflow_idx = self.workflows.index(workflow)
            
            # Collect rollout
            rollout = self.collect_rollout(workflow, steps_per_episode)
            
            # Update policy
            losses = self.update_policy(rollout)
            
            # Track performance
            total_reward = sum(rollout['rewards'])
            episode_rewards.append(total_reward)
            alignment_losses.append(rollout['alignment_loss'])
            
            # Update workflow performance
            self.workflow_performance[workflow_idx].append(total_reward)
            self.explored_workflows.add(workflow_idx)
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Workflow: {workflow.name}")
                print(f"  Total Reward: {total_reward:.3f}")
                print(f"  Alignment Loss: {rollout['alignment_loss']:.3f}")
                print(f"  Explored Workflows: {len(self.explored_workflows)}/{len(self.workflows)}")
                print(f"  Losses - Policy: {losses['policy_loss']:.4f}, "
                      f"Value: {losses['value_loss']:.4f}")
        
        return episode_rewards, alignment_losses


def demo_integrated_system():
    """
    Demonstrate the integrated workflow-based RL system
    """
    print("=" * 60)
    print("INTEGRATED WORKFLOW-BASED RL WITH SPARSE MILESTONES")
    print("=" * 60)
    
    # Create agent with your specified milestones
    milestone_groups = {
        0: [(0, 4), (2, 4)],  # First choice: top or middle right
        1: [(4, 1), (4, 3)]   # Second choice: different approaches to goal  
    }
    
    agent = WorkflowRLAgent(
        env_name='maze-sample-5x5-v0',
        milestone_groups=milestone_groups,
        alignment_weight=0.5  # Balance between env reward and alignment
    )
    
    print(f"\nMilestone Groups:")
    for group_id, positions in milestone_groups.items():
        print(f"  Group {group_id}: {positions}")
    
    print(f"\nGenerated {len(agent.workflows)} workflows:")
    for i, workflow in enumerate(agent.workflows):
        print(f"  {i+1}. {workflow.name}")
    
    print("\n" + "-" * 60)
    print("TRAINING WITH WORKFLOW EXPLORATION")
    print("-" * 60)
    
    # Train agent
    episode_rewards, alignment_losses = agent.train(
        num_episodes=50,  # Reduced for demo
        steps_per_episode=100
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alignment losses
    axes[0, 1].plot(alignment_losses)
    axes[0, 1].set_title('Alignment Loss L(T,W)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Workflow performance comparison
    workflow_names = [w.name for w in agent.workflows]
    mean_performances = []
    std_performances = []
    
    for i in range(len(agent.workflows)):
        perfs = agent.workflow_performance[i]
        if perfs:
            mean_performances.append(np.mean(perfs))
            std_performances.append(np.std(perfs))
        else:
            mean_performances.append(0)
            std_performances.append(0)
    
    x_pos = np.arange(len(workflow_names))
    axes[1, 0].bar(x_pos, mean_performances, yerr=std_performances, capsize=5)
    axes[1, 0].set_title('Workflow Performance Comparison')
    axes[1, 0].set_xlabel('Workflow')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f"W{i+1}" for i in range(len(workflow_names))], rotation=0)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Exploration frequency
    exploration_counts = [len(agent.workflow_performance[i]) 
                         for i in range(len(agent.workflows))]
    axes[1, 1].bar(x_pos, exploration_counts)
    axes[1, 1].set_title('Workflow Exploration Frequency')
    axes[1, 1].set_xlabel('Workflow')
    axes[1, 1].set_ylabel('Times Explored')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f"W{i+1}" for i in range(len(workflow_names))], rotation=0)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add workflow legend
    legend_text = "\n".join([f"W{i+1}: {w.name}" for i, w in enumerate(agent.workflows)])
    fig.text(0.02, 0.02, legend_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Integrated Workflow-based RL Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('integrated_workflow_rl_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Explored {len(agent.explored_workflows)}/{len(agent.workflows)} workflows")
    print(f"Best performing workflow: W{np.argmax(mean_performances)+1} - {agent.workflows[np.argmax(mean_performances)].name}")
    print(f"Mean final reward: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"Mean final alignment loss: {np.mean(alignment_losses[-10:]):.3f}")
    
    agent.env.close()


if __name__ == "__main__":
    demo_integrated_system()