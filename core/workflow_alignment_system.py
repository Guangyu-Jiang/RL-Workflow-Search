"""
Workflow Alignment System with Sparse Milestones

This system implements an alignment function L(T,W) that measures how well
a trajectory T follows a workflow W defined by sparse milestones.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import gym
import gym_maze


@dataclass
class Milestone:
    """Represents a milestone location with optional ordering"""
    position: Tuple[int, int]
    group_id: int  # Milestones in same group are alternatives
    order: int  # Order in which groups should be visited
    radius: float = 1.0  # Influence radius for soft alignment
    
    def __hash__(self):
        return hash((self.position, self.group_id, self.order))


@dataclass  
class Workflow:
    """Represents a workflow as a sequence of milestone choices"""
    milestones: List[Milestone]
    name: str = ""
    
    def get_ordered_positions(self) -> List[Tuple[int, int]]:
        """Get milestone positions in order"""
        sorted_milestones = sorted(self.milestones, key=lambda m: m.order)
        return [m.position for m in sorted_milestones]


class WorkflowAlignmentSystem:
    """
    Manages workflow alignment and reward shaping for sparse milestones
    """
    
    def __init__(self, maze_size: Tuple[int, int], start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze_size = maze_size
        self.start = start
        self.goal = goal
        
        # Alignment function parameters
        self.distance_weight = 0.4  # Weight for distance to next milestone
        self.order_weight = 0.3     # Weight for visiting milestones in order
        self.coverage_weight = 0.3  # Weight for covering all milestones
        
        # Soft alignment parameters
        self.use_soft_alignment = True
        self.temperature = 1.0  # Controls sharpness of distance penalties
        
    def compute_alignment_loss(self, trajectory: List[Tuple[int, int]], 
                              workflow: Workflow) -> Dict[str, float]:
        """
        Compute alignment loss L(T,W) between trajectory T and workflow W
        
        Returns dict with component losses and total loss
        """
        if not trajectory:
            return {'total': 1.0, 'distance': 1.0, 'order': 1.0, 'coverage': 1.0}
        
        milestones = workflow.get_ordered_positions()
        
        # 1. Distance Loss - How close does trajectory get to each milestone?
        distance_losses = []
        milestone_reached = [False] * len(milestones)
        
        for mi, milestone in enumerate(milestones):
            min_dist = float('inf')
            for pos in trajectory:
                dist = self._manhattan_distance(pos, milestone)
                min_dist = min(min_dist, dist)
                if dist <= 1.0:  # Within radius
                    milestone_reached[mi] = True
            
            if self.use_soft_alignment:
                # Soft distance penalty (exponential decay)
                distance_loss = 1 - np.exp(-min_dist / self.temperature)
            else:
                # Hard distance penalty
                distance_loss = min(1.0, min_dist / (self.maze_size[0] + self.maze_size[1]))
            
            distance_losses.append(distance_loss)
        
        avg_distance_loss = np.mean(distance_losses) if distance_losses else 1.0
        
        # 2. Order Loss - Are milestones visited in the correct order?
        order_loss = 0.0
        visit_times = []
        
        for milestone in milestones:
            first_visit = None
            for t, pos in enumerate(trajectory):
                if self._manhattan_distance(pos, milestone) <= 1.0:
                    first_visit = t
                    break
            visit_times.append(first_visit)
        
        # Check if visits are in correct order
        for i in range(len(visit_times) - 1):
            if visit_times[i] is not None and visit_times[i+1] is not None:
                if visit_times[i] > visit_times[i+1]:
                    order_loss += 1.0
        
        if len(milestones) > 1:
            order_loss /= (len(milestones) - 1)
        
        # 3. Coverage Loss - What fraction of milestones were visited?
        coverage_loss = 1.0 - (sum(milestone_reached) / len(milestones))
        
        # Combined loss
        total_loss = (self.distance_weight * avg_distance_loss +
                     self.order_weight * order_loss +
                     self.coverage_weight * coverage_loss)
        
        return {
            'total': total_loss,
            'distance': avg_distance_loss,
            'order': order_loss,
            'coverage': coverage_loss,
            'milestones_reached': sum(milestone_reached),
            'total_milestones': len(milestones)
        }
    
    def compute_reward_shaping(self, state: Tuple[int, int], 
                              next_state: Tuple[int, int],
                              workflow: Workflow,
                              trajectory_so_far: List[Tuple[int, int]]) -> float:
        """
        Compute immediate reward shaping based on workflow alignment
        
        This provides dense rewards to guide the agent towards milestones
        """
        # Find next unvisited milestone
        milestones = workflow.get_ordered_positions()
        next_milestone_idx = 0
        
        for i, milestone in enumerate(milestones):
            visited = any(self._manhattan_distance(pos, milestone) <= 1.0 
                         for pos in trajectory_so_far)
            if not visited:
                next_milestone_idx = i
                break
        else:
            # All milestones visited, head to goal
            return self._compute_progress_reward(state, next_state, self.goal)
        
        next_milestone = milestones[next_milestone_idx]
        return self._compute_progress_reward(state, next_state, next_milestone)
    
    def _compute_progress_reward(self, state: Tuple[int, int],
                                next_state: Tuple[int, int],
                                target: Tuple[int, int]) -> float:
        """Compute reward for making progress towards target"""
        old_dist = self._manhattan_distance(state, target)
        new_dist = self._manhattan_distance(next_state, target)
        
        # Positive reward for getting closer, negative for getting farther
        progress = old_dist - new_dist
        
        # Scale reward
        max_dist = self.maze_size[0] + self.maze_size[1]
        scaled_reward = progress / max_dist
        
        # Bonus for reaching target
        if new_dist <= 1.0:
            scaled_reward += 0.5
        
        return scaled_reward
    
    def _manhattan_distance(self, pos1: Tuple[int, int], 
                          pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class WorkflowGenerator:
    """
    Generates workflows from sparse milestone specifications
    """
    
    def __init__(self, milestone_groups: Dict[int, List[Tuple[int, int]]]):
        """
        milestone_groups: Dict mapping group_id to list of alternative positions
        Example: {0: [(4,1), (4,3)], 1: [(0,4), (2,4)]}
        """
        self.milestone_groups = milestone_groups
        
    def generate_all_workflows(self) -> List[Workflow]:
        """Generate all possible workflows from milestone combinations"""
        workflows = []
        
        # Get all combinations
        import itertools
        
        group_ids = sorted(self.milestone_groups.keys())
        group_choices = [self.milestone_groups[gid] for gid in group_ids]
        
        for combination in itertools.product(*group_choices):
            milestones = []
            for order, (group_id, position) in enumerate(zip(group_ids, combination)):
                milestone = Milestone(
                    position=position,
                    group_id=group_id,
                    order=order
                )
                milestones.append(milestone)
            
            # Create workflow name
            positions_str = " → ".join([f"{pos}" for pos in combination])
            workflow = Workflow(
                milestones=milestones,
                name=positions_str
            )
            workflows.append(workflow)
        
        return workflows


def visualize_workflow_alignment(env_name='maze-sample-5x5-v0'):
    """
    Visualize your proposed milestone-based workflows and alignment function
    """
    # Your specified milestones
    milestone_groups = {
        0: [(0, 4), (2, 4)],  # First choice: top or middle right
        1: [(4, 1), (4, 3)]   # Second choice: different approaches to goal
    }
    
    # Create environment
    env = gym.make(env_name)
    start = tuple(env.maze_view.entrance)
    goal = tuple(env.maze_view.goal)
    maze_size = env.maze_view.maze_size
    
    # Create systems
    alignment_system = WorkflowAlignmentSystem(maze_size, start, goal)
    workflow_generator = WorkflowGenerator(milestone_groups)
    
    # Generate all workflows
    workflows = workflow_generator.generate_all_workflows()
    
    print("=" * 60)
    print("WORKFLOW-BASED ALIGNMENT SYSTEM")
    print("=" * 60)
    print(f"Start: {start}, Goal: {goal}")
    print(f"\nMilestone Groups:")
    for group_id, positions in milestone_groups.items():
        print(f"  Group {group_id} (order {group_id}): {positions}")
    
    print(f"\nGenerated {len(workflows)} workflows:")
    for i, workflow in enumerate(workflows):
        print(f"  Workflow {i+1}: {workflow.name}")
    
    # Simulate different trajectories and compute alignment
    print("\n" + "=" * 60)
    print("ALIGNMENT LOSS EXAMPLES")
    print("=" * 60)
    
    # Example trajectories
    trajectories = {
        "Direct to goal": [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 4), (2, 4), (3, 4), (4, 4)
        ],
        "Via [0,4] and [4,1]": [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # Reach [0,4]
            (1, 4), (1, 3), (1, 2), (1, 1), (2, 1),
            (3, 1), (4, 1),  # Reach [4,1]
            (4, 2), (4, 3), (4, 4)  # To goal
        ],
        "Via [2,4] and [4,3]": [
            (0, 0), (1, 0), (2, 0), (2, 1), (2, 2),
            (2, 3), (2, 4),  # Reach [2,4]
            (3, 4), (4, 4), (4, 3),  # Reach [4,3] 
            (4, 4)  # To goal
        ],
        "Wrong order [4,1] then [0,4]": [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (4, 1),  # Reach [4,1] first (wrong order)
            (3, 1), (2, 1), (1, 1), (0, 1),
            (0, 2), (0, 3), (0, 4),  # Then [0,4]
            (1, 4), (2, 4), (3, 4), (4, 4)
        ]
    }
    
    # Compute alignment for each trajectory with each workflow
    for traj_name, trajectory in trajectories.items():
        print(f"\nTrajectory: {traj_name}")
        print(f"  Length: {len(trajectory)} steps")
        
        for i, workflow in enumerate(workflows):
            losses = alignment_system.compute_alignment_loss(trajectory, workflow)
            print(f"  Workflow {i+1} ({workflow.name}):")
            print(f"    Total Loss: {losses['total']:.3f}")
            print(f"    Components - Distance: {losses['distance']:.3f}, "
                  f"Order: {losses['order']:.3f}, Coverage: {losses['coverage']:.3f}")
            print(f"    Milestones reached: {losses['milestones_reached']}/{losses['total_milestones']}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (workflow, ax) in enumerate(zip(workflows, axes)):
        ax.set_title(f'Workflow {idx+1}: {workflow.name}', fontsize=12, fontweight='bold')
        
        # Draw maze grid
        for i in range(maze_size[0] + 1):
            ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(maze_size[1] + 1):
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw start and goal
        start_rect = Rectangle((start[1] - 0.45, start[0] - 0.45), 0.9, 0.9,
                              facecolor='green', alpha=0.5, edgecolor='darkgreen', linewidth=2)
        goal_rect = Rectangle((goal[1] - 0.45, goal[0] - 0.45), 0.9, 0.9,
                             facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=2)
        ax.add_patch(start_rect)
        ax.add_patch(goal_rect)
        
        # Draw milestones
        colors = ['blue', 'purple', 'orange', 'brown']
        for milestone in workflow.milestones:
            pos = milestone.position
            color = colors[milestone.order % len(colors)]
            
            # Draw influence radius
            circle = Circle((pos[1], pos[0]), milestone.radius, 
                          fill=False, edgecolor=color, linewidth=2, 
                          linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            
            # Draw milestone
            ax.plot(pos[1], pos[0], 'o', color=color, markersize=12)
            ax.text(pos[1], pos[0] - 0.3, f"M{milestone.order+1}", 
                   ha='center', fontsize=10, fontweight='bold')
        
        # Draw workflow path
        positions = [start] + workflow.get_ordered_positions() + [goal]
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            ax.annotate('', xy=(p2[1], p2[0]), xytext=(p1[1], p1[0]),
                       arrowprops=dict(arrowstyle='->', color='blue', 
                                     lw=2, alpha=0.6))
        
        ax.set_xlim(-0.6, maze_size[1] - 0.4)
        ax.set_ylim(maze_size[0] - 0.4, -0.6)
        ax.set_aspect('equal')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sparse Milestone Workflows with Alignment Regions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('workflow_alignment_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env.close()
    
    return alignment_system, workflows


def compare_milestone_strategies():
    """
    Compare sparse vs dense milestone strategies for scalability
    """
    print("\n" + "=" * 60)
    print("SPARSE VS DENSE MILESTONE STRATEGIES")
    print("=" * 60)
    
    maze_sizes = [(5, 5), (10, 10), (20, 20), (50, 50), (100, 100)]
    
    print("\n1. SPARSE MILESTONES (Your Approach)")
    print("-" * 40)
    print("Pros:")
    print("  - Scalable: O(k) where k is number of milestone groups")
    print("  - Interpretable: Each milestone has clear semantic meaning")
    print("  - Efficient: Fewer combinations to explore")
    print("  - Flexible: Can adjust milestone density based on task")
    
    print("\nCons:")
    print("  - Requires domain knowledge to identify good milestones")
    print("  - May miss optimal paths between milestones")
    print("  - Less fine-grained control")
    
    print("\nScalability Analysis:")
    for size in maze_sizes:
        # Assume ~4-8 milestone groups regardless of size
        num_groups = min(4 + size[0]//20, 8)
        alternatives_per_group = 2
        total_workflows = alternatives_per_group ** num_groups
        print(f"  {size[0]}x{size[1]} maze: ~{num_groups} groups → "
              f"{total_workflows} workflows")
    
    print("\n2. DENSE MILESTONES (All Cells)")
    print("-" * 40)
    print("Pros:")
    print("  - Complete coverage: Can represent any path")
    print("  - No domain knowledge required")
    print("  - Optimal path guaranteed to be in search space")
    
    print("\nCons:")
    print("  - Combinatorial explosion: O(n!) paths")
    print("  - Computationally intractable for large mazes")
    print("  - Difficult to learn meaningful patterns")
    
    print("\nScalability Analysis:")
    for size in maze_sizes:
        num_cells = size[0] * size[1]
        # Rough estimate of possible paths (simplified)
        if num_cells <= 25:
            estimate = f"~{num_cells}!"
        else:
            estimate = f">{10**20} (intractable)"
        print(f"  {size[0]}x{size[1]} maze: {num_cells} cells → {estimate} paths")
    
    print("\n3. HYBRID APPROACH (Recommended)")
    print("-" * 40)
    print("Combine sparse milestones with local exploration:")
    print("  - Use sparse milestones for high-level planning")
    print("  - Allow flexible paths between milestones")
    print("  - Adjust milestone density based on maze complexity")
    print("  - Use learned representations for milestone discovery")


if __name__ == "__main__":
    # Visualize the alignment system with your milestones
    alignment_system, workflows = visualize_workflow_alignment()
    
    # Compare strategies
    compare_milestone_strategies()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED IMPLEMENTATION")
    print("=" * 60)
    print("""
For your RL-Workflow approach with sparse milestones:

1. Alignment Function L(T,W):
   - Distance component: Soft penalties based on proximity to milestones
   - Order component: Reward visiting milestones in specified order
   - Coverage component: Encourage visiting all milestones
   
2. Reward Shaping:
   R_total = R_env + α * R_alignment
   where R_alignment = -L(T,W) provides dense guidance

3. Scalability Solution:
   - Use sparse milestones (4-8 groups) for 100x100 mazes
   - Learn milestone proposals using attention mechanisms
   - Hierarchical approach: Coarse milestones → Fine-grained local planning

4. Training Strategy:
   - Start with easier workflows (fewer milestones)
   - Gradually increase complexity
   - Use curriculum learning based on workflow difficulty
    """)