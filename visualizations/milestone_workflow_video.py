"""
Milestone-based Workflow System with Video Rendering
Renders agent movement using gym's built-in renderer (like maze_episode.gif)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
import gym
import gym_maze
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional, Set
import os


class MazeAnalyzer:
    """Analyzes maze structure to identify milestones and key locations"""
    
    # Bit flags for walls
    NORTH = 1
    SOUTH = 2
    WEST = 4
    EAST = 8
    
    def __init__(self, maze_cells: np.ndarray):
        """
        Initialize with maze cells
        maze_cells: 2D numpy array where each cell contains wall flags
        """
        self.maze_cells = maze_cells
        self.height, self.width = maze_cells.shape
        
        # Build connectivity graph
        self.connectivity = self._build_connectivity_graph()
        
    def _build_connectivity_graph(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Build a graph of which cells are connected (no walls between them)"""
        graph = {}
        
        for y in range(self.height):
            for x in range(self.width):
                pos = (y, x)
                neighbors = []
                cell_walls = self.maze_cells[y, x]
                
                # Check each direction
                if not (cell_walls & self.NORTH) and y > 0:
                    neighbors.append((y - 1, x))
                if not (cell_walls & self.SOUTH) and y < self.height - 1:
                    neighbors.append((y + 1, x))
                if not (cell_walls & self.WEST) and x > 0:
                    neighbors.append((y, x - 1))
                if not (cell_walls & self.EAST) and x < self.width - 1:
                    neighbors.append((y, x + 1))
                
                graph[pos] = neighbors
        
        return graph
    
    def identify_milestones(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Identify key locations in the maze based on structural features
        """
        milestones = []
        
        # 1. Junctions: cells with 3+ open paths
        junctions = self._find_junctions()
        
        # 2. Corners: cells where the path turns
        corners = self._find_corners()
        
        # 3. Dead ends and their entrances
        dead_ends = self._find_dead_ends()
        
        # Combine all milestones
        all_milestones = set()
        all_milestones.update(junctions)
        all_milestones.update(corners)
        all_milestones.update(dead_ends)
        
        # Remove start and goal from milestones
        all_milestones.discard(start)
        all_milestones.discard(goal)
        
        # Rank milestones by importance
        ranked = self._rank_milestones(list(all_milestones), start, goal)
        
        return ranked[:8]  # Return top 8 milestones
    
    def _find_junctions(self) -> List[Tuple[int, int]]:
        """Find cells with 3 or more open paths"""
        junctions = []
        for pos, neighbors in self.connectivity.items():
            if len(neighbors) >= 3:
                junctions.append(pos)
        return junctions
    
    def _find_corners(self) -> List[Tuple[int, int]]:
        """Find cells where the path turns"""
        corners = []
        for pos, neighbors in self.connectivity.items():
            if len(neighbors) == 2:
                n1, n2 = neighbors
                # Check if neighbors form an L-shape (not straight)
                if n1[0] != n2[0] and n1[1] != n2[1]:
                    corners.append(pos)
        return corners
    
    def _find_dead_ends(self) -> List[Tuple[int, int]]:
        """Find dead ends"""
        dead_ends = []
        for pos, neighbors in self.connectivity.items():
            if len(neighbors) == 1:
                dead_ends.append(pos)
        return dead_ends
    
    def _rank_milestones(self, milestones: List[Tuple[int, int]], 
                        start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Rank milestones by strategic importance"""
        scored_milestones = []
        
        for milestone in milestones:
            score = 0
            
            # Connectivity score
            num_connections = len(self.connectivity.get(milestone, []))
            score += num_connections * 10
            
            # Position score (prefer milestones between start and goal)
            dist_to_start = abs(milestone[0] - start[0]) + abs(milestone[1] - start[1])
            dist_to_goal = abs(milestone[0] - goal[0]) + abs(milestone[1] - goal[1])
            total_dist = dist_to_start + dist_to_goal
            direct_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
            if total_dist <= direct_dist * 1.5:
                score += 20
            
            scored_milestones.append((score, milestone))
        
        scored_milestones.sort(reverse=True)
        return [m[1] for m in scored_milestones]


class WorkflowProposer:
    """Proposes workflows based on maze structure analysis"""
    
    def __init__(self, analyzer: MazeAnalyzer):
        self.analyzer = analyzer
    
    def propose_workflows(self, start: Tuple[int, int], goal: Tuple[int, int], 
                         milestones: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Propose different workflow strategies based on maze structure
        """
        workflows = {}
        
        # 1. Direct Path: Attempt to go somewhat directly
        workflows["Direct"] = self._create_direct_workflow(start, goal)
        
        # 2. Milestone Route: Visit key milestones
        workflows["Milestone"] = self._create_milestone_workflow(start, goal, milestones)
        
        # 3. Wall Following: Follow walls
        workflows["WallFollow"] = self._create_wall_following_workflow(start, goal)
        
        # 4. Exploration: Cover different regions
        workflows["Explore"] = self._create_exploration_workflow(start, goal)
        
        return workflows
    
    def _create_direct_workflow(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a workflow that attempts to go relatively directly"""
        workflow = [start]
        
        # Create intermediate waypoints
        steps = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
        for i in range(1, min(steps, 3)):  # Limit to 3 intermediate points
            ratio = i / steps
            y = int(start[0] + ratio * (goal[0] - start[0]))
            x = int(start[1] + ratio * (goal[1] - start[1]))
            
            if (y, x) in self.analyzer.connectivity:
                if (y, x) not in workflow:
                    workflow.append((y, x))
        
        workflow.append(goal)
        return workflow
    
    def _create_milestone_workflow(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                  milestones: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Create a workflow that visits important milestones"""
        workflow = [start]
        
        if milestones:
            # Select up to 2 milestones that are roughly between start and goal
            relevant = []
            for m in milestones[:5]:
                dist_via_m = (abs(m[0] - start[0]) + abs(m[1] - start[1]) + 
                            abs(goal[0] - m[0]) + abs(goal[1] - m[1]))
                direct_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
                if dist_via_m <= direct_dist * 2:
                    relevant.append(m)
            
            relevant.sort(key=lambda m: abs(m[0] - start[0]) + abs(m[1] - start[1]))
            
            for m in relevant[:2]:
                if m not in workflow:
                    workflow.append(m)
        
        workflow.append(goal)
        return workflow
    
    def _create_wall_following_workflow(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a workflow based on wall following strategy"""
        workflow = [start]
        
        # Add a corner point if available
        for pos, neighbors in self.analyzer.connectivity.items():
            if len(neighbors) == 2:
                n1, n2 = neighbors
                if n1[0] != n2[0] and n1[1] != n2[1]:  # It's a corner
                    if pos != start and pos != goal:
                        workflow.append(pos)
                        break
        
        workflow.append(goal)
        return workflow
    
    def _create_exploration_workflow(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a workflow that explores different regions"""
        workflow = [start]
        
        # Add center point
        mid_y, mid_x = self.analyzer.height // 2, self.analyzer.width // 2
        if (mid_y, mid_x) in self.analyzer.connectivity:
            if (mid_y, mid_x) != start and (mid_y, mid_x) != goal:
                workflow.append((mid_y, mid_x))
        
        workflow.append(goal)
        return workflow


def follow_workflow_policy(current_pos: Tuple[int, int], 
                          workflow: List[Tuple[int, int]], 
                          workflow_idx: int,
                          connectivity: Dict) -> int:
    """
    Simple policy to follow a workflow by moving towards the next waypoint
    Returns action (0=N, 1=S, 2=W, 3=E)
    """
    if workflow_idx >= len(workflow):
        return np.random.randint(4)  # Random if no more waypoints
    
    target = workflow[workflow_idx]
    
    # If we reached the current waypoint, target the next one
    if current_pos == target:
        workflow_idx += 1
        if workflow_idx >= len(workflow):
            return np.random.randint(4)
        target = workflow[workflow_idx]
    
    # Move towards target using available connections
    neighbors = connectivity.get(current_pos, [])
    if not neighbors:
        return np.random.randint(4)
    
    # Choose neighbor that gets us closest to target
    best_neighbor = min(neighbors, 
                       key=lambda n: abs(n[0] - target[0]) + abs(n[1] - target[1]))
    
    # Convert position difference to action
    dy = best_neighbor[0] - current_pos[0]
    dx = best_neighbor[1] - current_pos[1]
    
    if dy == -1:
        return 0  # North
    elif dy == 1:
        return 1  # South
    elif dx == -1:
        return 2  # West
    elif dx == 1:
        return 3  # East
    else:
        return np.random.randint(4)


def run_episode_with_workflow(env_name: str, workflow: List[Tuple[int, int]], 
                             workflow_name: str, analyzer: MazeAnalyzer,
                             max_steps: int = 200) -> Tuple[List, List, List]:
    """
    Run an episode following a workflow and capture frames
    Returns: trajectory, rewards, frames
    """
    env = gym.make(env_name)
    
    # Get the raw environment to bypass wrapper issues
    raw_env = env
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    
    # Reset environment
    state = raw_env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    current_pos = tuple(state.astype(int)) if isinstance(state, np.ndarray) else state
    
    trajectory = [current_pos]
    rewards = []
    frames = []
    
    # Capture initial frame
    frame = raw_env.render(mode='rgb_array')
    if frame is not None:
        frames.append(frame)
    
    done = False
    total_reward = 0
    workflow_idx = 0
    
    for step in range(max_steps):
        if done:
            break
        
        # Get action based on workflow
        action = follow_workflow_policy(current_pos, workflow, workflow_idx, 
                                       analyzer.connectivity)
        
        # Take step using raw environment
        result = raw_env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        next_pos = tuple(next_state.astype(int)) if isinstance(next_state, np.ndarray) else next_state
        
        # Update workflow index if we reached a waypoint
        if workflow_idx < len(workflow) and next_pos == workflow[workflow_idx]:
            workflow_idx += 1
        
        current_pos = next_pos
        trajectory.append(current_pos)
        rewards.append(reward)
        total_reward += reward
        
        # Capture frame
        frame = raw_env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    
    env.close()
    
    print(f"   {workflow_name}: {len(trajectory)} steps, reward: {total_reward:.2f}")
    
    return trajectory, rewards, frames


def save_video(frames: List[np.ndarray], filename: str, fps: int = 5):
    """Save frames as video file"""
    if not frames:
        print(f"No frames to save for {filename}")
        return
    
    height, width = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"   Saved video: {filename}")


def save_gif(frames: List[np.ndarray], filename: str, duration: int = 200):
    """Save frames as GIF file"""
    if not frames:
        print(f"No frames to save for {filename}")
        return
    
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save as GIF
    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"   Saved GIF: {filename}")


def create_comparison_video(all_frames: Dict[str, List], output_name: str = "workflow_comparison"):
    """Create a side-by-side comparison video of different workflows"""
    if not all_frames:
        print("   No frames to create comparison video")
        return
    
    # Get dimensions from first workflow's first frame
    first_frames = list(all_frames.values())[0]
    if not first_frames:
        print("   First workflow has no frames")
        return
    
    height, width = first_frames[0].shape[:2]
    num_workflows = len(all_frames)
    
    # Create grid layout
    grid_cols = min(2, num_workflows)
    grid_rows = (num_workflows + grid_cols - 1) // grid_cols
    
    # Create combined frames
    max_frames = max(len(frames) for frames in all_frames.values())
    combined_frames = []
    
    print(f"   Creating comparison grid: {grid_rows}x{grid_cols} for {num_workflows} workflows")
    print(f"   Total frames to process: {max_frames}")
    
    for frame_idx in range(max_frames):
        # Create blank canvas for grid
        canvas_height = height * grid_rows + 50 * grid_rows  # Extra space for titles
        canvas_width = width * grid_cols
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place each workflow's frame
        for idx, (name, frames) in enumerate(all_frames.items()):
            row = idx // grid_cols
            col = idx % grid_cols
            
            if frame_idx < len(frames):
                frame = frames[frame_idx]
            else:
                frame = frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8)
            
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            
            # Calculate position
            y_start = row * (height + 50) + 50
            y_end = y_start + height
            x_start = col * width
            x_end = x_start + width
            
            # Ensure dimensions match
            if y_end <= canvas_height and x_end <= canvas_width:
                canvas[y_start:y_end, x_start:x_end] = frame
                
                # Add title using OpenCV
                title_y = y_start - 10
                cv2.putText(canvas, name, (x_start + 10, title_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        combined_frames.append(canvas)
    
    # Save combined video
    if combined_frames:
        save_video(combined_frames, f"{output_name}.mp4", fps=5)
        save_gif(combined_frames[:50], f"{output_name}.gif", duration=200)  # Limit GIF to 50 frames
    else:
        print("   No combined frames created")


def main():
    """Main function to demonstrate milestone-based workflows with video rendering"""
    print("=" * 70)
    print("MILESTONE-BASED WORKFLOW SYSTEM WITH VIDEO RENDERING")
    print("=" * 70)
    
    env_name = 'maze-sample-5x5-v0'
    
    # Create environment for analysis
    env = gym.make(env_name)
    
    # Get maze structure
    if hasattr(env, 'maze_view') and hasattr(env.maze_view, 'maze'):
        maze_cells = np.array(env.maze_view.maze.maze_cells)
    else:
        print("Could not extract maze structure")
        return
    
    env.close()
    
    # Initialize analyzer
    analyzer = MazeAnalyzer(maze_cells)
    
    # Define start and goal
    start = (0, 0)
    goal = (maze_cells.shape[0] - 1, maze_cells.shape[1] - 1)
    
    # Identify milestones
    print("\n1. Analyzing maze structure...")
    milestones = analyzer.identify_milestones(start, goal)
    print(f"   Identified {len(milestones)} key milestones")
    
    # Propose workflows
    proposer = WorkflowProposer(analyzer)
    workflows = proposer.propose_workflows(start, goal, milestones)
    
    print(f"\n2. Proposed {len(workflows)} workflow strategies:")
    for name, workflow in workflows.items():
        print(f"   - {name}: {len(workflow)} waypoints")
    
    # Run episodes and capture videos
    print("\n3. Running episodes with different workflows...")
    all_frames = {}
    
    for name, workflow in workflows.items():
        trajectory, rewards, frames = run_episode_with_workflow(
            env_name, workflow, name, analyzer
        )
        
        # Save individual video and GIF
        if frames:
            save_video(frames, f"workflow_{name.lower()}.mp4", fps=5)
            save_gif(frames[:50], f"workflow_{name.lower()}.gif", duration=200)
            all_frames[name] = frames
    
    # Create comparison video
    print("\n4. Creating comparison video...")
    create_comparison_video(all_frames, "milestone_workflows_comparison")
    
    # Create static visualization
    print("\n5. Creating static visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (name, workflow) in enumerate(workflows.items()):
        ax = axes[idx]
        
        # Draw maze structure
        for y in range(maze_cells.shape[0]):
            for x in range(maze_cells.shape[1]):
                cell_walls = maze_cells[y, x]
                
                # Draw walls
                if cell_walls & analyzer.NORTH:
                    ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], 'k-', linewidth=2)
                if cell_walls & analyzer.SOUTH:
                    ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], 'k-', linewidth=2)
                if cell_walls & analyzer.WEST:
                    ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], 'k-', linewidth=2)
                if cell_walls & analyzer.EAST:
                    ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], 'k-', linewidth=2)
        
        # Draw workflow path
        for i, pos in enumerate(workflow):
            color = plt.cm.coolwarm(i / len(workflow))
            circle = Circle((pos[1], pos[0]), 0.3,
                          facecolor=color, edgecolor='black', 
                          alpha=0.7, linewidth=1, zorder=3)
            ax.add_patch(circle)
            
            # Draw connections
            if i > 0:
                prev_pos = workflow[i-1]
                ax.plot([prev_pos[1], pos[1]], [prev_pos[0], pos[0]], 
                       'b-', alpha=0.5, linewidth=2, zorder=1)
            
            # Add waypoint number
            ax.text(pos[1], pos[0], str(i), 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', 
                   color='white', zorder=4)
        
        # Highlight milestones
        for milestone in milestones:
            ax.plot(milestone[1], milestone[0], 'y*', markersize=15, zorder=2)
        
        ax.set_title(f'{name} Workflow', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.6, maze_cells.shape[1] - 0.4)
        ax.set_ylim(maze_cells.shape[0] - 0.4, -0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xticks(range(maze_cells.shape[1]))
        ax.set_yticks(range(maze_cells.shape[0]))
    
    plt.suptitle('Milestone-based Workflows\n(Yellow stars = Identified Milestones)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('milestone_workflows_static.png', dpi=150, bbox_inches='tight')
    
    print("\n6. All visualizations saved:")
    print("   - Individual videos: workflow_*.mp4")
    print("   - Individual GIFs: workflow_*.gif")
    print("   - Comparison video: milestone_workflows_comparison.mp4")
    print("   - Comparison GIF: milestone_workflows_comparison.gif")
    print("   - Static visualization: milestone_workflows_static.png")
    
    print("\nDone! Check the generated videos to see the agent following different workflows.")


if __name__ == "__main__":
    main()