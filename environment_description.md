# Gridworld with Obstacles Environment

## Environment Description

We evaluate our workflow search method on a gridworld navigation task with obstacles. The environment consists of a 30×30 discrete grid where an agent must navigate to visit four checkpoint regions in a specified order while avoiding randomly placed obstacles.

### Environment Specifications

- **Grid Size**: 30×30 cells
- **Start Position**: Center of the grid at (15, 15)
- **Checkpoint Regions**: Four 4×4 regions positioned at the corners:
  - CP0: Top-left (rows 2-5, columns 2-5)
  - CP1: Bottom-right (rows 24-27, columns 24-27)
  - CP2: Top-right (rows 2-5, columns 24-27)
  - CP3: Bottom-left (rows 24-27, columns 2-5)
- **Obstacles**: Randomly placed with density 0.15 (approximately 15% of cells), ensuring all checkpoints remain reachable from the start position
- **Action Space**: Discrete 4 actions (up, down, left, right)
- **Observation Space**: Agent's current position coordinates (row, column)
- **Maximum Episode Length**: 1500 steps

### Task Description

The agent begins at the center of the grid and must visit all four checkpoint regions in a specified order (workflow). The canonical workflow is [0, 1, 2, 3], which requires the agent to traverse diagonally across the grid: from top-left (CP0) to bottom-right (CP1), then to top-right (CP2), and finally to bottom-left (CP3). This creates a challenging navigation pattern that requires the agent to plan efficient paths around obstacles.

### Reward Structure

The reward design is **strictly based on the workflow order**. The agent only receives checkpoint rewards when visiting checkpoints in the exact sequence specified by the workflow.

- **Step Penalty**: -0.01 per step to encourage efficient navigation
- **Checkpoint Reward**: The agent receives a reward only when entering the **correct next checkpoint** in the workflow sequence. The reward equals the checkpoint's ordinal position in the workflow (1, 2, 3, or 4). Visiting checkpoints out of order or revisiting already-visited checkpoints yields no reward.
- **Episode Completion**: The episode terminates when all four checkpoints are visited in the correct workflow order or when the maximum step limit is reached

This strict reward structure ensures that the agent must learn to follow the workflow sequence precisely, making it a suitable testbed for evaluating methods that can discover and optimize workflow orderings.

### Challenge

The random obstacle placement creates a dynamic navigation challenge where the agent must learn to:
1. Plan paths around obstacles to reach each checkpoint
2. Optimize the traversal order to minimize total path length
3. Adapt to different obstacle configurations across episodes

This environment tests the agent's ability to learn efficient navigation policies while respecting workflow constraints, making it an ideal benchmark for evaluating workflow search methods against standard reinforcement learning baselines.

