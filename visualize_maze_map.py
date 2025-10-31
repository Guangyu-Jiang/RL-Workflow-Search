"""Visualize the obstacle maze environment map."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from core.obstacle_maze_env import ObstacleMazeEnv

# Create maze instance
np.random.seed(42)
env = ObstacleMazeEnv(wall_density=0.15)

grid_size = env.grid_size
start_pos = env.start_pos
checkpoints = env.checkpoints
checkpoint_centers = env.checkpoint_centers
walls = env.walls

checkpoint_names = [f'CP{i}' for i in range(4)]
checkpoint_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create map figure
fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=120)

ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Draw walls
wall_positions = np.argwhere(walls == 1)
for (r, c) in wall_positions:
    rect = patches.Rectangle(
        (c - 0.5, r - 0.5), 1, 1,
        linewidth=0, facecolor='gray', alpha=0.6
    )
    ax.add_patch(rect)

# Draw checkpoints
for idx, (r_min, r_max, c_min, c_max) in enumerate(checkpoints):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=2,
        edgecolor=checkpoint_colors[idx],
        facecolor=checkpoint_colors[idx],
        alpha=0.4,
        label=f'{checkpoint_names[idx]} ({checkpoint_centers[idx]})'
    )
    ax.add_patch(rect)
    
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    ax.text(center_c, center_r, f'CP{idx}', fontsize=14, fontweight='bold',
            ha='center', va='center', color=checkpoint_colors[idx])

# Mark start
sr, sc = start_pos
ax.plot(sc, sr, marker='o', markersize=12, color='black', markeredgewidth=2,
        markerfacecolor='white', label=f'Start ({sr}, {sc})', zorder=10)
ax.text(sc, sr, 'S', fontsize=10, fontweight='bold', ha='center', va='center', zorder=11)

ax.set_xlabel('Column', fontsize=14, fontweight='bold')
ax.set_ylabel('Row', fontsize=14, fontweight='bold')
ax.set_title(f'Obstacle Maze Environment ({grid_size}×{grid_size})\n4 Checkpoints with {int(walls.sum())} Walls (density={env.wall_density:.2f})', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, frameon=True)
ax.set_xticks(range(0, grid_size, 5))
ax.set_yticks(range(0, grid_size, 5))
ax.grid(True, alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/obstacle_maze_map.png', dpi=150, bbox_inches='tight')
print(f"Saved: obstacle_maze_map.png (walls: {walls.sum()}/{grid_size**2})")
plt.close()

# Create workflow example figure
np.random.seed(42)
env = ObstacleMazeEnv(wall_density=0.15)
example_workflow = [0, 1, 2, 3]

fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=120)
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Walls
wall_positions = np.argwhere(walls == 1)
for (r, c) in wall_positions:
    rect = patches.Rectangle(
        (c - 0.5, r - 0.5), 1, 1,
        linewidth=0, facecolor='gray', alpha=0.4
    )
    ax.add_patch(rect)

# Checkpoints with visit order
for idx, (r_min, r_max, c_min, c_max) in enumerate(checkpoints):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=2,
        edgecolor=checkpoint_colors[idx],
        facecolor=checkpoint_colors[idx],
        alpha=0.4,
    )
    ax.add_patch(rect)
    
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    order_in_workflow = example_workflow.index(idx)
    ax.text(center_c, center_r, f'CP{idx}\n(visit: {order_in_workflow})', fontsize=12, fontweight='bold',
            ha='center', va='center', color=checkpoint_colors[idx])

# Draw path
path_centers = []
for wf_idx in example_workflow:
    r_min, r_max, c_min, c_max = checkpoints[wf_idx]
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    path_centers.append((center_c, center_r))

# Start to first
ax.annotate('', xy=path_centers[0], xytext=(sc, sr),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.6))

# Between checkpoints
for i in range(len(path_centers) - 1):
    ax.annotate('', xy=path_centers[i + 1], xytext=path_centers[i],
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.6))

ax.plot(sc, sr, marker='o', markersize=12, color='black', markeredgewidth=2,
        markerfacecolor='white', zorder=10)
ax.text(sc, sr, 'S', fontsize=10, fontweight='bold', ha='center', va='center', zorder=11)

ax.set_xlabel('Column', fontsize=14, fontweight='bold')
ax.set_ylabel('Row', fontsize=14, fontweight='bold')
ax.set_title('Example Workflow: [0, 1, 2, 3]\nMust navigate around walls to visit checkpoints in order', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(0, grid_size, 5))
ax.set_yticks(range(0, grid_size, 5))
ax.grid(True, alpha=0.2, linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/obstacle_maze_workflow_example.png', dpi=150, bbox_inches='tight')
print("Saved: obstacle_maze_workflow_example.png")
plt.close()

