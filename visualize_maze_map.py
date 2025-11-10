"""Visualize the gridworld with obstacles environment map."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from core.obstacle_maze_env import ObstacleMazeEnv

# Create environment instance
np.random.seed(42)
env = ObstacleMazeEnv(wall_density=0.15)

grid_size = env.grid_size
start_pos = env.start_pos
checkpoints = env.checkpoints
checkpoint_centers = env.checkpoint_centers
walls = env.walls

checkpoint_names = [f'CP{i}' for i in range(4)]
# More professional color scheme
checkpoint_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# Create map figure with better styling
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150, facecolor='white')
ax.set_facecolor('#fafafa')

ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Draw background grid (lighter)
for i in range(grid_size):
    for j in range(grid_size):
        rect = patches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            linewidth=0.5, edgecolor='#e0e0e0', facecolor='white', alpha=0.3
        )
        ax.add_patch(rect)

# Draw obstacles/walls with better styling
wall_positions = np.argwhere(walls == 1)
for (r, c) in wall_positions:
    rect = patches.Rectangle(
        (c - 0.5, r - 0.5), 1, 1,
        linewidth=0.5, edgecolor='#424242', facecolor='#757575', alpha=0.8
    )
    ax.add_patch(rect)

# Draw checkpoints with improved styling
for idx, (r_min, r_max, c_min, c_max) in enumerate(checkpoints):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    
    # Outer border
    border_rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=3,
        edgecolor=checkpoint_colors[idx],
        facecolor='none',
        zorder=5
    )
    ax.add_patch(border_rect)
    
    # Inner fill
    fill_rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=0,
        facecolor=checkpoint_colors[idx],
        alpha=0.25,
        zorder=4
    )
    ax.add_patch(fill_rect)
    
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    
    # Add checkpoint label with better styling
    ax.text(center_c, center_r, f'CP{idx}', 
            fontsize=16, fontweight='bold',
            ha='center', va='center', 
            color=checkpoint_colors[idx],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=checkpoint_colors[idx], linewidth=2, alpha=0.9),
            zorder=6)

# Mark start position with better styling
sr, sc = start_pos
start_circle = patches.Circle(
    (sc, sr), radius=0.4,
    linewidth=3, edgecolor='#000000', facecolor='#ffffff', zorder=10
)
ax.add_patch(start_circle)
ax.text(sc, sr, 'S', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#000000', zorder=11)

# Styling
ax.set_xlabel('Column', fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel('Row', fontsize=14, fontweight='bold', color='#333333')
ax.set_title(f'Gridworld with Obstacles ({grid_size}×{grid_size})\n4 Checkpoint Regions with Random Obstacles (density={env.wall_density:.2f})', 
             fontsize=16, fontweight='bold', pad=20, color='#1a1a1a')

# Grid ticks
ax.set_xticks(range(0, grid_size, 5))
ax.set_yticks(range(0, grid_size, 5))
ax.tick_params(colors='#666666', labelsize=10)
ax.grid(True, alpha=0.3, linewidth=0.8, color='#cccccc', linestyle='-')

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#999999')
ax.spines['bottom'].set_color('#999999')

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/gridworld_obstacles_map.png', 
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: gridworld_obstacles_map.png (obstacles: {walls.sum()}/{grid_size**2})")
plt.close()

# Create workflow example figure
np.random.seed(42)
env = ObstacleMazeEnv(wall_density=0.15)
example_workflow = [0, 1, 2, 3]

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150, facecolor='white')
ax.set_facecolor('#fafafa')

ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

# Draw background grid
for i in range(grid_size):
    for j in range(grid_size):
        rect = patches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            linewidth=0.5, edgecolor='#e0e0e0', facecolor='white', alpha=0.3
        )
        ax.add_patch(rect)

# Draw obstacles
wall_positions = np.argwhere(walls == 1)
for (r, c) in wall_positions:
    rect = patches.Rectangle(
        (c - 0.5, r - 0.5), 1, 1,
        linewidth=0.5, edgecolor='#424242', facecolor='#757575', alpha=0.8
    )
    ax.add_patch(rect)

# Checkpoints with visit order (indexed 1-4)
for idx, (r_min, r_max, c_min, c_max) in enumerate(checkpoints):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    
    # Get visit order (1-indexed)
    order_in_workflow = example_workflow.index(idx) + 1
    
    # Border
    border_rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=3,
        edgecolor=checkpoint_colors[idx],
        facecolor='none',
        zorder=5
    )
    ax.add_patch(border_rect)
    
    # Fill
    fill_rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=0,
        facecolor=checkpoint_colors[idx],
        alpha=0.25,
        zorder=4
    )
    ax.add_patch(fill_rect)
    
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    
    # Label with 1-indexed visit order
    ax.text(center_c, center_r, f'#{order_in_workflow}', 
            fontsize=16, fontweight='bold',
            ha='center', va='center', 
            color=checkpoint_colors[idx],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=checkpoint_colors[idx], linewidth=2, alpha=0.9),
            zorder=6)

# Draw path with arrows in correct order
# Workflow [0, 1, 2, 3] means: CP0 (top-left) → CP1 (bottom-right) → CP2 (top-right) → CP3 (bottom-left)
path_centers = []
for wf_idx in example_workflow:
    r_min, r_max, c_min, c_max = checkpoints[wf_idx]
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    path_centers.append((center_c, center_r))

# Verify the path order matches expected directions:
# 1. Top-left (CP0) → Bottom-right (CP1)
# 2. Bottom-right (CP1) → Top-right (CP2)
# 3. Top-right (CP2) → Bottom-left (CP3)

# Start to first checkpoint (S → CP0)
ax.annotate('', xy=path_centers[0], xytext=(sc, sr),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1976d2', 
                          alpha=0.8, connectionstyle='arc3,rad=0.1'), zorder=7)

# Between checkpoints: from i to i+1
for i in range(len(path_centers) - 1):
    # Arrow from checkpoint i to checkpoint i+1
    ax.annotate('', xy=path_centers[i + 1], xytext=path_centers[i],
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#1976d2', 
                              alpha=0.8, connectionstyle='arc3,rad=0.1'), zorder=7)

# Start position
start_circle = patches.Circle(
    (sc, sr), radius=0.4,
    linewidth=3, edgecolor='#000000', facecolor='#ffffff', zorder=10
)
ax.add_patch(start_circle)
ax.text(sc, sr, 'S', fontsize=14, fontweight='bold', 
        ha='center', va='center', color='#000000', zorder=11)

# Styling
ax.set_xlabel('Column', fontsize=14, fontweight='bold', color='#333333')
ax.set_ylabel('Row', fontsize=14, fontweight='bold', color='#333333')
ax.set_title('Example Workflow: [0, 1, 2, 3]\nAgent must navigate around obstacles to visit checkpoints in order', 
             fontsize=16, fontweight='bold', pad=20, color='#1a1a1a')
ax.set_xticks(range(0, grid_size, 5))
ax.set_yticks(range(0, grid_size, 5))
ax.tick_params(colors='#666666', labelsize=10)
ax.grid(True, alpha=0.3, linewidth=0.8, color='#cccccc', linestyle='-')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#999999')
ax.spines['bottom'].set_color('#999999')

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/gridworld_obstacles_workflow_example.png', 
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved: gridworld_obstacles_workflow_example.png")
plt.close()

