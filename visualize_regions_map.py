"""Visualize the diagonal regions environment map."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Grid setup
grid_size = 20
start_pos = (10, 10)

# Regions: (r_min, r_max, c_min, c_max)
regions = [
    (0, 5, 0, 5),       # R0: bottom-left
    (14, 19, 14, 19),   # R1: top-right
    (14, 19, 0, 5),     # R2: top-left
    (0, 5, 14, 19),     # R3: bottom-right
]

region_names = ['R0: Bottom-Left', 'R1: Top-Right', 'R2: Top-Left', 'R3: Bottom-Right']
region_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=120)

# Draw grid
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()  # row 0 at top in typical grid visualization

# Grid lines
for i in range(grid_size + 1):
    ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
    ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)

# Draw regions as rectangles
for idx, (r_min, r_max, c_min, c_max) in enumerate(regions):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=2,
        edgecolor=region_colors[idx],
        facecolor=region_colors[idx],
        alpha=0.3,
        label=region_names[idx]
    )
    ax.add_patch(rect)
    
    # Add region label at center
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    ax.text(center_c, center_r, f'R{idx}', fontsize=20, fontweight='bold',
            ha='center', va='center', color=region_colors[idx])

# Mark start position
sr, sc = start_pos
ax.plot(sc, sr, marker='o', markersize=15, color='black', markeredgewidth=2,
        markerfacecolor='white', label='Start (10, 10)', zorder=10)
ax.text(sc, sr, 'S', fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)

# Styling
ax.set_xlabel('Column', fontsize=14, fontweight='bold')
ax.set_ylabel('Row', fontsize=14, fontweight='bold')
ax.set_title('Diagonal Regions Environment (20×20 Grid)\nTask: Enter 4 regions in specified order', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, frameon=True)
ax.set_xticks(range(0, grid_size, 2))
ax.set_yticks(range(0, grid_size, 2))
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/diagonal_regions_map.png', dpi=150, bbox_inches='tight')
print("Saved: diagonal_regions_map.png")
plt.close()

# Create workflow example figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=120)
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()

for i in range(grid_size + 1):
    ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
    ax.axvline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)

# Example workflow: [0, 1, 2, 3]
example_workflow = [0, 1, 2, 3]

for idx, (r_min, r_max, c_min, c_max) in enumerate(regions):
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    rect = patches.Rectangle(
        (c_min - 0.5, r_min - 0.5),
        width,
        height,
        linewidth=2,
        edgecolor=region_colors[idx],
        facecolor=region_colors[idx],
        alpha=0.3,
    )
    ax.add_patch(rect)
    
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    
    # Show visit order (0-indexed)
    order_in_workflow = example_workflow.index(idx)
    ax.text(center_c, center_r, f'R{idx}\n(visit: {order_in_workflow})', fontsize=18, fontweight='bold',
            ha='center', va='center', color=region_colors[idx])

# Draw arrow path
path_centers = []
for wf_idx in example_workflow:
    r_min, r_max, c_min, c_max = regions[wf_idx]
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    path_centers.append((center_c, center_r))

# Start to first region
ax.annotate('', xy=path_centers[0], xytext=(sc, sr),
            arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.6))

# Between regions
for i in range(len(path_centers) - 1):
    ax.annotate('', xy=path_centers[i + 1], xytext=path_centers[i],
                arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.6))

ax.plot(sc, sr, marker='o', markersize=15, color='black', markeredgewidth=2,
        markerfacecolor='white', zorder=10)
ax.text(sc, sr, 'S', fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)

ax.set_xlabel('Column', fontsize=14, fontweight='bold')
ax.set_ylabel('Row', fontsize=14, fontweight='bold')
ax.set_title('Example Workflow: [0, 1, 2, 3]\nVisit Order: R0(0) → R1(1) → R2(2) → R3(3)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(0, grid_size, 2))
ax.set_yticks(range(0, grid_size, 2))
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('/home/ubuntu/RL-Workflow-Search/diagonal_regions_workflow_example.png', dpi=150, bbox_inches='tight')
print("Saved: diagonal_regions_workflow_example.png")
plt.close()

