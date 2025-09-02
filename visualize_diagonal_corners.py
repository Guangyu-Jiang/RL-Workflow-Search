"""
Visualize the Diagonal Corners Environment and save to PNG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from core.diagonal_corners_env import DiagonalCornersEnv


def visualize_and_save(filename: str = "diagonal_corners.png"):
    env = DiagonalCornersEnv()
    workflow = [0, 1, 2, 3]
    env.reset(workflow)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Grid
    for i in range(env.grid_size + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
        ax.axvline(x=i, color='lightgray', linewidth=0.5)

    # Targets
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#F1C40F']
    for idx, (r, c) in enumerate(env.target_positions):
        y = c + 0.5
        x = env.grid_size - r - 0.5
        circ = patches.Circle((y, x), 0.35, color=colors[idx], alpha=0.85)
        ax.add_patch(circ)
        ax.text(y, x, f'T{idx}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    # Start
    sr, sc = env.start_pos
    ax.add_patch(patches.Rectangle((sc + 0.1, env.grid_size - sr - 0.9), 0.8, 0.8, edgecolor='black', facecolor='#BBE1FA', linewidth=2))
    ax.text(sc + 0.5, env.grid_size - sr - 0.5, 'START', ha='center', va='center', fontsize=9, color='#0F4C81')

    # Workflow arrows
    workflow = [0, 1, 2, 3]
    for i in range(len(workflow) - 1):
        a = env.target_positions[workflow[i]]
        b = env.target_positions[workflow[i+1]]
        ay, axp = a[1] + 0.5, env.grid_size - a[0] - 0.5
        by, bxp = b[1] + 0.5, env.grid_size - b[0] - 0.5
        ax.annotate('', xy=(by, bxp), xytext=(ay, axp), arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title('Diagonal Corners Map (11x11)')

    ax.text(env.grid_size/2, -0.5, 'Workflow: T0 (0,0) → T1 (10,10) → T2 (10,0) → T3 (0,10)', ha='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to '{filename}'")


if __name__ == "__main__":
    visualize_and_save()


