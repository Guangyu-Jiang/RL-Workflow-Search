"""
Visualize the corner layout environment with targets at corners and start at center
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from core.corner_layout_env import CornerLayoutEnv


def create_corner_layout_visualization():
    """Create a detailed visualization of the corner layout"""
    
    # Create environment
    env = CornerLayoutEnv(grid_size=10)
    workflow = [0, 1, 2, 3]  # Clockwise order
    env.reset(workflow)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Main layout visualization
    ax1 = plt.subplot(1, 2, 1)
    
    # Draw grid
    for i in range(env.grid_size + 1):
        ax1.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.7)
        ax1.axvline(x=i, color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Color scheme for targets (rainbow progression)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green
    
    # Draw targets with labels
    for idx, (x, y) in enumerate(env.target_positions):
        # Find position in workflow
        workflow_pos = workflow.index(idx)
        
        # Draw target circle
        circle = patches.Circle((y + 0.5, env.grid_size - x - 0.5), 0.35, 
                               color=colors[workflow_pos], alpha=0.8, zorder=3)
        ax1.add_patch(circle)
        
        # Add target label
        ax1.text(y + 0.5, env.grid_size - x - 0.5, f'T{idx}', 
                ha='center', va='center', fontsize=16, fontweight='bold', 
                color='white', zorder=4)
        
        # Add workflow order number
        ax1.text(y + 0.5, env.grid_size - x - 0.5 - 0.6, f'#{workflow_pos+1}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=colors[workflow_pos])
    
    # Draw workflow path with curved arrows
    for i in range(len(workflow) - 1):
        current_target = workflow[i]
        next_target = workflow[i + 1]
        
        current_pos = env.target_positions[current_target]
        next_pos = env.target_positions[next_target]
        
        # Convert to plot coordinates
        start_y = current_pos[1] + 0.5
        start_x = env.grid_size - current_pos[0] - 0.5
        end_y = next_pos[1] + 0.5
        end_x = env.grid_size - next_pos[0] - 0.5
        
        # Draw curved arrow
        ax1.annotate('', xy=(end_y, end_x), xytext=(start_y, start_x),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5, alpha=0.6,
                                  connectionstyle="arc3,rad=0.3"))
    
    # Draw start position
    start_x, start_y = env.start_pos
    rect = FancyBboxPatch((start_y + 0.1, env.grid_size - start_x - 0.9), 0.8, 0.8,
                          boxstyle="round,pad=0.1", linewidth=3, 
                          edgecolor='#2ECC71', facecolor='#A8E6CF', alpha=0.8, zorder=2)
    ax1.add_patch(rect)
    ax1.text(start_y + 0.5, env.grid_size - start_x - 0.5, 'START', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#27AE60')
    
    # Set limits and labels
    ax1.set_xlim(0, env.grid_size)
    ax1.set_ylim(0, env.grid_size)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Column', fontsize=12)
    ax1.set_ylabel('Row (inverted)', fontsize=12)
    ax1.set_title('Corner Layout Environment\n(Targets at Corners, Start at Center)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid coordinates
    for i in range(env.grid_size):
        ax1.text(-0.3, env.grid_size - i - 0.5, str(i), ha='center', va='center', fontsize=8)
        ax1.text(i + 0.5, -0.3, str(i), ha='center', va='center', fontsize=8)
    
    # Second subplot: Distance analysis
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate distances from start to each target
    distances = []
    for idx, (tx, ty) in enumerate(env.target_positions):
        dist = abs(tx - env.start_pos[0]) + abs(ty - env.start_pos[1])
        distances.append(dist)
    
    # Bar chart of distances
    target_names = [f'T{i}\n(pos #{workflow.index(i)+1})' for i in range(4)]
    bars = ax2.bar(target_names, distances, color=[colors[workflow.index(i)] for i in range(4)], alpha=0.7)
    
    # Add value labels on bars
    for bar, dist in zip(bars, distances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist} steps', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Manhattan Distance from Start', fontsize=12)
    ax2.set_title('Distance Analysis\n(All targets equidistant from center)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(distances) + 2)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add workflow order annotation
    workflow_text = f"Workflow Order: {' → '.join([f'T{i}' for i in workflow])}"
    fig.text(0.5, 0.02, workflow_text, ha='center', fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.8))
    
    # Add layout properties text
    properties = [
        "• All targets equidistant from start (6 steps)",
        "• No proximity bias between targets",
        "• Clear paths without obstacles",
        "• Symmetric layout for easy learning"
    ]
    props_text = '\n'.join(properties)
    fig.text(0.98, 0.5, props_text, ha='right', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the figure
    plt.savefig('corner_layout_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'corner_layout_visualization.png'")
    
    return fig, env


def create_simple_text_visualization():
    """Create a simple text visualization"""
    env = CornerLayoutEnv(grid_size=10)
    workflow = [0, 1, 2, 3]
    env.reset(workflow)
    
    print("\n" + "="*60)
    print("CORNER LAYOUT ENVIRONMENT")
    print("="*60)
    print(f"Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"Start Position: {env.start_pos} (center)")
    print(f"Target Positions:")
    for i, pos in enumerate(env.target_positions):
        print(f"  T{i}: {pos} ({'top-left' if i==0 else 'top-right' if i==1 else 'bottom-right' if i==2 else 'bottom-left'})")
    print(f"Workflow Order: {workflow}")
    print(f"Max Steps: {env.max_steps}")
    
    print("\nDistance Analysis:")
    for i, (tx, ty) in enumerate(env.target_positions):
        dist = abs(tx - env.start_pos[0]) + abs(ty - env.start_pos[1])
        print(f"  T{i}: {dist} steps from start")
    
    print("\nEnvironment Grid:")
    env.render()
    
    print("\nKey Features:")
    print("✓ All targets are equidistant from the starting position")
    print("✓ No obstacles to navigate around")
    print("✓ Symmetric layout reduces learning complexity")
    print("✓ Clear separation between all targets")
    print("✓ Agent starts at the center with equal access to all corners")


if __name__ == "__main__":
    # Create visualizations
    create_simple_text_visualization()
    fig, env = create_corner_layout_visualization()
    
    # Also create a simple matplotlib show for different workflows
    print("\n" + "="*60)
    print("Testing different workflow orders:")
    print("="*60)
    
    test_workflows = [
        [0, 1, 2, 3],  # Clockwise
        [0, 2, 1, 3],  # Diagonal pattern
        [3, 2, 1, 0],  # Counter-clockwise
        [1, 3, 0, 2],  # Cross pattern
    ]
    
    for wf in test_workflows:
        print(f"\nWorkflow {wf}:")
        path = []
        for target_id in wf:
            pos = env.target_positions[target_id]
            path.append(f"T{target_id}({pos[0]},{pos[1]})")
        print(f"  Path: {' → '.join(path)}")
    
    print("\n✓ Corner layout visualization complete!")
    print("✓ Image saved as 'corner_layout_visualization.png'")
    
    # Show the plot (if in interactive mode)
    try:
        plt.show()
    except:
        pass
