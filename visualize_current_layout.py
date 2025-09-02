"""
Visualize the current progressive layout and show how to create custom layouts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from core.improved_layout_env import ImprovedLayoutEnv

def visualize_layout(env, workflow, title="Environment Layout"):
    """Create a visual representation of the environment layout"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw grid
    for i in range(env.grid_size + 1):
        ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.5)
    
    # Draw targets with workflow order
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown']
    for idx, (x, y) in enumerate(env.target_positions):
        # Find position in workflow
        workflow_pos = workflow.index(idx)
        color = colors[workflow_pos % len(colors)]
        
        # Draw target
        circle = patches.Circle((y + 0.5, env.grid_size - x - 0.5), 0.3, 
                               color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Add label
        ax.text(y + 0.5, env.grid_size - x - 0.5, f'T{idx}\n#{workflow_pos+1}', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw arrow to next target in workflow
        if workflow_pos < len(workflow) - 1:
            next_target_idx = workflow[workflow_pos + 1]
            next_x, next_y = env.target_positions[next_target_idx]
            
            # Calculate arrow positions (with slight offset to avoid overlap)
            start_y = y + 0.5
            start_x = env.grid_size - x - 0.5
            end_y = next_y + 0.5
            end_x = env.grid_size - next_x - 0.5
            
            # Draw arrow
            ax.annotate('', xy=(end_y, end_x), xytext=(start_y, start_x),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.5))
    
    # Draw start position
    start_x, start_y = env.start_pos
    rect = patches.Rectangle((start_y, env.grid_size - start_x - 1), 1, 1,
                            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5)
    ax.add_patch(rect)
    ax.text(start_y + 0.5, env.grid_size - start_x - 0.5, 'START', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Set limits and labels
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row (inverted)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    legend_text = f"Workflow Order: {' → '.join([f'T{i}' for i in workflow])}"
    ax.text(env.grid_size/2, -0.5, legend_text, ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    """Visualize different layouts"""
    
    # 1. Progressive Layout (current)
    print("="*60)
    print("PROGRESSIVE LAYOUT (Current)")
    print("="*60)
    env = ImprovedLayoutEnv(grid_size=10, num_targets=4, layout='progressive', seed=42)
    workflow = [0, 1, 2, 3]
    env.reset(workflow)
    
    print(f"Target positions: {env.target_positions}")
    print(f"Start position: {env.start_pos}")
    print("\nText representation:")
    env.render()
    
    # Create visualization
    fig = visualize_layout(env, workflow, "Progressive Layout - Forces Sequential Movement")
    plt.savefig('progressive_layout.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'progressive_layout.png'")
    
    # 2. Diagonal Layout
    print("\n" + "="*60)
    print("DIAGONAL LAYOUT")
    print("="*60)
    env_diag = ImprovedLayoutEnv(grid_size=10, num_targets=4, layout='diagonal', seed=42)
    env_diag.reset(workflow)
    
    print(f"Target positions: {env_diag.target_positions}")
    print(f"Start position: {env_diag.start_pos}")
    
    fig2 = visualize_layout(env_diag, workflow, "Diagonal Layout - Maximized Distances")
    plt.savefig('diagonal_layout.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'diagonal_layout.png'")
    
    # 3. Zigzag Layout
    print("\n" + "="*60)
    print("ZIGZAG LAYOUT")
    print("="*60)
    env_zig = ImprovedLayoutEnv(grid_size=10, num_targets=4, layout='zigzag', seed=42)
    env_zig.reset(workflow)
    
    print(f"Target positions: {env_zig.target_positions}")
    print(f"Start position: {env_zig.start_pos}")
    
    fig3 = visualize_layout(env_zig, workflow, "Zigzag Layout - Alternating Pattern")
    plt.savefig('zigzag_layout.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'zigzag_layout.png'")
    
    plt.show()
    
    print("\n" + "="*60)
    print("HOW TO MANUALLY SPECIFY A MAP LAYOUT")
    print("="*60)
    print("""
You can manually specify a map layout by modifying the ImprovedLayoutEnv class.
Here's how:

1. MODIFY THE ENVIRONMENT CLASS:
   Edit /home/ubuntu/RL-Workflow-Search/core/improved_layout_env.py
   
   In the _setup_layout() method, add a new layout option:
   
   elif self.layout == 'custom':
       # Manually specify exact positions
       self.target_positions = [
           (1, 1),   # T0 position (row, col)
           (3, 8),   # T1 position
           (8, 6),   # T2 position  
           (5, 2)    # T3 position
       ]
       self.start_pos = (0, 0)  # Starting position

2. CREATE A CUSTOM LAYOUT FUNCTION:
   You can also create a function that generates layouts based on parameters:
   
   def create_spiral_layout(self):
       # Example: targets in a spiral pattern
       positions = []
       center = self.grid_size // 2
       radius = self.grid_size // 3
       for i in range(self.num_targets):
           angle = 2 * np.pi * i / self.num_targets
           x = int(center + radius * np.cos(angle))
           y = int(center + radius * np.sin(angle))
           positions.append((x, y))
       return positions

3. USE THE CUSTOM LAYOUT:
   python train_stable_layout.py --layout custom --workflow 0,1,2,3

4. LAYOUT DESIGN TIPS:
   - Avoid placing targets too close to each other
   - Consider the workflow order when placing targets
   - Place targets to encourage the desired path
   - Test with different starting positions
   - Use symmetry or patterns for consistency

5. EXAMPLE CUSTOM LAYOUTS:

   # Linear layout (forces specific order)
   self.target_positions = [
       (5, 1),   # T0: leftmost
       (5, 3),   # T1: second  
       (5, 6),   # T2: third
       (5, 8)    # T3: rightmost
   ]
   
   # Square layout (corners)
   self.target_positions = [
       (1, 1),                          # T0: top-left
       (1, self.grid_size-2),          # T1: top-right
       (self.grid_size-2, self.grid_size-2),  # T2: bottom-right
       (self.grid_size-2, 1)           # T3: bottom-left
   ]
   
   # Cross pattern
   mid = self.grid_size // 2
   self.target_positions = [
       (mid, 1),                # T0: left
       (1, mid),                # T1: top
       (mid, self.grid_size-2), # T2: right
       (self.grid_size-2, mid)  # T3: bottom
   ]
""")

if __name__ == "__main__":
    main()
