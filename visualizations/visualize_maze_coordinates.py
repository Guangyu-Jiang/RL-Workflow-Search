"""
Visualize the maze with coordinate labels to help identify milestone locations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
import gym_maze


def visualize_maze_with_coordinates(env_name='maze-sample-5x5-v0'):
    """
    Create a detailed visualization of the maze with coordinate labels
    """
    # Create environment
    env = gym.make(env_name)
    
    # Get maze structure
    if hasattr(env, 'maze_view') and hasattr(env.maze_view, 'maze'):
        maze_cells = np.array(env.maze_view.maze.maze_cells)
    else:
        print("Could not extract maze structure")
        return
    
    # Get entrance and goal
    entrance = env.maze_view.entrance
    goal = env.maze_view.goal
    
    print("=" * 60)
    print("MAZE COORDINATE SYSTEM")
    print("=" * 60)
    print(f"Coordinate format: [row, column]")
    print(f"  - Start (Entrance): {entrance}")
    print(f"  - Goal: {goal}")
    print(f"  - Maze size: {maze_cells.shape}")
    print(f"\nActions:")
    print(f"  0 = North (move up, decrease row)")
    print(f"  1 = South (move down, increase row)")
    print(f"  2 = West (move left, decrease column)")
    print(f"  3 = East (move right, increase column)")
    
    # Bit flags for walls
    NORTH = 1
    SOUTH = 2
    WEST = 4
    EAST = 8
    
    # Build connectivity graph to identify walkable paths
    connectivity = {}
    for y in range(maze_cells.shape[0]):
        for x in range(maze_cells.shape[1]):
            pos = (y, x)
            neighbors = []
            cell_walls = maze_cells[y, x]
            
            if not (cell_walls & NORTH) and y > 0:
                neighbors.append((y - 1, x))
            if not (cell_walls & SOUTH) and y < maze_cells.shape[0] - 1:
                neighbors.append((y + 1, x))
            if not (cell_walls & WEST) and x > 0:
                neighbors.append((y, x - 1))
            if not (cell_walls & EAST) and x < maze_cells.shape[1] - 1:
                neighbors.append((y, x + 1))
            
            connectivity[pos] = neighbors
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # First subplot: Maze structure with coordinates
    ax1.set_title('Maze Structure with Coordinate Labels', fontsize=14, fontweight='bold')
    
    # Draw maze walls
    for y in range(maze_cells.shape[0]):
        for x in range(maze_cells.shape[1]):
            cell_walls = maze_cells[y, x]
            
            # Draw walls
            if cell_walls & NORTH:
                ax1.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], 'k-', linewidth=2)
            if cell_walls & SOUTH:
                ax1.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], 'k-', linewidth=2)
            if cell_walls & WEST:
                ax1.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], 'k-', linewidth=2)
            if cell_walls & EAST:
                ax1.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], 'k-', linewidth=2)
            
            # Add coordinate labels
            ax1.text(x, y, f'[{y},{x}]', ha='center', va='center', 
                    fontsize=9, color='gray')
    
    # Highlight start and goal
    start_rect = Rectangle((entrance[1] - 0.45, entrance[0] - 0.45), 0.9, 0.9,
                          facecolor='green', alpha=0.5, edgecolor='darkgreen', linewidth=2)
    goal_rect = Rectangle((goal[1] - 0.45, goal[0] - 0.45), 0.9, 0.9,
                         facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=2)
    ax1.add_patch(start_rect)
    ax1.add_patch(goal_rect)
    
    ax1.text(entrance[1], entrance[0] - 0.25, 'START', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkgreen')
    ax1.text(goal[1], goal[0] + 0.25, 'GOAL', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='darkred')
    
    ax1.set_xlim(-0.6, maze_cells.shape[1] - 0.4)
    ax1.set_ylim(maze_cells.shape[0] - 0.4, -0.6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Column', fontsize=12)
    ax1.set_ylabel('Row', fontsize=12)
    ax1.set_xticks(range(maze_cells.shape[1]))
    ax1.set_yticks(range(maze_cells.shape[0]))
    
    # Second subplot: Connectivity graph
    ax2.set_title('Cell Connectivity (Walkable Paths)', fontsize=14, fontweight='bold')
    
    # Draw cells
    for y in range(maze_cells.shape[0]):
        for x in range(maze_cells.shape[1]):
            # Color cells based on connectivity
            num_connections = len(connectivity.get((y, x), []))
            if num_connections == 1:
                color = 'lightcoral'  # Dead end
                label = 'Dead'
            elif num_connections == 2:
                neighbors = connectivity[(y, x)]
                if neighbors[0][0] != neighbors[1][0] and neighbors[0][1] != neighbors[1][1]:
                    color = 'lightblue'  # Corner
                    label = 'Turn'
                else:
                    color = 'lightgray'  # Corridor
                    label = 'Path'
            elif num_connections >= 3:
                color = 'yellow'  # Junction
                label = 'Junc'
            else:
                color = 'white'
                label = ''
            
            rect = Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                           facecolor=color, edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
            
            # Add coordinate and type label
            ax2.text(x, y - 0.1, f'[{y},{x}]', ha='center', va='center', 
                    fontsize=8, color='black')
            if label:
                ax2.text(x, y + 0.15, label, ha='center', va='center', 
                        fontsize=7, color='darkblue', fontweight='bold')
    
    # Draw connections
    for pos, neighbors in connectivity.items():
        for neighbor in neighbors:
            # Only draw each connection once
            if neighbor > pos:
                ax2.plot([pos[1], neighbor[1]], [pos[0], neighbor[0]], 
                        'b-', alpha=0.3, linewidth=1)
    
    # Highlight start and goal
    ax2.plot(entrance[1], entrance[0], 'go', markersize=12, label='Start')
    ax2.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    
    ax2.set_xlim(-0.6, maze_cells.shape[1] - 0.4)
    ax2.set_ylim(maze_cells.shape[0] - 0.4, -0.6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)
    ax2.set_xticks(range(maze_cells.shape[1]))
    ax2.set_yticks(range(maze_cells.shape[0]))
    ax2.legend(loc='upper right')
    
    # Add legend for cell types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='yellow', edgecolor='black', label='Junction (3+ paths)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Corner/Turn'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Corridor'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='Dead End'),
    ]
    ax2.legend(handles=legend_elements, loc='lower left', fontsize=8)
    
    plt.suptitle('Maze Coordinate Reference for Milestone Identification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('maze_coordinate_reference.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print connectivity information
    print("\n" + "=" * 60)
    print("CELL CONNECTIVITY ANALYSIS")
    print("=" * 60)
    
    # Categorize cells
    junctions = []
    corners = []
    dead_ends = []
    corridors = []
    
    for pos, neighbors in connectivity.items():
        num_connections = len(neighbors)
        if num_connections == 1:
            dead_ends.append(pos)
        elif num_connections == 2:
            n1, n2 = neighbors
            if n1[0] != n2[0] and n1[1] != n2[1]:
                corners.append(pos)
            else:
                corridors.append(pos)
        elif num_connections >= 3:
            junctions.append(pos)
    
    print(f"\nJunctions (3+ connections): {junctions}")
    print(f"Corners/Turns: {corners}")
    print(f"Dead Ends: {dead_ends}")
    print(f"Straight Corridors: {corridors}")
    
    print("\n" + "=" * 60)
    print("SUGGESTED MILESTONE CANDIDATES")
    print("=" * 60)
    print("Based on maze structure, consider these locations as milestones:")
    
    # Suggest key milestones
    suggested = []
    
    # Add junctions (important decision points)
    for j in junctions[:3]:
        suggested.append(j)
        print(f"  - Junction at {j}: Key decision point")
    
    # Add strategic corners
    for c in corners:
        # Check if corner is between start and goal
        if c[0] > 0 and c[0] < goal[0] and c[1] > 0 and c[1] < goal[1]:
            suggested.append(c)
            print(f"  - Corner at {c}: Strategic turn point")
            if len(suggested) >= 5:
                break
    
    print("\n" + "=" * 60)
    print("You can now specify your milestone locations using [row, column] format")
    print("Example: milestones = [[1, 2], [2, 3], [3, 1]]")
    
    env.close()
    
    return connectivity, suggested


if __name__ == "__main__":
    connectivity, suggested_milestones = visualize_maze_with_coordinates()
    
    print("\n" + "=" * 60)
    print("READY FOR YOUR MILESTONE INPUT")
    print("=" * 60)
    print("Please provide your identified milestone locations as a list of [row, column] coordinates.")
    print(f"Suggested milestones based on structure: {suggested_milestones}")