"""
Custom Manual Environment - Easily specify your own map layouts
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Optional, Dict


class CustomManualEnv(gym.Env):
    """
    Environment with manually specified target layouts.
    Easy to modify for testing different configurations.
    """
    
    def __init__(self, layout_config: Dict = None, seed: Optional[int] = None):
        super().__init__()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Default configuration
        default_config = {
            'grid_size': 10,
            'target_positions': [
                (2, 2),   # T0
                (2, 7),   # T1  
                (7, 7),   # T2
                (7, 2)    # T3
            ],
            'start_pos': (0, 0),
            'max_steps_multiplier': 2  # max_steps = grid_size * grid_size * multiplier
        }
        
        # Use provided config or default
        self.config = layout_config if layout_config else default_config
        
        # Extract configuration
        self.grid_size = self.config['grid_size']
        self.target_positions = self.config['target_positions']
        self.start_pos = self.config['start_pos']
        self.num_targets = len(self.target_positions)
        
        # Validate positions
        self._validate_positions()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # State variables
        self.agent_pos = None
        self.visited_targets = None
        self.current_target_idx = None
        self.correct_order = None
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * self.config.get('max_steps_multiplier', 2)
        
    def _validate_positions(self):
        """Validate that all positions are within bounds and unique"""
        all_positions = self.target_positions + [self.start_pos]
        
        for pos in all_positions:
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
                raise ValueError(f"Position {pos} is out of bounds for grid size {self.grid_size}")
        
        # Check for duplicate target positions
        if len(self.target_positions) != len(set(self.target_positions)):
            raise ValueError("Duplicate target positions found")
    
    def reset(self, workflow: Optional[List[int]] = None) -> np.ndarray:
        """Reset environment with optional workflow specification"""
        self.agent_pos = list(self.start_pos)
        self.visited_targets = set()
        self.current_target_idx = 0
        self.steps = 0
        
        # Set workflow order
        if workflow is not None:
            assert len(workflow) == self.num_targets, f"Workflow length {len(workflow)} doesn't match num_targets {self.num_targets}"
            assert set(workflow) == set(range(self.num_targets)), f"Workflow must contain all target indices 0-{self.num_targets-1}"
            self.correct_order = workflow
        else:
            # Random workflow
            self.correct_order = np.random.permutation(self.num_targets).tolist()
        
        return np.array(self.agent_pos, dtype=np.int32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return step information"""
        self.steps += 1
        
        # Move agent
        old_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # right
            self.agent_pos[1] += 1
        
        # Calculate reward
        reward = -0.1  # Step penalty
        done = False
        
        # Check if at a target
        current_pos = tuple(self.agent_pos)
        for target_idx, target_pos in enumerate(self.target_positions):
            if current_pos == target_pos:
                if target_idx == self.correct_order[self.current_target_idx]:
                    # Correct target in sequence
                    if target_idx not in self.visited_targets:
                        reward = 10.0  # Big reward for correct first visit
                        self.visited_targets.add(target_idx)
                        self.current_target_idx += 1
                        
                        # Check if completed
                        if self.current_target_idx >= self.num_targets:
                            reward = 100.0  # Completion bonus
                            done = True
                    else:
                        reward = -1.0  # Small penalty for revisiting
                else:
                    # Wrong target
                    if target_idx in self.correct_order[self.current_target_idx:]:
                        reward = -5.0  # Future target (out of order)
                    else:
                        reward = -2.0  # Past target or not in workflow
        
        # Check timeout
        if self.steps >= self.max_steps:
            done = True
            reward = -10.0  # Timeout penalty
        
        # Info for debugging
        info = {
            'visited_targets': list(self.visited_targets),
            'current_target_idx': self.current_target_idx,
            'correct_order': self.correct_order,
            'agent_pos': self.agent_pos.copy(),
            'target_positions': self.target_positions
        }
        
        return np.array(self.agent_pos, dtype=np.int32), reward, done, info
    
    def render(self, mode='human'):
        """Simple text rendering"""
        if mode != 'human':
            return
        
        # Create grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark targets
        for idx, (x, y) in enumerate(self.target_positions):
            if idx in self.visited_targets:
                grid[x][y] = str(idx).lower()  # Visited targets in lowercase
            else:
                grid[x][y] = str(idx)  # Unvisited in normal case
        
        # Mark agent
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        # Print grid
        print("\n" + "=" * (self.grid_size * 2 + 1))
        print(f"Workflow: {self.correct_order}")
        print(f"Next target: T{self.correct_order[self.current_target_idx] if self.current_target_idx < len(self.correct_order) else 'DONE'}")
        print(f"Visited: {sorted(list(self.visited_targets))}")
        print("-" * (self.grid_size * 2 + 1))
        
        for row in grid:
            print(' '.join(row))
        print("=" * (self.grid_size * 2 + 1))
    
    def get_state_for_policy(self) -> np.ndarray:
        """Get full state representation for policy network"""
        state = np.zeros(2 + self.num_targets * 2 + self.num_targets, dtype=np.float32)
        
        # Agent position (normalized)
        state[0] = self.agent_pos[0] / (self.grid_size - 1)
        state[1] = self.agent_pos[1] / (self.grid_size - 1)
        
        # Target positions (normalized)
        for i, (x, y) in enumerate(self.target_positions):
            state[2 + i*2] = x / (self.grid_size - 1)
            state[2 + i*2 + 1] = y / (self.grid_size - 1)
        
        # Visited status
        for i in range(self.num_targets):
            state[2 + self.num_targets*2 + i] = float(i in self.visited_targets)
        
        return state


# Predefined layout configurations
LAYOUT_CONFIGS = {
    'linear_horizontal': {
        'grid_size': 10,
        'target_positions': [
            (5, 1),   # T0: leftmost
            (5, 3),   # T1: second  
            (5, 6),   # T2: third
            (5, 8)    # T3: rightmost
        ],
        'start_pos': (5, 0)
    },
    
    'linear_vertical': {
        'grid_size': 10,
        'target_positions': [
            (1, 5),   # T0: topmost
            (3, 5),   # T1: second  
            (6, 5),   # T2: third
            (8, 5)    # T3: bottommost
        ],
        'start_pos': (0, 5)
    },
    
    'square_corners': {
        'grid_size': 10,
        'target_positions': [
            (1, 1),   # T0: top-left
            (1, 8),   # T1: top-right
            (8, 8),   # T2: bottom-right
            (8, 1)    # T3: bottom-left
        ],
        'start_pos': (5, 5)  # Center
    },
    
    'cross_pattern': {
        'grid_size': 10,
        'target_positions': [
            (5, 1),   # T0: left
            (1, 5),   # T1: top
            (5, 8),   # T2: right
            (8, 5)    # T3: bottom
        ],
        'start_pos': (5, 5)  # Center
    },
    
    'spiral': {
        'grid_size': 10,
        'target_positions': [
            (3, 3),   # T0: inner top-left
            (3, 6),   # T1: inner top-right
            (6, 6),   # T2: inner bottom-right
            (6, 3)    # T3: inner bottom-left
        ],
        'start_pos': (5, 5)  # Center
    },
    
    'zigzag_tight': {
        'grid_size': 10,
        'target_positions': [
            (2, 2),   # T0
            (2, 4),   # T1
            (4, 4),   # T2
            (4, 6)    # T3
        ],
        'start_pos': (2, 0)
    },
    
    'your_custom_layout': {
        'grid_size': 10,
        'target_positions': [
            # MODIFY THESE POSITIONS TO CREATE YOUR OWN LAYOUT
            # Format: (row, column) where (0,0) is top-left
            (1, 5),   # T0: your position
            (4, 8),   # T1: your position
            (7, 5),   # T2: your position
            (4, 2)    # T3: your position
        ],
        'start_pos': (5, 5)  # Starting position
    }
}


def test_layout(layout_name: str, workflow: List[int] = [0, 1, 2, 3]):
    """Test a specific layout configuration"""
    config = LAYOUT_CONFIGS.get(layout_name)
    if not config:
        print(f"Layout '{layout_name}' not found. Available layouts:")
        for name in LAYOUT_CONFIGS.keys():
            print(f"  - {name}")
        return
    
    env = CustomManualEnv(layout_config=config)
    obs = env.reset(workflow)
    
    print(f"\n{'='*50}")
    print(f"Testing Layout: {layout_name}")
    print(f"{'='*50}")
    print(f"Grid Size: {config['grid_size']}x{config['grid_size']}")
    print(f"Target Positions: {config['target_positions']}")
    print(f"Start Position: {config['start_pos']}")
    print(f"Workflow: {workflow}")
    
    env.render()
    
    return env


if __name__ == "__main__":
    # Test different layouts
    layouts_to_test = ['linear_horizontal', 'square_corners', 'cross_pattern', 'your_custom_layout']
    
    for layout in layouts_to_test:
        test_layout(layout)
    
    print("\n" + "="*60)
    print("HOW TO USE YOUR OWN CUSTOM LAYOUT:")
    print("="*60)
    print("""
1. Edit the 'your_custom_layout' configuration above with your desired positions

2. Or create a new configuration:
   LAYOUT_CONFIGS['my_layout'] = {
       'grid_size': 10,
       'target_positions': [(row1, col1), (row2, col2), ...],
       'start_pos': (start_row, start_col)
   }

3. Use it in training:
   from core.custom_manual_env import CustomManualEnv, LAYOUT_CONFIGS
   
   env = CustomManualEnv(layout_config=LAYOUT_CONFIGS['my_layout'])
   
4. Or create it directly:
   my_config = {
       'grid_size': 8,
       'target_positions': [(1,1), (1,6), (6,6), (6,1)],
       'start_pos': (4, 4)
   }
   env = CustomManualEnv(layout_config=my_config)
""")
