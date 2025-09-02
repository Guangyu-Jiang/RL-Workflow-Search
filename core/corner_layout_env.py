"""
Corner Layout Environment - Targets at corners, start at center
This should be easier as all targets are equidistant from center
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Optional


class CornerLayoutEnv(gym.Env):
    """
    Simple environment with targets at corners and start at center.
    This layout minimizes conflicts between workflow order and physical proximity.
    """
    
    def __init__(self, grid_size: int = 10, seed: Optional[int] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_targets = 4
        
        if seed is not None:
            np.random.seed(seed)
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # Fixed layout: targets at corners, start at center
        # Using corners but with 1 cell padding to make them reachable
        self.target_positions = [
            (1, 1),                          # T0: top-left corner
            (1, self.grid_size - 2),        # T1: top-right corner
            (self.grid_size - 2, self.grid_size - 2),  # T2: bottom-right corner
            (self.grid_size - 2, 1)         # T3: bottom-left corner
        ]
        
        # Start at center
        self.start_pos = (self.grid_size // 2, self.grid_size // 2)
        
        # State variables
        self.agent_pos = None
        self.visited_targets = None
        self.current_target_idx = None
        self.correct_order = None
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2
        
    def reset(self, workflow: Optional[List[int]] = None) -> np.ndarray:
        """Reset environment with optional workflow specification"""
        self.agent_pos = list(self.start_pos)
        self.visited_targets = set()
        self.current_target_idx = 0
        self.steps = 0
        
        # Set workflow order
        if workflow is not None:
            assert len(workflow) == self.num_targets
            assert set(workflow) == set(range(self.num_targets))
            self.correct_order = workflow
        else:
            # Default clockwise order
            self.correct_order = [0, 1, 2, 3]
        
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
        
        # Mark start position if agent not there
        if tuple(self.agent_pos) != self.start_pos:
            grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        
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
