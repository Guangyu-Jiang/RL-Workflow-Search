"""
Diagonal Regions Environment - 4 rectangular target regions on a 20x20 grid.

Regions (each 6x6):
- Region 0: bottom-left (rows 0-5, cols 0-5)
- Region 1: top-right (rows 14-19, cols 14-19)
- Region 2: top-left (rows 14-19, cols 0-5)
- Region 3: bottom-right (rows 0-5, cols 14-19)

Agent starts at center (10, 10). Task: enter regions in specified workflow order.
Reward: ordinal reward when entering the i'th required region for the first time.
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Optional, Dict


class DiagonalRegionsEnv(gym.Env):
    """
    Environment with 4 rectangular target regions requiring diagonal traversals.
    Grid: 20x20 (indices 0..19). Start at center (10,10).
    Default order: R0 -> R1 -> R2 -> R3
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, step_penalty: float = -0.01, seed: Optional[int] = None, max_steps: Optional[int] = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.grid_size = 20
        self.num_regions = 4

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0:up, 1:down, 2:left, 3:right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        # Define 4 rectangular regions (inclusive bounds): (r_min, r_max, c_min, c_max)
        self.regions: List[Tuple[int, int, int, int]] = [
            (0, 5, 0, 5),       # R0: bottom-left
            (14, 19, 14, 19),   # R1: top-right
            (14, 19, 0, 5),     # R2: top-left
            (0, 5, 14, 19),     # R3: bottom-right
        ]
        # Region centers for potential computation
        self.region_centers: List[Tuple[int, int]] = [
            (2, 2),       # R0 center
            (16, 16),     # R1 center
            (16, 2),      # R2 center
            (2, 16),      # R3 center
        ]
        self.start_pos: Tuple[int, int] = (10, 10)
        self.correct_order: List[int] = [0, 1, 2, 3]

        # Reward config
        self.step_penalty: float = step_penalty

        # State
        self.agent_pos: List[int] = None
        self.visited_regions: set = None
        self.current_region_idx: int = None
        self.steps: int = 0
        self.max_steps: int = int(max_steps) if max_steps is not None else 1000

    def reset(self, workflow: Optional[List[int]] = None, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = list(self.start_pos)
        self.visited_regions = set()
        self.current_region_idx = 0
        self.steps = 0

        if workflow is not None:
            assert len(workflow) == self.num_regions
            assert set(workflow) == set(range(self.num_regions))
            self.correct_order = list(workflow)
        else:
            self.correct_order = [0, 1, 2, 3]

        return np.array(self.agent_pos, dtype=np.int32)

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def _in_region(self, pos: Tuple[int, int], region_idx: int) -> bool:
        """Check if position is inside the specified region."""
        r_min, r_max, c_min, c_max = self.regions[region_idx]
        r, c = pos
        return r_min <= r <= r_max and c_min <= c <= c_max

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1

        # Move agent within bounds
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # right
            self.agent_pos[1] += 1

        reward = self.step_penalty
        done = False

        # Check region entry
        current_pos = tuple(self.agent_pos)
        for region_idx in range(self.num_regions):
            if self._in_region(current_pos, region_idx):
                # Only reward if it's the required next region and first entry
                if (
                    self.current_region_idx < self.num_regions
                    and region_idx == self.correct_order[self.current_region_idx]
                    and region_idx not in self.visited_regions
                ):
                    # Ordinal reward
                    reward += float(self.current_region_idx + 1)
                    self.visited_regions.add(region_idx)
                    self.current_region_idx += 1

                    if self.current_region_idx >= self.num_regions:
                        done = True
                break

        # Timeout
        if self.steps >= self.max_steps and not done:
            done = True

        info = {
            "visited_regions": sorted(list(self.visited_regions)),
            "current_region_idx": self.current_region_idx,
            "correct_order": self.correct_order,
            "agent_pos": self.agent_pos.copy(),
            "regions": self.regions,
        }

        return np.array(self.agent_pos, dtype=np.int32), float(reward), bool(done), info

    def render(self, mode: str = "human"):
        if mode != "human":
            return
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark regions
        for idx, (r_min, r_max, c_min, c_max) in enumerate(self.regions):
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if grid[r][c] == ".":
                        grid[r][c] = str(idx)
        
        # Mark start
        sr, sc = self.start_pos
        if grid[sr][sc] in [str(i) for i in range(self.num_regions)]:
            grid[sr][sc] = "S"
        
        # Mark agent
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"

        print("\n" + "=" * (self.grid_size * 2 + 1))
        print(f"Workflow: {self.correct_order}")
        nxt = self.correct_order[self.current_region_idx] if self.current_region_idx < len(self.correct_order) else "DONE"
        print(f"Next region: R{nxt}")
        print(f"Visited: {sorted(list(self.visited_regions))}")
        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print("=" * (self.grid_size * 2 + 1))

    def get_state_for_policy(self) -> np.ndarray:
        """State = [agent_r, agent_c] normalized + region centers normalized + visited flags."""
        state = np.zeros(2 + self.num_regions * 2 + self.num_regions, dtype=np.float32)
        norm = float(self.grid_size - 1)
        state[0] = self.agent_pos[0] / norm
        state[1] = self.agent_pos[1] / norm
        for i, (r, c) in enumerate(self.region_centers):
            state[2 + i * 2] = r / norm
            state[2 + i * 2 + 1] = c / norm
        for i in range(self.num_regions):
            state[2 + self.num_regions * 2 + i] = 1.0 if i in self.visited_regions else 0.0
        return state

