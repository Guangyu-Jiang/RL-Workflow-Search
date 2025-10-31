"""
Obstacle Maze Environment - 4 checkpoint regions on 30x30 grid with random obstacles.

Checkpoints (each 4×4) - positioned for diagonal crisscross traversal:
- CP0: top-left (rows 2-5, cols 2-5)
- CP1: bottom-right (rows 24-27, cols 24-27)
- CP2: top-right (rows 2-5, cols 24-27)
- CP3: bottom-left (rows 24-27, cols 2-5)

Canonical order [0,1,2,3] creates full-map diagonal traversal pattern.
Agent starts at center (15, 15).
Random walls are placed with controllable density; connectivity is verified.
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Optional, Dict
from collections import deque


class ObstacleMazeEnv(gym.Env):
    """
    Environment with 4 checkpoint regions and random obstacles.
    Grid: 30×30. Start at center (15,15).
    Default order: CP0 → CP1 → CP2 → CP3
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, step_penalty: float = -0.01, wall_density: float = 0.15, 
                 seed: Optional[int] = None, max_steps: Optional[int] = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.grid_size = 30
        self.num_checkpoints = 4
        self.wall_density = float(wall_density)

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0:up, 1:down, 2:left, 3:right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        # Define 4 checkpoint regions (4×4 each): (r_min, r_max, c_min, c_max)
        # Positioned for diagonal crisscross: 0(TL) → 1(BR) → 2(TR) → 3(BL)
        self.checkpoints: List[Tuple[int, int, int, int]] = [
            (2, 5, 2, 5),       # CP0: top-left
            (24, 27, 24, 27),   # CP1: bottom-right (diagonal from CP0)
            (2, 5, 24, 27),     # CP2: top-right
            (24, 27, 2, 5),     # CP3: bottom-left (diagonal from CP2)
        ]
        # Checkpoint centers for potential computation
        self.checkpoint_centers: List[Tuple[int, int]] = [
            (3, 3),   # CP0: top-left
            (25, 25), # CP1: bottom-right
            (3, 25),  # CP2: top-right
            (25, 3),  # CP3: bottom-left
        ]
        self.start_pos: Tuple[int, int] = (15, 15)
        self.correct_order: List[int] = [0, 1, 2, 3]

        # Reward config
        self.step_penalty: float = step_penalty

        # State
        self.agent_pos: List[int] = None
        self.visited_checkpoints: set = None
        self.current_checkpoint_idx: int = None
        self.steps: int = 0
        self.max_steps: int = int(max_steps) if max_steps is not None else 1500
        
        # Walls grid (1 = wall, 0 = free)
        self.walls: np.ndarray = None
        self._generate_maze()

    def _generate_maze(self):
        """Generate random walls ensuring all checkpoints remain reachable."""
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Randomly place walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.random.rand() < self.wall_density:
                    self.walls[i, j] = 1
        
        # Clear start position
        sr, sc = self.start_pos
        self.walls[sr, sc] = 0
        
        # Clear all checkpoint regions
        for (r_min, r_max, c_min, c_max) in self.checkpoints:
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        self.walls[r, c] = 0
        
        # Verify connectivity (BFS from start to all checkpoint centers)
        if not self._verify_connectivity():
            # Retry generation if not connected (rare with density < 0.3)
            self._generate_maze()

    def _verify_connectivity(self) -> bool:
        """Check that all checkpoint centers are reachable from start via BFS."""
        sr, sc = self.start_pos
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        queue = deque([(sr, sc)])
        visited[sr, sc] = True
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and
                    not visited[nr, nc] and self.walls[nr, nc] == 0):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        
        # Check all checkpoint centers are reachable
        for (cr, cc) in self.checkpoint_centers:
            if not visited[cr, cc]:
                return False
        return True

    def _in_checkpoint(self, pos: Tuple[int, int], cp_idx: int) -> bool:
        """Check if position is inside the specified checkpoint."""
        r_min, r_max, c_min, c_max = self.checkpoints[cp_idx]
        r, c = pos
        return r_min <= r <= r_max and c_min <= c <= c_max

    def reset(self, workflow: Optional[List[int]] = None, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            self._generate_maze()  # New maze on seed change
        
        self.agent_pos = list(self.start_pos)
        self.visited_checkpoints = set()
        self.current_checkpoint_idx = 0
        self.steps = 0

        if workflow is not None:
            assert len(workflow) == self.num_checkpoints
            assert set(workflow) == set(range(self.num_checkpoints))
            self.correct_order = list(workflow)
        else:
            self.correct_order = [0, 1, 2, 3, 4, 5]

        return np.array(self.agent_pos, dtype=np.int32)

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self._generate_maze()
        return [seed]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1

        # Move agent if not blocked by wall
        new_pos = list(self.agent_pos)
        if action == 0 and self.agent_pos[0] > 0:  # up
            new_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # down
            new_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            new_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:  # right
            new_pos[1] += 1
        
        # Only move if not hitting a wall
        if self.walls[new_pos[0], new_pos[1]] == 0:
            self.agent_pos = new_pos

        reward = self.step_penalty
        done = False

        # Check checkpoint entry
        current_pos = tuple(self.agent_pos)
        for cp_idx in range(self.num_checkpoints):
            if self._in_checkpoint(current_pos, cp_idx):
                # Reward if it's the required next checkpoint and first entry
                if (
                    self.current_checkpoint_idx < self.num_checkpoints
                    and cp_idx == self.correct_order[self.current_checkpoint_idx]
                    and cp_idx not in self.visited_checkpoints
                ):
                    # Ordinal reward
                    reward += float(self.current_checkpoint_idx + 1)
                    self.visited_checkpoints.add(cp_idx)
                    self.current_checkpoint_idx += 1

                    if self.current_checkpoint_idx >= self.num_checkpoints:
                        done = True
                break

        # Timeout
        if self.steps >= self.max_steps and not done:
            done = True

        info = {
            "visited_checkpoints": sorted(list(self.visited_checkpoints)),
            "current_checkpoint_idx": self.current_checkpoint_idx,
            "correct_order": self.correct_order,
            "agent_pos": self.agent_pos.copy(),
            "checkpoints": self.checkpoints,
        }

        return np.array(self.agent_pos, dtype=np.int32), float(reward), bool(done), info

    def render(self, mode: str = "human"):
        if mode != "human":
            return
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.walls[i, j] == 1:
                    grid[i][j] = "#"
        
        # Mark checkpoints
        for idx, (r_min, r_max, c_min, c_max) in enumerate(self.checkpoints):
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if grid[r][c] != "#":
                        grid[r][c] = str(idx)
        
        # Mark start
        sr, sc = self.start_pos
        if grid[sr][sc] in [str(i) for i in range(self.num_checkpoints)]:
            grid[sr][sc] = "S"
        
        # Mark agent
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"

        print("\n" + "=" * (self.grid_size + 2))
        print(f"Workflow: {self.correct_order}")
        nxt = self.correct_order[self.current_checkpoint_idx] if self.current_checkpoint_idx < len(self.correct_order) else "DONE"
        print(f"Next checkpoint: CP{nxt}")
        print(f"Visited: {sorted(list(self.visited_checkpoints))}")
        print("-" * (self.grid_size + 2))
        for row in grid:
            print("".join(row))
        print("=" * (self.grid_size + 2))

    def get_state_for_policy(self) -> np.ndarray:
        """State = [agent_r, agent_c] normalized + checkpoint centers normalized + visited flags."""
        state = np.zeros(2 + 4 * 2 + 4, dtype=np.float32)
        norm = float(self.grid_size - 1)
        state[0] = self.agent_pos[0] / norm
        state[1] = self.agent_pos[1] / norm
        for i, (r, c) in enumerate(self.checkpoint_centers):
            state[2 + i * 2] = r / norm
            state[2 + i * 2 + 1] = c / norm
        for i in range(4):
            state[2 + 8 + i] = 1.0 if i in self.visited_checkpoints else 0.0
        return state

