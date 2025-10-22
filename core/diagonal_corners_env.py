"""
Diagonal Corners Environment - Targets at (0,0), (10,10), (10,0), (0,10)
Start at center (5,5) on an 11x11 grid so coordinates are valid.
Reward: small step penalty; give reward only when i'th visit matches i'th desired order.
"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Optional, Dict


class DiagonalCornersEnv(gym.Env):
    """
    Environment with 4 targets at outer corners requiring diagonal traversals in order.
    Grid: 11x11 (indices 0..10). Start at center (5,5).
    Order: T0=(0,0) -> T1=(10,10) -> T2=(10,0) -> T3=(0,10)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, step_penalty: float = -0.01, seed: Optional[int] = None, max_steps: Optional[int] = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        # Fixed grid size to match requested coordinates
        self.grid_size = 11  # indices 0..10 inclusive
        self.num_targets = 4

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0:up, 1:down, 2:left, 3:right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        # Fixed layout
        self.target_positions: List[Tuple[int, int]] = [
            (0, 0),              # T0
            (10, 10),            # T1
            (10, 0),             # T2
            (0, 10)              # T3
        ]
        self.start_pos: Tuple[int, int] = (5, 5)
        self.correct_order: List[int] = [0, 1, 2, 3]

        # Reward config
        self.step_penalty: float = step_penalty
        # Ordinal rewards: reward i when visiting i'th correct target (1-indexed)
        self.first_visit_reward: float = 0.0  # unused in ordinal scheme
        self.completion_bonus: float = 0.0

        # State
        self.agent_pos: List[int] = None
        self.visited_targets: set = None
        self.current_target_idx: int = None
        self.steps: int = 0
        # Allow overriding episode horizon
        self.max_steps: int = int(max_steps) if max_steps is not None else self.grid_size * self.grid_size * 2

    def reset(self, workflow: Optional[List[int]] = None, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = list(self.start_pos)
        self.visited_targets = set()
        self.current_target_idx = 0
        self.steps = 0

        # Enforce the specified order by default; allow override
        if workflow is not None:
            assert len(workflow) == self.num_targets
            assert set(workflow) == set(range(self.num_targets))
            self.correct_order = list(workflow)
        else:
            self.correct_order = [0, 1, 2, 3]

        return np.array(self.agent_pos, dtype=np.int32)

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

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

        # Check target visit
        current_pos = tuple(self.agent_pos)
        for target_idx, target_pos in enumerate(self.target_positions):
            if current_pos == target_pos:
                # Only reward if it's exactly the required next target and first time
                if (
                    self.current_target_idx < self.num_targets
                    and target_idx == self.correct_order[self.current_target_idx]
                    and target_idx not in self.visited_targets
                ):
                    # Ordinal reward: + (current_target_idx + 1)
                    reward += float(self.current_target_idx + 1)
                    self.visited_targets.add(target_idx)
                    self.current_target_idx += 1

                    # Completed all targets in order
                    if self.current_target_idx >= self.num_targets:
                        done = True
                # Else: no extra reward (only step penalty applies)
                break

        # Timeout
        if self.steps >= self.max_steps and not done:
            done = True

        info = {
            "visited_targets": sorted(list(self.visited_targets)),
            "current_target_idx": self.current_target_idx,
            "correct_order": self.correct_order,
            "agent_pos": self.agent_pos.copy(),
            "target_positions": self.target_positions,
        }

        return np.array(self.agent_pos, dtype=np.int32), float(reward), bool(done), info

    def render(self, mode: str = "human"):
        if mode != "human":
            return
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for idx, (r, c) in enumerate(self.target_positions):
            grid[r][c] = str(idx)
        # Mark start
        sr, sc = self.start_pos
        if grid[sr][sc] == ".":
            grid[sr][sc] = "S"
        # Mark agent
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"

        print("\n" + "=" * (self.grid_size * 2 + 1))
        print(f"Workflow: {self.correct_order}")
        nxt = self.correct_order[self.current_target_idx] if self.current_target_idx < len(self.correct_order) else "DONE"
        print(f"Next target: T{nxt}")
        print(f"Visited: {sorted(list(self.visited_targets))}")
        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print("=" * (self.grid_size * 2 + 1))

    def get_state_for_policy(self) -> np.ndarray:
        """State = [agent_r, agent_c] normalized + flattened target coords normalized + visited flags."""
        state = np.zeros(2 + self.num_targets * 2 + self.num_targets, dtype=np.float32)
        norm = float(self.grid_size - 1)
        state[0] = self.agent_pos[0] / norm
        state[1] = self.agent_pos[1] / norm
        for i, (r, c) in enumerate(self.target_positions):
            state[2 + i * 2] = r / norm
            state[2 + i * 2 + 1] = c / norm
        for i in range(self.num_targets):
            state[2 + self.num_targets * 2 + i] = 1.0 if i in self.visited_targets else 0.0
        return state


