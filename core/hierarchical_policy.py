"""
Hierarchical RL Policy - Options Framework (2-Level)
Based on Sutton et al., 1999 Options Framework.

Architecture:
- High-Level Policy: Selects options (temporal abstractions) every ~option_duration steps
- Low-Level Policy: Executes primitive actions using state + option embedding
- Termination: Fixed duration (can be extended to learnable termination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HighLevelPolicy(nn.Module):
    """High-level policy that selects options based on state."""
    def __init__(self, state_dim: int, num_options: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_options = num_options
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.option_head = nn.Linear(64, num_options)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns option logits."""
        h = self.net(state)
        return self.option_head(h)

    def select_option(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select an option from state.
        
        Returns:
            option_idx: Selected option index
            log_prob: Log probability of selected option
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            option_idx = torch.argmax(logits, dim=-1)
            log_prob = dist.log_prob(option_idx)
        else:
            option_idx = dist.sample()
            log_prob = dist.log_prob(option_idx)
        
        return option_idx, log_prob


class LowLevelPolicy(nn.Module):
    """Low-level policy that selects actions given state and option embedding."""
    def __init__(self, state_dim: int, num_options: int = 8, num_actions: int = 4, 
                 hidden_dim: int = 128, option_embed_dim: int = 8):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        self.option_embed_dim = option_embed_dim
        
        # Option embedding (learnable)
        self.option_embedding = nn.Embedding(num_options, option_embed_dim)
        
        # Combined input: state + option embedding
        combined_dim = state_dim + option_embed_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, option: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and value.
        
        Args:
            state: State tensor [batch, state_dim]
            option: Option indices [batch]
        
        Returns:
            action_logits: [batch, num_actions]
            value: [batch]
        """
        # Get option embeddings
        option_emb = self.option_embedding(option)  # [batch, option_embed_dim]
        
        # Concatenate state and option embedding
        combined = torch.cat([state, option_emb], dim=-1)  # [batch, state_dim + option_embed_dim]
        
        h = self.net(combined)
        action_logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        
        return action_logits, value

    def act(self, state: torch.Tensor, option: torch.Tensor, 
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action given state and option.
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        action_logits, value = self.forward(state, option)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value


class HierarchicalPolicy(nn.Module):
    """Combined hierarchical policy (high-level + low-level)."""
    def __init__(self, state_dim: int, num_options: int = 8, num_actions: int = 4,
                 hidden_dim: int = 128, option_embed_dim: int = 8):
        super().__init__()
        self.high_level = HighLevelPolicy(state_dim, num_options, hidden_dim)
        self.low_level = LowLevelPolicy(state_dim, num_options, num_actions, hidden_dim, option_embed_dim)
        self.num_options = num_options
        self.num_actions = num_actions
        self.state_dim = state_dim

    def select_option(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """High-level: select option."""
        return self.high_level.select_option(state, deterministic)

    def select_action(self, state: torch.Tensor, option: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Low-level: select action given option."""
        return self.low_level.act(state, option, deterministic)

    def get_value(self, state: torch.Tensor, option: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state-option pair."""
        _, value = self.low_level.forward(state, option)
        return value

    def forward(self, state: torch.Tensor, option: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: state + option -> action logits + value."""
        return self.low_level.forward(state, option)

