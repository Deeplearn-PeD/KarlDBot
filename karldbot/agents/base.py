from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel

from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState, QualityLevel

ActionType = TypeVar("ActionType")
InfoDict = dict[str, Any]


class BaseAgent(ABC, Generic[ActionType]):
    def __init__(self, config: AgentConfig, n_actions: int, n_states: int = 5):
        self.config = config
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.n_actions = n_actions
        self.n_states = n_states
        self._q_table: np.ndarray | None = None

    @property
    def q_table(self) -> np.ndarray:
        if self._q_table is None:
            self._q_table = np.zeros((self.n_states, self.n_actions))
        return self._q_table

    @q_table.setter
    def q_table(self, value: np.ndarray):
        self._q_table = value

    @abstractmethod
    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        """Execute an action based on the current state."""
        ...

    @abstractmethod
    def get_available_actions(self) -> list[ActionType]:
        """Return list of available actions for this agent."""
        ...

    def select_action(self, quality_level: QualityLevel) -> int:
        """Select action using epsilon-greedy policy."""
        state_idx = int(quality_level)
        if np.random.binomial(1, self.epsilon):
            return np.random.choice(range(self.n_actions))
        q_values = self.q_table[state_idx, :]
        max_indices = np.argwhere(q_values == np.max(q_values)).flatten()
        return int(np.random.choice(max_indices))

    def update_policy(
        self,
        state: QualityLevel,
        action: int,
        reward: float,
        next_state: QualityLevel,
    ) -> None:
        """Update Q-value using expected SARSA algorithm."""
        state_idx = int(state)
        next_state_idx = int(next_state)

        q_next = self.q_table[next_state_idx, :]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()

        target = 0.0
        for action_idx in range(self.n_actions):
            if action_idx in best_actions:
                prob = (1 - self.epsilon) / len(best_actions)
                prob += self.epsilon / self.n_actions
            else:
                prob = self.epsilon / self.n_actions
            target += prob * q_next[action_idx]

        target *= self.gamma
        self.q_table[state_idx, action] += self.learning_rate * (
            reward + target - self.q_table[state_idx, action]
        )


class CodeOutput(BaseModel):
    code: str
    explanation: str


class QualityReport(BaseModel):
    correctness: float
    efficiency: float
    clarity: float
    approved: bool
    recommendations: str
