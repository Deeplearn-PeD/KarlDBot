import pytest

from karldbot.agents.base import BaseAgent, AgentConfig
from karldbot.models.state import EnvironmentState, QualityLevel


class MockAgent(BaseAgent):
    def __init__(self, config=None):
        config = config or AgentConfig()
        super().__init__(config, n_actions=3, n_states=5)

    async def act(self, state, info):
        return info

    def get_available_actions(self):
        return [0, 1, 2]


class TestBaseAgent:
    def test_agent_initialization(self):
        agent = MockAgent()
        assert agent.epsilon == 0.1
        assert agent.gamma == 0.99
        assert agent.learning_rate == 0.5
        assert agent.n_actions == 3

    def test_q_table_lazy_initialization(self):
        agent = MockAgent()
        assert agent._q_table is None
        q_table = agent.q_table
        assert q_table is not None
        assert q_table.shape == (5, 3)

    def test_select_action_random(self):
        agent = MockAgent()
        agent.epsilon = 1.0
        actions = [agent.select_action(QualityLevel.POOR) for _ in range(100)]
        assert set(actions).issubset({0, 1, 2})

    def test_select_action_greedy(self):
        agent = MockAgent()
        agent.epsilon = 0.0
        agent.q_table[0, 2] = 10.0
        action = agent.select_action(QualityLevel.POOR)
        assert action == 2

    def test_update_policy(self):
        agent = MockAgent()
        initial_value = agent.q_table[0, 0]
        agent.update_policy(
            state=QualityLevel.POOR, action=0, reward=10.0, next_state=QualityLevel.GOOD
        )
        assert agent.q_table[0, 0] != initial_value
