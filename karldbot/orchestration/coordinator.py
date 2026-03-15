from typing import Any

from karldbot.agents.base import BaseAgent
from karldbot.environment import DataScienceProblem, Environment
from karldbot.models.state import EnvironmentState, WorkflowState

InfoDict = dict[str, Any]


class WorkflowStateMachine:
    TRANSITIONS: dict[WorkflowState, list[WorkflowState]] = {
        WorkflowState.INIT: [WorkflowState.CODING],
        WorkflowState.CODING: [WorkflowState.REVIEWING],
        WorkflowState.REVIEWING: [
            WorkflowState.DEBUGGING,
            WorkflowState.OPTIMIZING,
            WorkflowState.COMPLETED,
        ],
        WorkflowState.DEBUGGING: [WorkflowState.CODING],
        WorkflowState.OPTIMIZING: [WorkflowState.CODING],
        WorkflowState.TESTING: [WorkflowState.CODING, WorkflowState.COMPLETED],
        WorkflowState.VISUALIZING: [WorkflowState.COMPLETED],
        WorkflowState.COMPLETED: [],
        WorkflowState.FAILED: [],
    }

    def __init__(self):
        self.current_state = WorkflowState.INIT
        self.history: list[WorkflowState] = [self.current_state]

    def can_transition_to(self, target: WorkflowState) -> bool:
        return target in self.TRANSITIONS.get(self.current_state, [])

    def transition(self, target: WorkflowState) -> bool:
        if self.can_transition_to(target):
            self.current_state = target
            self.history.append(target)
            return True
        return False

    def reset(self) -> None:
        self.current_state = WorkflowState.INIT
        self.history = [self.current_state]


class AgentCoordinator:
    def __init__(
        self,
        agents: dict[str, BaseAgent],
        problem: DataScienceProblem,
        max_iterations: int = 50,
    ):
        self.agents = agents
        self.problem = problem
        self.env = Environment(problem, max_iterations)
        self.workflow = WorkflowStateMachine()
        self.episode_history: list[InfoDict] = []

    async def run_episode(self) -> tuple[EnvironmentState, list[InfoDict]]:
        state, info = self.env.reset()
        self.episode_history = []

        while not state.done and not state.truncated:
            info = await self._run_step(state, info)
            self.episode_history.append(info.copy())
            state = self.env.state

        return state, self.episode_history

    async def _run_step(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        coder = self.agents.get("coder")
        reviewer = self.agents.get("reviewer")

        if coder is None or reviewer is None:
            raise ValueError("Both coder and reviewer agents are required")

        info = await coder.act(state, info)
        state, reward, done, info = self.env.step_coder(info)

        info = await reviewer.act(state, info)
        state, reward, done, info = self.env.step_reviewer(info)

        coder.update_policy(
            state.score.quality_level,
            0,
            reward,
            state.score.quality_level,
        )
        reviewer.update_policy(
            state.score.quality_level,
            0,
            reward,
            state.score.quality_level,
        )

        return info

    def get_agent_q_tables(self) -> dict[str, Any]:
        return {
            name: agent.q_table.tolist() if hasattr(agent, "q_table") else None
            for name, agent in self.agents.items()
        }
