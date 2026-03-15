from typing import Any

from pydantic import BaseModel

from karldbot.agents.base import BaseAgent
from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState
from karldbot.models.actions import TesterAction

InfoDict = dict[str, Any]


class TestResult(BaseModel):
    test_code: str
    passed: bool
    coverage: float
    failures: list[str]


class Tester(BaseAgent[TesterAction]):
    def __init__(self, config: AgentConfig | None = None):
        config = config or AgentConfig()
        super().__init__(config, n_actions=2)
        self._actions = {
            TesterAction.GENERATE_TESTS: self._generate_tests,
            TesterAction.RUN_TESTS: self._run_tests,
        }

    def get_available_actions(self) -> list[TesterAction]:
        return list(TesterAction)

    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        action_idx = self.select_action(state.score.quality_level)
        action = TesterAction(action_idx + 1)
        return await self._actions[action](info)

    async def _generate_tests(self, info: InfoDict) -> InfoDict:
        code = info.get("solution", "")
        problem = info.get("problem_description", "")
        prompt = (
            f"Generate comprehensive pytest unit tests for this code:\n{code}\n"
            f"The code solves this problem: {problem}"
        )
        info["test_generation_prompt"] = prompt
        info["test_code"] = {"status": "pending_implementation"}
        return info

    async def _run_tests(self, info: InfoDict) -> InfoDict:
        test_code = info.get("test_code", "")
        code = info.get("solution", "")
        prompt = (
            f"Run the following tests against this code and report results:\n"
            f"Tests:\n{test_code}\n"
            f"Code:\n{code}"
        )
        info["test_run_prompt"] = prompt
        info["test_results"] = {"status": "pending_implementation"}
        return info
