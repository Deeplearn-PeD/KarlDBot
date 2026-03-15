from typing import Any

from pydantic import BaseModel

from karldbot.agents.base import BaseAgent
from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState
from karldbot.models.actions import AnalystAction

InfoDict = dict[str, Any]


class AnalysisResult(BaseModel):
    summary: str
    statistics: dict[str, float]
    insights: list[str]
    suggested_approach: str


class DataAnalyst(BaseAgent[AnalystAction]):
    def __init__(self, config: AgentConfig | None = None):
        config = config or AgentConfig()
        super().__init__(config, n_actions=3)
        self._actions = {
            AnalystAction.ANALYZE_DATA: self._analyze_data,
            AnalystAction.GENERATE_STATISTICS: self._generate_statistics,
            AnalystAction.SUGGEST_APPROACH: self._suggest_approach,
        }

    def get_available_actions(self) -> list[AnalystAction]:
        return list(AnalystAction)

    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        action_idx = self.select_action(state.score.quality_level)
        action = AnalystAction(action_idx + 1)
        return await self._actions[action](info)

    async def _analyze_data(self, info: InfoDict) -> InfoDict:
        sample_data = info.get("sample_data", "")
        problem = info.get("problem_description", "")
        prompt = (
            f"Analyze the following data sample for the problem: {problem}\n"
            f"Data:\n{sample_data}\n"
            f"Provide a comprehensive analysis."
        )
        info["analysis_prompt"] = prompt
        info["analysis"] = {"status": "pending_implementation"}
        return info

    async def _generate_statistics(self, info: InfoDict) -> InfoDict:
        sample_data = info.get("sample_data", "")
        prompt = (
            f"Generate descriptive statistics for the following data:\n{sample_data}"
        )
        info["stats_prompt"] = prompt
        info["statistics"] = {"status": "pending_implementation"}
        return info

    async def _suggest_approach(self, info: InfoDict) -> InfoDict:
        problem = info.get("problem_description", "")
        sample_data = info.get("sample_data", "")
        prompt = (
            f"Based on this problem: {problem}\n"
            f"And this data sample:\n{sample_data}\n"
            f"Suggest the best analytical approach."
        )
        info["approach_prompt"] = prompt
        info["suggested_approach"] = {"status": "pending_implementation"}
        return info
