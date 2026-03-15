from typing import Any

from pydantic import BaseModel

from karldbot.agents.base import BaseAgent
from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState
from karldbot.models.actions import VisualizerAction

InfoDict = dict[str, Any]


class VisualizationSpec(BaseModel):
    plot_type: str
    title: str
    x_label: str
    y_label: str
    code: str


class Visualizer(BaseAgent[VisualizerAction]):
    def __init__(self, config: AgentConfig | None = None):
        config = config or AgentConfig()
        super().__init__(config, n_actions=2)
        self._actions = {
            VisualizerAction.CREATE_PLOT: self._create_plot,
            VisualizerAction.GENERATE_REPORT: self._generate_report,
        }

    def get_available_actions(self) -> list[VisualizerAction]:
        return list(VisualizerAction)

    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        action_idx = self.select_action(state.score.quality_level)
        action = VisualizerAction(action_idx + 1)
        return await self._actions[action](info)

    async def _create_plot(self, info: InfoDict) -> InfoDict:
        code = info.get("solution", "")
        results = info.get("analysis_results", {})
        prompt = (
            f"Based on the analysis results: {results}\n"
            f"And this code:\n{code}\n"
            f"Generate matplotlib visualization code to illustrate the findings."
        )
        info["viz_prompt"] = prompt
        info["visualization"] = {"status": "pending_implementation"}
        return info

    async def _generate_report(self, info: InfoDict) -> InfoDict:
        analysis = info.get("analysis", {})
        code = info.get("solution", "")
        prompt = (
            f"Create a visual report summarizing:\n"
            f"Analysis: {analysis}\n"
            f"Code solution: {code}"
        )
        info["report_prompt"] = prompt
        info["visual_report"] = {"status": "pending_implementation"}
        return info
