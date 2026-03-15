from typing import Any

from loguru import logger

from karldbot.agents.base import BaseAgent, QualityReport
from karldbot.llm import LLMInterface
from karldbot.llm.prompts import PromptManager
from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState
from karldbot.models.actions import ReviewerAction

InfoDict = dict[str, Any]


class CodeReviewer(BaseAgent[ReviewerAction]):
    def __init__(self, config: AgentConfig | None = None):
        config = config or AgentConfig()
        super().__init__(config, n_actions=3)
        llm_config = config.llm
        self.llm = LLMInterface(
            model=llm_config.model,
            provider=llm_config.provider,
            retries=llm_config.retries,
        )
        self.prompt_manager = PromptManager(self.llm)
        self._actions = {
            ReviewerAction.REVIEW_CODE: self._review_code,
            ReviewerAction.OPTIMIZE_PROMPT: self._optimize_prompt,
            ReviewerAction.APPROVE_CODE: self._approve_code,
        }

    def get_available_actions(self) -> list[ReviewerAction]:
        return list(ReviewerAction)

    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        action_idx = self.select_action(state.score.quality_level)
        action = ReviewerAction(action_idx + 1)
        return await self._actions[action](info)

    async def _review_code(self, info: InfoDict) -> InfoDict:
        code = info.get("solution", "")
        prompt = self.prompt_manager.generate_code_review_prompt(code)
        prompt += (
            "\nPlease give a numerical grade for correctness (between 0.0 and 10.0), "
            "efficiency (between 0.0 and 10.0) and clarity (between 0.0 and 10.0) "
            "of the code snippet. Don't hesitate to give it a top grade if you think "
            "it deserves it. Only approve the code if you think it solves the problem."
        )

        info["review_prompt"] = prompt

        try:
            result = await self._get_llm().get_structured_response(
                prompt, QualityReport, context=""
            )
            info["review"] = result
            logger.info(
                f"CodeReviewer: review completed - scores: "
                f"{result.correctness}/{result.efficiency}/{result.clarity}"
            )
        except Exception as exc:
            logger.warning(f"CodeReviewer error: {exc}")
            info["error"] = str(exc)

        return info

    async def _optimize_prompt(self, info: InfoDict) -> InfoDict:
        recommendations = info.get("recommendations", "")
        current_prompt = info.get("code_prompt", "")
        optimized = self.prompt_manager.optimize_prompt(current_prompt, recommendations)
        info["coders_opt_prompt"] = optimized
        logger.info("CodeReviewer: prompt optimized")
        return info

    async def _approve_code(self, info: InfoDict) -> InfoDict:
        if "review" in info and hasattr(info["review"], "approved"):
            info["review"].approved = True
        logger.info("CodeReviewer: code approved")
        return info

    def _get_llm(self):
        from karldbot.llm import AsyncLLMInterface

        return AsyncLLMInterface(
            self.config.llm.model,
            self.config.llm.provider,
            self.config.llm.retries,
        )
