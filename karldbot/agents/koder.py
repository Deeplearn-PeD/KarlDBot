from typing import Any

from loguru import logger

from karldbot.agents.base import BaseAgent, CodeOutput
from karldbot.llm import LLMInterface
from karldbot.llm.prompts import PromptManager
from karldbot.models.config import AgentConfig, LLMConfig
from karldbot.models.state import EnvironmentState
from karldbot.models.actions import CoderAction

InfoDict = dict[str, Any]


class Koder(BaseAgent[CoderAction]):
    def __init__(
        self,
        config: AgentConfig | None = None,
        problem_description: str = "",
        data_source: str = "",
        sample_data: str = "",
    ):
        config = config or AgentConfig()
        super().__init__(config, n_actions=3)
        llm_config = config.llm
        self.llm = LLMInterface(
            model=llm_config.model,
            provider=llm_config.provider,
            retries=llm_config.retries,
        )
        self.prompt_manager = PromptManager(self.llm)
        self.problem_description = problem_description
        self.data_source = data_source
        self.sample_data = sample_data
        self._actions = {
            CoderAction.WRITE_CODE: self._write_code,
            CoderAction.DEBUG_CODE: self._debug_code,
            CoderAction.OPTIMIZE_CODE: self._optimize_code,
        }

    def get_available_actions(self) -> list[CoderAction]:
        return list(CoderAction)

    async def act(self, state: EnvironmentState, info: InfoDict) -> InfoDict:
        action_idx = self.select_action(state.score.quality_level)
        action = CoderAction(action_idx + 1)
        return await self._actions[action](info)

    def set_problem(
        self, description: str, data_source: str, sample_data: str = ""
    ) -> None:
        self.problem_description = description
        self.data_source = data_source
        self.sample_data = sample_data

    async def _write_code(self, info: InfoDict) -> InfoDict:
        if not self.problem_description:
            raise ValueError("Problem description not set")

        step = info.get("step", 0)
        if step == 0:
            prompt = self.prompt_manager.generate_code_writing_prompt(
                self.problem_description, self.data_source, self.sample_data
            )
        else:
            existing = info.get("solution", "")
            prompt = (
                f"Considering the problem description: '{self.problem_description}'\n"
                f"and your existing solution:\n{existing}\n"
                f"Please continue to enhance it."
            )

        info["code_prompt"] = prompt
        try:
            result = await self._get_llm().get_structured_response(
                prompt, CodeOutput, context=""
            )
            info["solution"] = result.code
            info["code_explanation"] = result.explanation
            logger.info(f"Koder: write_code completed")
        except Exception as exc:
            logger.warning(f"Koder error: {exc}")
            info["error"] = str(exc)

        return info

    async def _debug_code(self, info: InfoDict) -> InfoDict:
        code = info.get("solution", "")
        bugs = info.get("bugs", [])
        prompt = self.prompt_manager.generate_code_debugging_prompt(code, bugs)
        info["code_prompt"] = prompt

        try:
            result = await self._get_llm().get_structured_response(
                prompt, CodeOutput, context=""
            )
            info["solution"] = result.code
            info["code_explanation"] = result.explanation
            logger.info(f"Koder: debug_code completed")
        except Exception as exc:
            logger.warning(f"Koder error: {exc}")
            info["error"] = str(exc)

        return info

    async def _optimize_code(self, info: InfoDict) -> InfoDict:
        code = info.get("solution", "")
        recommendations = info.get("recommendations", "")
        prompt = self.prompt_manager.generate_code_optimization_prompt(
            code, recommendations
        )
        info["code_prompt"] = prompt

        try:
            result = await self._get_llm().get_structured_response(
                prompt, CodeOutput, context=""
            )
            info["solution"] = result.code
            info["code_explanation"] = result.explanation
            logger.info(f"Koder: optimize_code completed")
        except Exception as exc:
            logger.warning(f"Koder error: {exc}")
            info["error"] = str(exc)

        return info

    def _get_llm(self):
        from karldbot.llm import AsyncLLMInterface

        return AsyncLLMInterface(
            self.config.llm.model,
            self.config.llm.provider,
            self.config.llm.retries,
        )
