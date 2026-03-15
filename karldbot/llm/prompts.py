from typing import Any

from karldbot.llm import LLMInterface
from karldbot.models.config import AgentConfig
from karldbot.models.state import EnvironmentState, QualityLevel
from karldbot.agents.base import BaseAgent, CodeOutput


class PromptManager:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.base_code_prompt = (
            "You are an experienced Python coder. "
            "Your job is to write correct, efficient, and well-structured code "
            "to solve data-science problems.\n"
        )
        self.base_code_review_prompt = (
            "You are a senior data scientist. "
            "Your job is to review the code written by a junior data scientist "
            "for correctness, efficiency, and style.\n"
        )

    def generate_code_writing_prompt(
        self, problem_description: str, data_source: str, sample_data: str = ""
    ) -> str:
        problem_definition = (
            f"You have to write a Python code to solve the following "
            f"datascience problem. The data source is: {data_source}. "
            f"The problem definition is: {problem_description}"
        )
        prompt = self.base_code_prompt + "\n" + problem_definition
        if sample_data:
            prompt += f"\nHere is a sample of the data:\n{sample_data}"
        return prompt

    def generate_code_debugging_prompt(self, code: str, bugs: list[str]) -> str:
        buglist = "\n".join(bugs)
        return (
            self.base_code_prompt
            + f"Debug the following code snippet: {code}, "
            + f"and fix the following bugs: {buglist}"
        )

    def generate_code_optimization_prompt(self, code: str, recommendations: str) -> str:
        return (
            f"Improve the following code according to the recommendations:\n"
            f"Recommendations: '{recommendations}'.\n"
            f"Code:\n{code}"
        )

    def generate_code_review_prompt(self, code: str) -> str:
        return (
            self.base_code_review_prompt
            + f"Review the following code for correctness, efficiency, and style: {code}"
        )

    def optimize_prompt(self, prompt: str, optimization_target: str) -> str:
        opt_prompt = (
            f"Please optimize the prompt below so that the LLM output will be "
            f"better with respect to {optimization_target}:\n'{prompt}'"
        )
        return self.llm.get_response(opt_prompt, context="")
