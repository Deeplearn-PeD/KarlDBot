import codeop
from typing import Any

from loguru import logger

from karldbot.environment.problem import DataScienceProblem
from karldbot.models.state import (
    CodeScore,
    EnvironmentState,
    QualityLevel,
    WorkflowState,
)

InfoDict = dict[str, Any]


class Environment:
    def __init__(self, problem: DataScienceProblem, max_iterations: int = 50):
        self.problem = problem
        self.max_iterations = max_iterations
        self.state = EnvironmentState()
        self.reward = 0.0

    def reset(self) -> tuple[EnvironmentState, InfoDict]:
        self.state = EnvironmentState()
        self.reward = 0.0
        return self.state, {
            "step": 0,
            "solution": "",
            "recommendations": "",
            "bugs": [],
        }

    def step_coder(
        self, info: InfoDict
    ) -> tuple[EnvironmentState, float, bool, InfoDict]:
        self.state.workflow_state = WorkflowState.CODING
        self.state.current_code = info.get("solution", "")
        self.state.advance_iteration()
        info["step"] = self.state.iteration
        return self.state, 0.0, False, info

    def step_reviewer(
        self, info: InfoDict
    ) -> tuple[EnvironmentState, float, bool, InfoDict]:
        self.state.workflow_state = WorkflowState.REVIEWING

        if "review" not in info:
            return self.state, self.reward, False, info

        review = info["review"]
        if hasattr(review, "model_dump"):
            review_dict = review.model_dump()
        elif hasattr(review, "dict"):
            review_dict = review.dict()
        else:
            review_dict = review

        self.state.score = CodeScore(
            correctness=review_dict.get("correctness", 0.0),
            efficiency=review_dict.get("efficiency", 0.0),
            clarity=review_dict.get("clarity", 0.0),
            approved=review_dict.get("approved", False),
        )
        self.state.recommendations = review_dict.get("recommendations", "")

        self._calculate_reward(info)
        self._check_syntax(info)

        if self.state.score.approved:
            self.state.done = True
            self.state.workflow_state = WorkflowState.COMPLETED

        if self.state.iteration >= self.max_iterations:
            self.state.truncated = True

        return self.state, self.reward, self.state.done, info

    def _calculate_reward(self, info: InfoDict) -> None:
        avg_score = self.state.score.average
        self.reward = 100.0 if self.state.score.approved else 0.0
        self.reward += avg_score

    def _check_syntax(self, info: InfoDict) -> None:
        code = info.get("solution", "")
        bugs: list[str] = []
        for line in code.split("\n"):
            try:
                codeop.compile_command(line)
            except SyntaxError:
                logger.info(f"Syntax error in line: {line}")
                bugs.append(line)
                self.reward -= 5

        info["bugs"] = bugs
        self.state.bugs = bugs

    def get_info(self) -> InfoDict:
        return {
            "step": self.state.iteration,
            "solution": self.state.current_code,
            "recommendations": self.state.recommendations,
            "bugs": self.state.bugs,
            "score": self.state.score.model_dump(),
        }
