from enum import IntEnum, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class WorkflowState(StrEnum):
    INIT = "init"
    CODING = "coding"
    REVIEWING = "reviewing"
    DEBUGGING = "debugging"
    OPTIMIZING = "optimizing"
    TESTING = "testing"
    VISUALIZING = "visualizing"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityLevel(IntEnum):
    POOR = 0
    BASIC = 1
    GOOD = 2
    VERY_GOOD = 3
    EXCELLENT = 4

    @classmethod
    def from_score(cls, avg_score: float) -> "QualityLevel":
        if avg_score <= 3:
            return QualityLevel.POOR
        elif avg_score <= 6:
            return QualityLevel.BASIC
        elif avg_score <= 7.9:
            return QualityLevel.GOOD
        elif avg_score <= 9:
            return QualityLevel.VERY_GOOD
        return QualityLevel.EXCELLENT


class CodeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(default=0.0, ge=0.0, le=10.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=10.0)
    clarity: float = Field(default=0.0, ge=0.0, le=10.0)
    approved: bool = False

    @property
    def average(self) -> float:
        return (self.correctness + self.efficiency + self.clarity) / 3

    @property
    def quality_level(self) -> QualityLevel:
        return QualityLevel.from_score(self.average)

    def is_acceptable(self, threshold: float = 8.0) -> bool:
        return self.average >= threshold and self.approved


class EnvironmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_state: WorkflowState = WorkflowState.INIT
    score: CodeScore = Field(default_factory=CodeScore)
    current_code: str = ""
    bugs: list[str] = Field(default_factory=list)
    recommendations: str = ""
    iteration: int = 0
    done: bool = False
    truncated: bool = False

    def check_completion(self) -> None:
        if self.score.approved:
            self.done = True
            self.workflow_state = WorkflowState.COMPLETED

    def advance_iteration(self) -> None:
        self.iteration += 1

    def reset(self) -> "EnvironmentState":
        return EnvironmentState()
