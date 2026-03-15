from enum import Enum, auto
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class CoderAction(Enum):
    WRITE_CODE = auto()
    DEBUG_CODE = auto()
    OPTIMIZE_CODE = auto()


class ReviewerAction(Enum):
    REVIEW_CODE = auto()
    OPTIMIZE_PROMPT = auto()
    APPROVE_CODE = auto()


class AnalystAction(Enum):
    ANALYZE_DATA = auto()
    GENERATE_STATISTICS = auto()
    SUGGEST_APPROACH = auto()


class VisualizerAction(Enum):
    CREATE_PLOT = auto()
    GENERATE_REPORT = auto()


class TesterAction(Enum):
    GENERATE_TESTS = auto()
    RUN_TESTS = auto()


class Action(Protocol):
    def execute(self, info: dict[str, Any]) -> dict[str, Any]: ...


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    action_type: str
    output: dict[str, Any]
    reward_delta: float = 0.0
