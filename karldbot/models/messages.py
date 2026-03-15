from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender: str
    receiver: str
    content: str | dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    action_taken: str
    success: bool = True
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
