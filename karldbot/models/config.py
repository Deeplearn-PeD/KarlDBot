from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "gpt-4o"
    provider: Literal["openai", "ollama", "anthropic"] = "openai"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    retries: int = Field(default=3, ge=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ProblemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_source: Path | str
    description: str
    llm_model: str = "gpt-4o"
    max_iterations: int = Field(default=50, ge=1)
    target_score: float = Field(default=8.0, ge=0.0, le=10.0)

    @field_validator("data_source", mode="before")
    @classmethod
    def validate_data_source(cls, v: str | Path) -> str:
        if isinstance(v, Path):
            return str(v)
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProblemConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data.get("problem_name", "Unnamed Problem"),
            data_source=data.get("data_source", ""),
            description=data.get("description", ""),
            llm_model=data.get("llm_model", "gpt-4o"),
            max_iterations=data.get("max_iterations", 50),
        )
