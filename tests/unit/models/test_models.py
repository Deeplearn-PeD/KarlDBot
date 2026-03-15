import pytest

from karldbot.models.config import AgentConfig, LLMConfig, ProblemConfig
from karldbot.models.state import (
    CodeScore,
    EnvironmentState,
    QualityLevel,
    WorkflowState,
)
from karldbot.agents.base import BaseAgent, CodeOutput, QualityReport


class TestConfig:
    def test_llm_config_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-4o"
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.retries == 3

    def test_agent_config_defaults(self):
        config = AgentConfig()
        assert config.epsilon == 0.1
        assert config.gamma == 0.99
        assert config.learning_rate == 0.5

    def test_problem_config_from_yaml(self, tmp_path):
        yaml_content = """
problem_name: Test Problem
data_source: data.csv
description: A test problem
llm_model: gpt-4o
max_iterations: 30
"""
        yaml_file = tmp_path / "problem.yaml"
        yaml_file.write_text(yaml_content)

        config = ProblemConfig.from_yaml(yaml_file)
        assert config.name == "Test Problem"
        assert config.data_source == "data.csv"
        assert config.description == "A test problem"
        assert config.llm_model == "gpt-4o"
        assert config.max_iterations == 30


class TestState:
    def test_code_score_defaults(self):
        score = CodeScore()
        assert score.correctness == 0.0
        assert score.efficiency == 0.0
        assert score.clarity == 0.0
        assert score.approved is False

    def test_code_score_average(self):
        score = CodeScore(correctness=8.0, efficiency=6.0, clarity=7.0)
        assert score.average == pytest.approx(7.0)

    def test_code_score_quality_level(self):
        assert QualityLevel.from_score(2.0) == QualityLevel.POOR
        assert QualityLevel.from_score(4.0) == QualityLevel.BASIC
        assert QualityLevel.from_score(7.0) == QualityLevel.GOOD
        assert QualityLevel.from_score(8.5) == QualityLevel.VERY_GOOD
        assert QualityLevel.from_score(9.5) == QualityLevel.EXCELLENT

    def test_environment_state_defaults(self):
        state = EnvironmentState()
        assert state.workflow_state == WorkflowState.INIT
        assert state.iteration == 0
        assert state.done is False
        assert state.truncated is False

    def test_environment_state_reset(self):
        state = EnvironmentState(iteration=10, done=True, current_code="some code")
        new_state = state.reset()
        assert new_state.iteration == 0
        assert new_state.done is False
        assert new_state.current_code == ""


class TestModels:
    def test_code_output(self):
        output = CodeOutput(code="print('hello')", explanation="A simple print")
        assert output.code == "print('hello')"
        assert output.explanation == "A simple print"

    def test_quality_report(self):
        report = QualityReport(
            correctness=8.0,
            efficiency=7.0,
            clarity=9.0,
            approved=True,
            recommendations="Good work!",
        )
        assert report.correctness == 8.0
        assert report.approved is True
        assert report.recommendations == "Good work!"
