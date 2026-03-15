import pytest

from karldbot.environment.core import Environment
from karldbot.environment.problem import DataScienceProblem
from karldbot.models.config import ProblemConfig
from karldbot.models.state import WorkflowState


@pytest.fixture
def problem_config():
    return ProblemConfig(
        name="Test Problem",
        data_source="examples/data.csv.gz",
        description="A test problem for unit testing",
        max_iterations=10,
    )


@pytest.fixture
def problem(problem_config):
    return DataScienceProblem(problem_config)


class TestDataScienceProblem:
    def test_problem_initialization(self, problem):
        assert problem.problem_name == "Test Problem"
        assert problem.description == "A test problem for unit testing"
        assert problem.data_loaded is True

    def test_sample_data(self, problem):
        sample = problem.sample_data(n=5)
        assert isinstance(sample, str)
        assert len(sample) > 0

    def test_get_schema(self, problem):
        schema = problem.get_schema()
        assert isinstance(schema, str)


class TestEnvironment:
    def test_environment_initialization(self, problem):
        env = Environment(problem, max_iterations=10)
        assert env.max_iterations == 10
        assert env.state.iteration == 0
        assert env.state.done is False

    def test_environment_reset(self, problem):
        env = Environment(problem)
        env.state.iteration = 5
        env.state.done = True

        state, info = env.reset()
        assert state.iteration == 0
        assert state.done is False
        assert "step" in info
        assert "solution" in info

    def test_step_coder(self, problem):
        env = Environment(problem)
        state, info = env.reset()

        info["solution"] = "print('test')"
        state, reward, done, info = env.step_coder(info)

        assert state.workflow_state == WorkflowState.CODING
        assert state.current_code == "print('test')"
        assert state.iteration == 1

    def test_step_reviewer_with_review(self, problem):
        env = Environment(problem)
        state, info = env.reset()

        info["solution"] = "print('test')"
        info["review"] = {
            "correctness": 8.0,
            "efficiency": 7.0,
            "clarity": 9.0,
            "approved": False,
            "recommendations": "Good code",
        }

        state, reward, done, info = env.step_reviewer(info)

        assert state.score.correctness == 8.0
        assert state.score.efficiency == 7.0
        assert state.score.clarity == 9.0
        assert state.score.approved is False

    def test_step_reviewer_approved(self, problem):
        env = Environment(problem)
        state, info = env.reset()

        info["solution"] = "print('test')"
        info["review"] = {
            "correctness": 9.0,
            "efficiency": 9.0,
            "clarity": 9.0,
            "approved": True,
            "recommendations": "Excellent!",
        }

        state, reward, done, info = env.step_reviewer(info)

        assert state.score.approved is True
        assert state.done is True
        assert state.workflow_state == WorkflowState.COMPLETED
