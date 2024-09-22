"""
Here the Environment class is defined. This class is responsible for
managing the environment of a reinforcement learning agent.
The environment is the world in which the agent interacts with.
The environment is responsible for providing the agent with the
state of the world, the reward for the action taken by the agent,
and the next state of the world after the agent has taken an action.

This environment is specialized in writing code, with the help of
an LLM to solve a specific data science problem. In order to do this it needs
to be able to take in the code written by the agent, evaluate its quality,
and provide a reward to the agent based on the quality of the code.

The reward will also depend on if the code is correct or not, and how well
it performs on the data science problem.
"""
import duckdb
import random
from typing import Dict, Any, Tuple


class DataScienceProblem:
    def __init__(self, problem_name, data_source):
        """
        Initialize the data science problem with a name and a data source.

        :param problem_name: Name of the data science problem.
        :param data_source: Path or URL to the external data source.
        """
        self.problem_name = problem_name
        self.data_source = data_source
        self.data_table = 'data'
        self.data_loaded = False
        self.description = None
        self.connection = None

    def set_description(self, description: str):
        """
        Set the description of the data science problem.

        :param description: Description of the data science problem.
        """
        self.description = description

    def load_data(self):
        """
        Load data from the data source. This method should be implemented
        to handle different types of data sources (e.g., CSV files, databases, APIs).
        """
        self.connection = duckdb.connect(":memory:")
        self.connection.execute(
            f"CREATE TABLE {self.data_table} AS SELECT * FROM read_csv_auto('{self.data_source}')")
        self.data_loaded = True

    def sample_data(self):
        """
        Sample data from the loaded data. This method should be implemented
        to handle different types of data sampling strategies.
        """
        sample = self.connection.execute(f"SELECT * FROM {self.data_table} LIMIT 100").fetchdf()
        return sample.to_dict()

    def preprocess_data(self):
        """
        Preprocess the loaded data. This method should be implemented
        to handle specific preprocessing steps required for the problem.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate_solution(self, solution):
        """
        Evaluate the provided solution. This method should be implemented
        to handle the evaluation logic specific to the problem.

        :param solution: The solution to be evaluated.
        :return: Evaluation metrics or score.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class Environment:
    def __init__(self, env_name: str, problem: DataScienceProblem):
        self.ename = env_name
        self.problem = problem
        self.coder_action_space = ["write_code", "debug_code", "optimize_prompt"]
        self.reviewer_action_space = ["review_code", "optimize_prompt", "approve_code"]
        self.observation_space = ["code_correctness", "code_efficiency", "code_style", "approved"]
        self.score = {"code_correctness": 0, "code_efficiency": 0, "code_style": 0, "approved": False}
        self.state = 0
        self.reward = 0
        self.done = False
        self.truncated = False
        self.t = 0
        self.info = {"recommendations": "", "solution": ""}

    def step(self, action: str, info: Dict[str, Any], agent: str = 'coder') -> tuple:
        """
        Execute an action in the environment.
        :param self:
        :param action: action to be executed
        :param info: information about the environment
        :param agent:  agent that is executing the action: "coder" or "reviewer"
        :return:
        """
        if agent == 'coder':
            self.score["aproved"] = False
            self.info["solution"] = info['solution']
            self.reward = 0
        elif agent == 'reviewer':
            self.score["code_correctness"] = info['review'].correctness
            self.score["code_efficiency"] = info['review'].efficiency
            self.score["code_style"] = info['review'].style
            self.info["recommendations"] = info['review'].recommendations
            self.calc_reward()
            self.score["approved"] = info['review'].approved and self.state == 3
            if self.score["approved"]:
                self.done = True
            else:

                self.t += 1  # increment time step only after reviewer has reviewed the code
        return self.state, self.reward, self.done, self.truncated, self.info

    def calc_reward(self):
        """
        Calculate the reward based on the quality of the code.
        """
        avg_score = (self.score["code_correctness"] + self.score["code_efficiency"] + self.score["code_style"]) / 3
        if avg_score <=3:
            self.state = 0
        elif avg_score <= 6:
            self.state = 1
        elif avg_score <= 9:
            self.state = 2
        else:
            self.state = 3
        self.reward = 10 if self.score["aproved"] else 0

    def action_sample(self) -> Tuple[int, int]:
        coder_action = 0 if self.t == 0 else random.sample(range(len(self.coder_action_space)), 1)[0]
        reviewer_action = 0 if self.t == 0 else random.sample(range(len(self.reviewer_action_space)), 1)[0]
        return coder_action, reviewer_action

    def reset(self):
        self.score = {"code_correctness": 0, "code_efficiency": 0, "code_style": 0, "aproved": False}
        self.reward = 0
        self.done = 0
        self.truncated = False
        self.info = {"recommendations": "", "solution": ""}
        return self.state, self.reward, self.done, self.truncated, self.info

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
