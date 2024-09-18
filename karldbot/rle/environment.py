"""
Here the Environment class is defined. This class is responsible for
managing the environment of a reinfoircement learning agent.
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
import os
from base_agent.llminterface import LangModel, StructuredLangModel


class Environment:
    def __init__(self, env_name):
        self.ename = env_name
        self.state = None
        self.reward = None
        self.done = None
        self.info = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

class DataScienceProblem:
    def __init__(self, problem_name, data_source):
        """
        Initialize the data science problem with a name and a data source.

        :param problem_name: Name of the data science problem.
        :param data_source: Path or URL to the external data source.
        """
        self.problem_name = problem_name
        self.data_source = data_source
        self.data = None

    def load_data(self):
        """
        Load data from the data source. This method should be implemented
        to handle different types of data sources (e.g., CSV files, databases, APIs).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

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

