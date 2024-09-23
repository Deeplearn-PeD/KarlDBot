from base_agent.llminterface import LangModel, StructuredLangModel
from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel
from karldbot.rle.environment import DataScienceProblem
from karldbot.rle.agents import Agent
import numpy as np
from typing import Dict, Any, Tuple
import dotenv
from loguru import logger

dotenv.load_dotenv()
logger.add("karldbot.log", rotation="1 MB", level="WARNING")
logger.add("karldbot_work.log", rotation="1 MB", level="INFO")


class CodeOutput(BaseModel):
    code: str
    explanation: str

class Koder(Agent):
    def __init__(self, language_model='gpt-4o', problem: DataScienceProblem = None):
        """
        Initialize the Koder with a language model and a prompt manager.

        :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
        :param prooblem: An instance of a DataScienceProblem.
        """
        super().__init__()
        self.language_model = StructuredLangModel(language_model, 10)
        self.prompt_manager = PromptManager(LangModel(language_model))
        self.prompt = self.prompt_manager.base_code_prompt
        self.problem = problem
        self.actions =  {0: self.write_code, 1: self.debug_code, 2: self.optimize_code}
        self.n_actions = len(self.actions)
        self.sample_data = problem.sample_data()

    def set_problem(self, problem: DataScienceProblem):
        """
        Set the problem to be solved by the Koder.

        :param problem: The problem to be solved. Is an instance of environment.DataSciencePrloblem.
        """
        self.problem = problem
    def write_code(self, info: Dict[str, Any])-> Dict[str, Any]:
        """
        Write code to accomplish the given task.

        :param info: A dictionary containing the task description.
        :return: The generated code.
        """
        if self.problem is None:
            raise ValueError("No problem set for the Koder.")
        prompt = self.prompt_manager.generate_code_writing_prompt(self.problem.description)
        prompt = f"Considering the following data sample below\n{self.sample_data}\n{prompt}"
        info["code_prompt"] = prompt
        try:
            code = self.language_model.get_response(prompt, context='', response_model=CodeOutput)
            info['solution'] = code.code
            info["code_explanation"] = code.explanation
            logger.info(f"Agent: Koder; Prompt: {prompt}.")
        except InstructorRetryException as exc:
            logger.warning(f"Error: {exc}")
        return info

    def debug_code(self, info):
        """
        Debug the given code snippet.

        :param info: A dictionary containing the code snippet to be debugged.
        :return: The debugged code.
        """
        task_description = f"Debug the following code snippet.\n{info['solution']}"
        prompt = self.prompt_manager.generate_code_writing_prompt(task_description)
        info["code_prompt"] = prompt
        try:
            debugged_code = self.language_model.get_response(prompt,'', response_model=CodeOutput)
            changed = debugged_code.code != info['solution']
            info['solution'] = debugged_code.code
            info["code_explanation"] = debugged_code.explanation
            logger.info(f"Agent: Koder; Prompt: {prompt}; Changed: {changed}")
        except InstructorRetryException as exc:
            logger.warning(f"Error: {exc}")

        return info

    def optimize_code(self, info):
        """
        Optimize the given code snippet for the specified optimization target.

        :param info: A dictionary containing the code snippet to be optimized.
        :return: The optimized code.
        """
        task_description = f"Improve the following code snippet according to these recomendations: '{info["recommendations"]}'.\n{info['solution']}"
        prompt = self.prompt_manager.generate_code_writing_prompt(task_description)
        info["code_prompt"] = prompt
        try:
            optimized_code = self.language_model.get_response(prompt, context='', response_model=CodeOutput)
            changed = optimized_code.code != info['solution']
            info['solution'] = optimized_code.code
            info["code_explanation"] = optimized_code.explanation
            logger.info(f"Agent: Koder; Prompt: {prompt}; Changed: {changed}")
        except InstructorRetryException as exc:
            logger.warning(f"Error: {exc}")

        return info

    def update_policy(self , state, next_state, reward):
        """
        Update q(s,a) value using the expected SARSA algorithm.
        :param state: Tuple (s,a) representing the state and action.
        :return:
        """
        target = 0
        q_next = self.q_value[next_state,:]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        for action_ in range(self.n_actions):  # 3 is the number of actions
            if action_ in best_actions:
                target += (1 - self.epsilon)/len(best_actions) + self.epsilon/3 * q_next[action_]
            else:
                target += self.epsilon/self.n_actions * q_next[action_]
        target *= self.gamma
        self.q_value[state] += self.step_size * (reward + target - self.q_value[state])
        #TODO: update self.policy
        return self.q_value


    def select_action(self, state):
        """
        Select an action based on the epsilon-greedy policy.
        :param state: state of the environment
        :return:
        """
        if np.random.binomial(1, self.epsilon):
            return np.random.choice(range(self.n_actions))
        else:
            values_ = self.q_value[state,:]
            return np.random.choice(np.argwhere(values_ == np.max(values_)).flatten())


class QualityReport(BaseModel):
    correctness: float
    efficiency: float
    clarity: float
    approved: bool
    recommendations: str


class CodeReviewer(Agent):
    def __init__(self, language_model='gpt-4o'):
        """
        Initialize the CodeReviewer with a language model and a prompt manager.

        :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
        :param prompt_manager: An instance of a prompt manager.
        """
        super().__init__()
        self.language_model = StructuredLangModel(language_model)
        self.prompt_manager = PromptManager(LangModel(language_model))
        self.prompt = self.prompt_manager.base_code_review_prompt
        self.score_model = QualityReport
        self.language_model = StructuredLangModel(language_model,10)
        self.prompt_manager = PromptManager(LangModel(language_model))
        self.score_model = QualityReport
        self.actions =  {0: self.review_code, 1: self.optimize_prompt, 2: self.approve_code}
        self.n_actions = len(self.actions)

    def review_code(self, info: Dict[str, Any])-> Dict[str, Any]:
        """
        Review the given code snippet for correctness, efficiency, and style.

        :param info: A dictionary containing the code snippet to be reviewed.
        :return: The review feedback.
        """
        prompt = self.prompt_manager.generate_code_review_prompt(info['solution'])
        prompt += ("\n Please give a numerical grade for correctness(between 0.0 and 10.0 ), efficiency(between 0.0 and 10.0 ) and clarity(between 0.0 and 10.0 ) of the code snippet. "
                   "Don't hesitate to give it a top grade, if you think it deserves it. Only approve the code if you think it solves the problem.")
        info["review_prompt"] = prompt
        try:
            review_feedback = self.language_model.get_response(prompt, context='', response_model=self.score_model)
            info['review'] = review_feedback
            # logger.info(f"Agent: CodeReviewer; Prompt: {prompt}; Review: {review_feedback}")
        except InstructorRetryException as exc:
            logger.warning(f"Error: {exc}")

        return info

    def optimize_prompt(self, info):
        """
        Optimize the given prompt for the specified optimization target.

        :param prompt: The prompt to be optimized.
        :param optimization_target: The target for optimization (e.g., code quality, efficiency).
        :return: An optimized prompt.
        """
        optimized_prompt = self.prompt_manager.optimize_prompt(self, info["recommendations"])
        info['coders_opt_prompt'] = optimized_prompt
        # logger.info(f"Agent: CodeReviewer; Prompt: {info['code_prompt']}; Optimized prompt: {optimized_prompt}")
        return info

    def approve_code(self, info):
        """
        Approve the given code snippet.

        :param info: A dictionary containing the code snippet to be approved.
        :return: The approval feedback.
        """
        info['review'].approved = True
        return info

    def update_policy(self , state: Tuple[int,int], next_state: int, reward: int)-> np.ndarray:
        """
        Update q(s,a) value using the expected SARSA algorithm.
        :param state: Tuple (s,a) representing the state and action.
        :param next_state: int representing the next state.
        :return:
        """
        target = 0
        q_next = self.q_value[next_state,:]
        best_actions = np.argwhere(q_next == np.max(q_next)).flatten()
        for action_ in range(self.n_actions):  # 3 is the number of actions
            if action_ in best_actions:
                target += (1 - self.epsilon)/len(best_actions) + self.epsilon/3 * q_next[action_]
            else:
                target += self.epsilon/self.n_actions * q_next[action_]
        target *= self.gamma
        self.q_value[state] += self.step_size * (reward + target - self.q_value[state])
        #TODO: update self.policy
        return self.q_value


    def select_action(self, state):
        """
        Select an action based on the epsilon-greedy policy.
        :param state: state of the environment
        :return:
        """
        if np.random.binomial(1, self.epsilon):
            return np.random.choice(range(self.n_actions))
        else:
            values_ = self.q_value[state,:]
            return np.random.choice(np.argwhere(values_ == np.max(values_)).flatten())


class PromptManager:
    def __init__(self, language_model):
        """
        Initialize the PromptManager with a language model.

        :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
        """
        self.language_model = language_model
        self.base_code_prompt = "You are an experienced Python coder. Your job is to write correct, efficient, and well-structured code to solve data-science problems.\n"
        self.base_code_review_prompt = "You are a senior data scientist. Your job is to review the code written by a junior data scientist for correctness, efficiency, and style.\n"

    def generate_code_writing_prompt(self, task_description):
        """
        Generate a prompt for code writing based on the task description.

        :param task_description: A description of the coding task.
        :return: A prompt for code writing.
        """
        prompt = self.base_code_prompt + f"Write code to accomplish the following task: {task_description}"
        return prompt

    def generate_code_review_prompt(self, code_snippet):
        """
        Generate a prompt for code review based on the provided code snippet.

        :param code_snippet: A snippet of code to be reviewed.
        :return: A prompt for code review.
        """
        prompt = self.base_code_review_prompt + f"Review the following code for correctness, efficiency, and style: {code_snippet}"
        return prompt

    def optimize_prompt(self, prompt, optimization_target):
        """
        Optimize the given prompt using the language model. for the given optimization target.

        :param prompt: The prompt to be optimized.
         :param optimization_target:  The target for optimization (e.g., code quality, efficiency).
        :return: An optimized prompt.
        """
        prompt = f"Please optimize the prompt below so that the LLM output will be better with respect to {optimization_target}:\n'{prompt}'"
        optimized_prompt = self.language_model.get_response(prompt, context='')
        return optimized_prompt
