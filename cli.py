"""
This is the entry point for the CLI application.

"""
import fire
import yaml
from karldbot.rle.environment import Environment, DataScienceProblem
from karldbot.brain import Koder, CodeReviewer
from karldbot.brain.report import Report

class KarlInterface:
    def __init__(self, config_file: str='config.yaml'):
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.llm_model = self.config['llm_model']
        self.data_source = self.config['data_source']

    def train(self):
        problem_name = self.config['problem_name']
        data_source = self.config['data_source']
        description = self.config.get('description', '')

        problem = DataScienceProblem(problem_name, data_source)
        problem.set_description(description)
        env = Environment('test_env', problem)
        coder = Koder(self.llm_model)
        coder.n_actions = len(env.coder_action_space)
        reviewer = CodeReviewer()
        state, reward, done, _ , info=env.reset()
        rewards = []
        it = 0
        print(f"Training model {self.model_name} on data source {self.data_source}...")
        while not done:
            action = env.action_sample()
            state, reward, done, truncated, info = env.step(action, info={'solution': coder.generate_code()}, agent='coder')
            review_action = reviewer.review_code(state, info['solution'])
            state, reward_rev, done, truncated, info = env.step(review_action, info=info, agent='reviewer')
            rewards.append(reward)


            it += 1




    def view_report(self):
        pass


def main():
    fire.Fire(KarlInterface)