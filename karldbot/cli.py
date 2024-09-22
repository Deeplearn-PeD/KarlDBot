"""
This is the entry point for the CLI application.

"""
import fire
import yaml
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from karldbot.rle.environment import Environment, DataScienceProblem
from karldbot.brain import Koder, CodeReviewer


class KarlInterface:
    def __init__(self, config_file: str = 'config.yaml'):
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
        coder = Koder(self.llm_model, problem)
        coder.n_actions = len(env.coder_action_space)
        reviewer = CodeReviewer()
        # initialize the environment and agents
        state, reward, done, _, info = env.reset()
        q_value = np.zeros((len(env.observation_space), coder.n_actions))
        coder.q_value = q_value
        reviewer.q_value = q_value
        rewards = []
        evolution = [state]
        it = 0
        print(f"Training model {self.llm_model} on data source {self.data_source}...")
        with tqdm.tqdm(total=100) as pbar:
            while not done:
                c_action, r_action = env.action_sample()
                info = coder.actions[c_action](info)
                new_state, reward, done, truncated, info = env.step(c_action, info=info, agent='coder')
                info = reviewer.actions[r_action](info)
                new_state, reward_rev, done, truncated, info = env.step(r_action, info=info, agent='reviewer')
                reviewer.update_policy((state, r_action), new_state, reward_rev)
                coder.update_policy((state, c_action), new_state, reward)
                rewards.append(reward + reward_rev)
                state = new_state
                print(f"Step {it}: Reward: {reward + reward_rev}, State: {state}")
                evolution.append(state)

                it += 1
                pbar.update(1)
        self.plot_rewards(rewards)

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel('Time step')
        plt.ylabel('Total Reward')
        plt.show()

    def view_report(self):
        pass


def main():
    fire.Fire(KarlInterface)
