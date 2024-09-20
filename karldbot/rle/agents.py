from typing import Dict, Any

class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def play(self, n_episodes=1, render=False):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
            self.env.close()

    def update_policy(self, state: Dict[str,Any], action:str):
        raise NotImplementedError

    def select_action(self, state: Dict[str,Any]) -> str:
        raise NotImplementedError
