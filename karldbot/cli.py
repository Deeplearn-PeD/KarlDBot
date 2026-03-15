import asyncio
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml

from karldbot.agents.koder import Koder
from karldbot.agents.reviewer import CodeReviewer
from karldbot.environment import DataScienceProblem, Environment
from karldbot.models.config import AgentConfig, LLMConfig, ProblemConfig
from karldbot.orchestration import AgentCoordinator
from karldbot.report import Report


class KarlInterface:
    def __init__(self, config: str = "config.yaml"):
        self.config_file = config
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)
        self.llm_model = self.config.get("llm_model", "gpt-4o")
        self.data_source = self.config.get("data_source", "")

    def _create_problem_config(self) -> ProblemConfig:
        return ProblemConfig(
            name=self.config.get("problem_name", "Unnamed"),
            data_source=self.config.get("data_source", ""),
            description=self.config.get("description", ""),
            llm_model=self.llm_model,
            max_iterations=self.config.get("max_iterations", 50),
        )

    def train(self) -> None:
        asyncio.run(self._train_async())

    async def _train_async(self) -> None:
        problem_config = self._create_problem_config()
        problem = DataScienceProblem(problem_config)

        llm_config = LLMConfig(model=self.llm_model)
        agent_config = AgentConfig(llm=llm_config)

        coder = Koder(
            config=agent_config,
            problem_description=problem.description,
            data_source=problem.data_source,
            sample_data=problem.sample_data(),
        )
        reviewer = CodeReviewer(config=agent_config)

        coordinator = AgentCoordinator(
            agents={"coder": coder, "reviewer": reviewer},
            problem=problem,
            max_iterations=problem_config.max_iterations,
        )

        report = Report(problem, self.llm_model)
        rewards: list[float] = []
        states: list[int] = []

        print(
            f"Training with {self.llm_model} LLM on data source {self.data_source}..."
        )

        state, info = coordinator.env.reset()
        done = False
        iteration = 0

        with tqdm.tqdm(total=problem_config.max_iterations) as pbar:
            while not done and not state.truncated:
                info["step"] = iteration

                info = await coder.act(state, info)
                report.add_coding_step(info)
                state, reward, done, info = coordinator.env.step_coder(info)

                info = await reviewer.act(state, info)
                report.add_review_step(info)
                state, reward_rev, done, info = coordinator.env.step_reviewer(info)

                total_reward = reward + reward_rev
                if rewards:
                    rewards.append(total_reward + rewards[-1])
                else:
                    rewards.append(total_reward)

                states.append(state.score.quality_level)
                print(
                    f"Step {iteration}: Reward: {total_reward:.2f}, State: {state.score.quality_level}"
                )

                report.save(f"{problem.problem_name}.md")

                iteration += 1
                pbar.update(1)

                if done:
                    break

        self._plot_rewards(rewards)
        self._plot_policies(coder.q_table, reviewer.q_table)

    def _plot_rewards(self, rewards: list[float]) -> None:
        plt.plot(rewards)
        plt.xlabel("Time step")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards")
        plt.savefig("rewards.png")
        plt.show()

    def _plot_policies(
        self, coder_policy: np.ndarray, reviewer_policy: np.ndarray
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(coder_policy)
        ax1.set_title("Coder Q-Table")
        ax1.set_ylabel("State")
        ax1.set_xlabel("Action")
        fig.colorbar(im1, ax=ax1, orientation="vertical")

        im2 = ax2.imshow(reviewer_policy)
        ax2.set_title("Reviewer Q-Table")
        ax2.set_ylabel("State")
        ax2.set_xlabel("Action")
        fig.colorbar(im2, ax=ax2, orientation="vertical")

        plt.tight_layout()
        plt.savefig("policies.png")
        plt.show()

    def view_report(self, filename: str | None = None) -> None:
        problem_name = self.config.get("problem_name", "report")
        report_file = filename or f"{problem_name}.md"

        report = Report(
            DataScienceProblem(self._create_problem_config()),
            self.llm_model,
        )
        report.filename = report_file
        report.open()


def main() -> None:
    fire.Fire(KarlInterface)
