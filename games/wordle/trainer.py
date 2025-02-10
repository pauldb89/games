from typing import Any

import numpy as np
import wandb

from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import NO_MATCH
from games.wordle.environment import BatchRoller
from games.wordle.environment import Policy
from games.wordle.environment import Sample
from games.wordle.environment import Transition
from games.wordle.reward import Reward
from games.wordle.tracker import Tracker


class Trainer:
    def __init__(
        self,
        checkpoint_path: str,
        epochs: int,
        num_episodes_per_epoch: int,
        num_eval_episodes_per_epoch: int,
        evaluate_every_n_epochs: int,
        checkpoint_every_n_epochs: int,
        batch_size: int,
        lr: float,
        vocab_path: str,
        reward_fn: Reward,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.vocab_path = vocab_path
        self.reward_fn = reward_fn

    def checkpoint(self, epoch_id: int) -> None:
        pass

    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        roller = BatchRoller(self.vocab_path)
        policy = Policy()
        transitions = roller.run(
            policy,
            seeds=[epoch_id * self.num_episodes_per_epoch + idx for idx in range(self.num_episodes_per_epoch)]
        )

        samples = []
        for episode_transitions in transitions:
            rewards = self.reward_fn(episode_transitions)



    def train(self, samples: list[Any], tracker: Tracker) -> None:
        pass

    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        pass

    def run(self) -> None:
        for epoch_id in range(self.epochs):
            tracker = Tracker()
            with tracker.timer("t_overall"):
                if epoch_id % self.checkpoint_every_n_epochs == 0:
                    with tracker.scope("checkpoint"):
                        with tracker.timer("t_overall"):
                            self.checkpoint(epoch_id)

                if epoch_id % self.evaluate_every_n_epochs == 0:
                    with tracker.scope("evaluate"):
                        with tracker.timer("t_overall"):
                            self.evaluate(tracker, epoch_id)

                with tracker.scope("collect_samples"):
                    with tracker.timer("t_overall"):
                        samples = self.collect_samples(tracker, epoch_id)

                with tracker.scope("train"):
                    with tracker.timer("t_overall"):
                        self.train(samples, tracker)

            metrics = tracker.report()
            print(
                f"Epoch: {epoch_id}, Loss: {metrics['train/loss_mean']}, "
                f"Total time: {metrics['t_overall_mean']} seconds"
            )
            wandb.log(metrics, step=epoch_id)

        tracker = Tracker()
        with tracker.scope("checkpoint"):
            with tracker.timer("t_overall"):
                self.checkpoint(epoch_id=self.epochs)

        with tracker.scope("evaluate"):
            with tracker.timer("t_overall"):
                self.evaluate(tracker, epoch_id=self.epochs)
