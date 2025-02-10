import os
import random
from typing import Any

import more_itertools
import numpy as np
import torch
import tqdm
import wandb

from games.wordle.environment import BatchRoller
from games.wordle.model import Sample
from games.wordle.model import Model
from games.wordle.reward import Reward
from games.wordle.tracker import Tracker
from games.wordle.policy import Policy, SamplingPolicy
from games.wordle.vocab import Vocab


def compute_returns(episode_samples: list[Sample], return_discount: float) -> list[Sample]:
    long_term_return = 0
    for sample in reversed(episode_samples):
        long_term_return = sample.reward + return_discount * long_term_return
        sample.long_term_return = long_term_return
    return episode_samples


def normalize_returns(samples: list[Sample]) -> list[Sample]:
    returns = [sample.long_term_return for sample in samples]
    mean, std = np.mean(returns), np.std(returns)
    for sample in samples:
        sample.long_term_return = (sample.long_term_return - mean) / (std + 1e-8)
    return samples


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        epochs: int,
        num_episodes_per_epoch: int,
        num_eval_episodes_per_epoch: int,
        evaluate_every_n_epochs: int,
        checkpoint_every_n_epochs: int,
        batch_size: int,
        lr: float,
        vocab: Vocab,
        reward_fn: Reward,
        return_discount: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.return_discount = return_discount
        self.scaler = torch.GradScaler(init_scale=2**16)

    def checkpoint(self, epoch_id: int) -> None:
        self.model.eval()
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"model_{epoch_id:05d}.pth"))

    @torch.no_grad
    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()

        roller = BatchRoller(self.vocab)
        transitions = roller.run(
            policy=SamplingPolicy(model=self.model, vocab=self.vocab),
            seeds=[epoch_id * self.num_episodes_per_epoch + idx for idx in range(self.num_episodes_per_epoch)]
        )

        samples = []
        for episode_transitions in transitions:
            rewards = self.reward_fn(episode_transitions)

            episode_samples = []
            for transition, reward in zip(episode_transitions, rewards):
                episode_samples.append(Sample(state=transition.source_state, action=transition.action, reward=reward))

            episode_samples = compute_returns(episode_samples, return_discount=self.return_discount)
            samples.extend(episode_samples)

        return normalize_returns(samples)

        
    def train(self, samples: list[Any], tracker: Tracker) -> None:
        self.model.train()
        random.shuffle(samples)

        num_batches = 0
        for batch_samples in tqdm.tqdm(more_itertools.chunked(samples, self.batch_size), desc="Train step"):
            self.optimizer.zero_grad()

            num_batches += 1
            batch_weight = len(batch_samples) / len(samples)

            with torch.amp.autocast():
                loss = batch_weight * self.model.loss(batch_samples)
                assert torch.isnan(loss).sum() == 0
                tracker.log_value("loss", loss.item())

                loss = self.scaler.scale(loss)

                loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        tracker.log_value("num_batches", num_batches)


    @torch.no_grad
    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        self.model.eval()

        roller = BatchRoller(self.vocab)
        transitions = roller.run(
            policy=SamplingPolicy(model=self.model, vocab=self.vocab),
            seeds=list(range(self.num_episodes_per_epoch))
        )

        for episode_transitions in transitions:
            tracker.log_value("wins", episode_transitions[-1].target_state.win)


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
