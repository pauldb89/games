import collections
import copy
import time
import hashlib
import json
import math
import os
import random
from typing import Any

import more_itertools
import numpy as np
import torch

from common.distributed import get_rank, tqdm_once, world_size
from games.wordle.distributed import print_once
from games.wordle.wandb import wandb_log
from games.wordle.consts import AMP_ENABLED, EXACT_MATCH, LETTER_MATCH, MAX_GUESSES
from games.wordle.environment import BatchRoller, Rollout
from games.wordle.model import AlgoConfig, Sample, amp_context
from games.wordle.model import Model
from games.wordle.reward import Reward
from games.wordle.tracker import Tracker
from games.wordle.policy import ArgmaxPolicy, StochasticPolicy
from games.wordle.vocab import Vocab


def compute_returns(episode_samples: list[Sample], return_discount: float, gae_lambda: float) -> list[Sample]:
    long_term_return = 0
    next_value = 0
    advantage = 0
    for sample in reversed(episode_samples):
        long_term_return = sample.reward + return_discount * long_term_return
        sample.long_term_return = long_term_return

        td_error = sample.reward + return_discount * next_value - sample.action.value
        advantage = td_error + gae_lambda * return_discount * advantage
        next_value = sample.action.value

        sample.td_error = td_error
        sample.advantage = advantage
        sample.estimated_return = advantage + sample.action.value

    return episode_samples


def normalize_values(samples: list[Sample], source_field: str, target_field: str) -> list[Sample]:
    values = [getattr(s, source_field) for s in samples]
    mean, std = np.mean(values), np.std(values)
    for sample in samples:
        setattr(sample, target_field, (getattr(sample, source_field) - mean) / (std + 1e-8))
    return samples


def normalize_returns(samples: list[Sample]) -> list[Sample]:
    samples = normalize_values(samples, source_field="long_term_return", target_field="normalized_long_term_return")
    samples = normalize_values(samples, source_field="estimated_return", target_field="normalized_estimated_return")
    return normalize_values(samples, source_field="advantage", target_field="normalized_advantage")


def compute_entropy(rollouts: list[Rollout]) -> float:
    lprobs = []
    masks = []
    for rollout in rollouts:
        for transition in rollout.transitions:
            assert transition.action.lprobs is not None
            lprobs.append(transition.action.lprobs)
            masks.append(transition.action.mask)

    lprobs = np.stack(lprobs, axis=0)
    masks = np.array(masks)
    masked_lprobs = np.where(masks, lprobs, 0)
    return -(np.exp(masked_lprobs) * masked_lprobs).sum(axis=1).mean()


def compute_metrics(rollouts: list[Rollout], tracker: Tracker) -> None:
    initial_guesses = collections.defaultdict(int)
    for rollout in rollouts:
        end_state = rollout.transitions[-1].target_state
        initial_guesses[end_state.guesses[0]] += 1

        tracker.log_value("wins", end_state.win)

        for turn_id, hint in enumerate(end_state.hints, start=1):
            turn_exact_matches = turn_letter_matches = turn_no_matches = 0
            for feedback in hint:
                if feedback == EXACT_MATCH:
                    turn_exact_matches += 1
                elif feedback == LETTER_MATCH:
                    turn_letter_matches += 1
                else:
                    turn_no_matches += 1

            tracker.log_value(f"turn_{turn_id}_exact_matches", turn_exact_matches)
            tracker.log_value(f"turn_{turn_id}_letter_matches", turn_letter_matches)
            tracker.log_value(f"turn_{turn_id}_no_matches", turn_no_matches)

        follow_ups = good_follow_ups = 0
        follow_ups_last_turn = good_follow_ups_last_turn = 0
        for turn_id in range(1, len(end_state.hints)):
            prev_hint = end_state.hints[turn_id - 1]
            hint = end_state.hints[turn_id]

            for feedback in prev_hint:
                if feedback == EXACT_MATCH:
                    follow_ups += 1
                    if turn_id == len(end_state.hints) - 1:
                        follow_ups_last_turn += 1

            prev_guess = end_state.guesses[turn_id - 1]
            guess = end_state.guesses[turn_id]
            if prev_guess == guess:
                continue

            for prev_feedback, feedback in zip(prev_hint, hint):
                if prev_feedback == EXACT_MATCH and feedback == EXACT_MATCH:
                    good_follow_ups += 1
                    if turn_id == len(end_state.hints) - 1:
                        good_follow_ups_last_turn += 1

        tracker.log_value("good_follow_up_ratio", good_follow_ups / follow_ups if follow_ups else 0)
        tracker.log_value(
            "good_follow_up_ratio_last_turn",
            good_follow_ups_last_turn / follow_ups_last_turn if follow_ups_last_turn else 0
        )

        tracker.log_value("repeated_guesses", len(end_state.guesses) - len(set(end_state.guesses)))
        if len(end_state.guesses) >= 2:
            tracker.log_value("repeated_last_guess", end_state.guesses[-1] == end_state.guesses[-2])

        if end_state.win:
            tracker.log_value("turns_to_win", len(end_state.hints))
            for num_turns in range(1, MAX_GUESSES+1):
                tracker.log_value(f"wins_in_{num_turns}_turns", len(end_state.hints) == num_turns)

    tracker.log_value("repeated_initial_guesses", len(rollouts) - len(initial_guesses))
    tracker.log_value("entropy", compute_entropy(rollouts))


def compute_reward_metrics(samples: list[Sample], tracker: Tracker) -> None:
    for sample in samples:
        tracker.log_value("reward", sample.reward)
        tracker.log_value("long_term_return", sample.long_term_return)
        tracker.log_value("normalized_long_term_return", sample.normalized_long_term_return)
        tracker.log_value("win_prob", sample.action.win_prob)
        tracker.log_value("win_accuracy", int(sample.action.win_prob >= 0.5) == sample.win)

    values = np.array([s.action.value for s in samples])
    value_targets = np.array([s.normalized_long_term_return for s in samples])
    tracker.log_value("explained_variance", 1 - np.var(values - value_targets) / (np.var(value_targets) + 1e-8))


def batch_samples(samples: list[Sample], algo_config: AlgoConfig, updates_per_epoch: int) -> list[list[Sample]]:
    if algo_config.sampling_type == "none":
        return [list(batch) for batch in more_itertools.divide(updates_per_epoch, samples)]

    batch_size = len(samples) // updates_per_epoch
    weights = []
    for s in samples:
        if algo_config.sampling_type == "advantage":
            s.importance_weight = math.exp(s.advantage) ** algo_config.sampling_beta
        elif algo_config.sampling_type == "td_error":
            s.importance_weight = abs(s.td_error) ** algo_config.sampling_beta
        else:
            raise ValueError(f"Unknown sampling strategy {algo_config.sampling_type}")
        weights.append(s.importance_weight)

    return [random.choices(samples, weights=weights, k=batch_size) for _ in range(updates_per_epoch)]


def gather_trackers(tracker: Tracker) -> Tracker:
    all_trackers = [None] * world_size()
    torch.distributed.all_gather_object(all_trackers, tracker)

    agg_tracker = copy.deepcopy(tracker)
    for rank, gathered_tracker in enumerate(all_trackers):
        if rank != get_rank():
            for metric_name, values in gathered_tracker.metrics.items():
                agg_tracker.metrics[metric_name].extend(values)

    return agg_tracker


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
        updates_per_epoch: int,
        lr: float,
        vocab: Vocab,
        reward_fn: Reward,
        algo_config: AlgoConfig,
        return_discount: float,
        gae_lambda: float,
        value_loss_weight: float,
        entropy_loss_weight: float,
        win_loss_weight: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.updates_per_epoch = updates_per_epoch
        self.lr = lr
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.algo_config = algo_config
        self.return_discount = return_discount
        self.gae_lambda = gae_lambda
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.win_loss_weight = win_loss_weight
        self.scaler = torch.GradScaler(init_scale=2**16)

    def checkpoint(self, epoch_id: int) -> None:
        self.model.eval()

        model = self.model.module
        checkpoint_dir = os.path.join(self.checkpoint_path, f"{epoch_id:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            f.write(model.config.model_dump_json())
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pth"))

    @torch.no_grad
    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()

        roller = BatchRoller(self.vocab)

        rollouts = roller.run(
            policy=StochasticPolicy(model=self.model, vocab=self.vocab),
            seeds=[
                epoch_id * self.num_episodes_per_epoch + idx
                for idx in range(get_rank(), self.num_episodes_per_epoch, world_size())
            ],
        )

        compute_metrics(rollouts, tracker)
        # for r in rollouts:
        #     end_state = r.transitions[-1].target_state
        #     print(f"Secret: {r.secret}, guesses: {end_state.guesses}, hints: {end_state.hints}")

        samples = []
        for rollout in rollouts:
            rewards = self.reward_fn(rollout.transitions)
            end_state = rollout.transitions[-1].target_state

            episode_samples = []
            for transition, reward in zip(rollout.transitions, rewards):
                episode_samples.append(
                    Sample(
                        state=transition.source_state,
                        action=transition.action,
                        reward=reward,
                        win=int(end_state.win),
                        secret=rollout.secret
                    )
                )

            episode_samples = compute_returns(
                episode_samples,
                return_discount=self.return_discount,
                gae_lambda=self.gae_lambda,
            )
            samples.extend(episode_samples)

        samples = normalize_returns(samples)
        compute_reward_metrics(samples, tracker)
        return samples


    def train(self, samples: list[Any], tracker: Tracker) -> None:
        self.model.train()

        random.shuffle(samples)
        tracker.log_value("num_samples", len(samples))

        num_batches = 0
        for batch in tqdm_once(batch_samples(samples, self.algo_config, self.updates_per_epoch), desc="Train step"):
            num_batches += 1
            batch_weight = len(batch) / len(samples)

            tracker.log_value("batch_size", len(batch))
            for sample in batch:
                tracker.log_value("sample_win", sample.win)

            self.optimizer.zero_grad()

            with amp_context():
                policy_loss, value_loss, entropy_loss, win_loss = self.model(batch, self.algo_config)

                policy_loss *= batch_weight
                entropy_loss *= batch_weight
                value_loss *= batch_weight
                win_loss *= batch_weight

                tracker.log_value("policy_loss", policy_loss.item())
                tracker.log_value("value_loss", value_loss.item())
                tracker.log_value("entropy_loss", entropy_loss.item())
                tracker.log_value("win_loss", win_loss.item())

                loss = (
                    policy_loss
                    + self.value_loss_weight * value_loss
                    + self.entropy_loss_weight * entropy_loss
                    + self.win_loss_weight * win_loss
                )
                assert torch.isnan(loss).sum() == 0
                tracker.log_value("loss", loss.item())

            if AMP_ENABLED:
                loss = self.scaler.scale(loss)
                loss.backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                self.optimizer.step()

        tracker.log_value("num_batches", num_batches)


    @torch.no_grad
    def evaluate(self, tracker: Tracker, epoch_id: int) -> None:
        self.model.eval()

        roller = BatchRoller(self.vocab)
        rollouts = roller.run(
            policy=ArgmaxPolicy(model=self.model, vocab=self.vocab),
            seeds=[-(idx+1) for idx in range(get_rank(), self.num_eval_episodes_per_epoch, world_size())]
        )

        compute_metrics(rollouts, tracker)

        agg_tracker = gather_trackers(tracker)

        metrics = {k: v for k, v in agg_tracker.report().items() if k.startswith("eval") and k.endswith("mean")}
        print_once(f"Evaluation step {epoch_id}: {json.dumps(metrics, indent=2)}")


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

            tracker = gather_trackers(tracker)
            metrics = tracker.report()
            print_once(
                f"Epoch: {epoch_id}, Loss: {metrics['train/loss_sum']}, "
                f"Win rate: {100 * metrics['collect_samples/wins_mean']:.2f}%, "
                f"Total time: {metrics['t_overall_mean']} seconds"

            )
            wandb_log(metrics, step=epoch_id)

        tracker = Tracker()
        with tracker.scope("checkpoint"):
            with tracker.timer("t_overall"):
                self.checkpoint(epoch_id=self.epochs)

        with tracker.scope("evaluate"):
            with tracker.timer("t_overall"):
                self.evaluate(tracker, epoch_id=self.epochs)
