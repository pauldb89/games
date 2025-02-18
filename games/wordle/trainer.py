import collections
import json
import os
import random
from typing import Any

import more_itertools
import numpy as np
import torch
import tqdm
import wandb

from games.wordle.consts import AMP_ENABLED, EXACT_MATCH, LETTER_MATCH, MAX_GUESSES
from games.wordle.environment import BatchRoller, Rollout
from games.wordle.model import Sample, amp_context
from games.wordle.model import Model
from games.wordle.reward import Reward
from games.wordle.tracker import Tracker
from games.wordle.policy import ArgmaxPolicy, StochasticPolicy
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
        updates_per_batch: int,
        lr: float,
        vocab: Vocab,
        reward_fn: Reward,
        return_discount: float,
        entropy_loss_weight: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.updates_per_batch = updates_per_batch
        self.lr = lr
        self.vocab = vocab
        self.reward_fn = reward_fn
        self.return_discount = return_discount
        self.entropy_loss_weight = entropy_loss_weight
        self.scaler = torch.GradScaler(init_scale=2**16)

    def checkpoint(self, epoch_id: int) -> None:
        self.model.eval()
        checkpoint_dir = os.path.join(self.checkpoint_path, f"{epoch_id:05d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            f.write(self.model.config.model_dump_json())
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))

    @torch.no_grad
    def collect_samples(self, tracker: Tracker, epoch_id: int) -> list[Sample]:
        self.model.eval()

        roller = BatchRoller(self.vocab)
            
        rollouts = roller.run(
            policy=StochasticPolicy(model=self.model, vocab=self.vocab),
            seeds=[epoch_id * self.num_episodes_per_epoch + idx for idx in range(self.num_episodes_per_epoch)],
        )

        compute_metrics(rollouts, tracker)
        for r in rollouts:
            end_state = r.transitions[-1].target_state
            print(f"Secret: {r.secret}, guesses: {end_state.guesses}, hints: {end_state.hints}")

        samples = []
        for rollout in rollouts:
            rewards = self.reward_fn(rollout.transitions)

            episode_samples = []
            for transition, reward in zip(rollout.transitions, rewards):
                episode_samples.append(
                    Sample(
                        state=transition.source_state, action=transition.action, reward=reward, secret=rollout.secret
                    )
                )

            episode_samples = compute_returns(episode_samples, return_discount=self.return_discount)
            samples.extend(episode_samples)

        for sample in samples:
            tracker.log_value("reward", sample.reward)
            tracker.log_value("long_term_return", sample.long_term_return)

        return normalize_returns(samples)

        
    def train(self, samples: list[Any], tracker: Tracker) -> None:
        self.model.train()
        random.shuffle(samples)

        num_batches = 0
        for batch_samples in tqdm.tqdm(more_itertools.divide(self.updates_per_batch, samples), desc="Train step"):
            batch_samples = list(batch_samples)
            num_batches += 1
            batch_weight = len(batch_samples) / len(samples)

            self.optimizer.zero_grad()
            
            with amp_context():
                policy_loss, entropy_loss = self.model.loss(batch_samples)

                policy_loss *= batch_weight
                entropy_loss *= batch_weight

                tracker.log_value("policy_loss", policy_loss.item())
                tracker.log_value("entropy_loss", entropy_loss.item())

                loss = policy_loss + self.entropy_loss_weight * entropy_loss                
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
        
        # with tracker.scope("argmax"):
        roller = BatchRoller(self.vocab)
        rollouts = roller.run(
            policy=ArgmaxPolicy(model=self.model, vocab=self.vocab),
            seeds=[-(idx+1) for idx in range(self.num_eval_episodes_per_epoch)]
        )

        compute_metrics(rollouts, tracker)

            # print("Argmax rollouts")
            # for rollout in rollouts[:5]:
            #     print(f"Secret: {rollout.secret}")
            #     end_state = rollout.transitions[-1].target_state
            #     print(f"Guesses: {list(zip(end_state.guesses, end_state.hints))}")

        # with tracker.scope("stochastic"):
        #     roller = BatchRoller(self.vocab)
        #     rollouts = roller.run(
        #         policy=StochasticPolicy(model=self.model, vocab=self.vocab),
        #         seeds=[-(idx+1) for idx in range(self.num_eval_episodes_per_epoch)]
        #     )

        #     compute_metrics(rollouts, tracker)

        #     print("Stochastic rollouts")
        #     for rollout in rollouts[:5]:
        #         print(f"Secret: {rollout.secret}")
        #         end_state = rollout.transitions[-1].target_state
        #         print(f"Guesses: {list(zip(end_state.guesses, end_state.hints))}")

        metrics = {k: v for k, v in tracker.report().items() if k.startswith("eval") and k.endswith("mean")}
        print(f"Evaluation step {epoch_id}: {json.dumps(metrics, indent=2)}")


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
                f"Epoch: {epoch_id}, Loss: {metrics['train/loss_sum']}, "
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
