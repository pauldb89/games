from argparse import ArgumentParser
import collections
from dataclasses import dataclass
import json
import os
import random

import numpy as np
import torch
import wandb

from games.ticket2ride.tracker import Tracker
from games.wordle.consts import EXACT_MATCH, LETTER_MATCH, MAX_GUESSES, NO_MATCH, WORD_LENGTH
from games.wordle.environment import BatchRoller, Environment, compute_hint
from games.wordle.model import Model, ModelConfig, Sample, amp_context
from games.wordle.policy import ArgmaxPolicy, StochasticPolicy
from games.wordle.state import Action, State
from games.wordle.trainer import compute_metrics
from games.wordle.vocab import Vocab


def is_exact_match_with_earlier_exact_match(
    guesses: list[str], 
    hints: list[list[int]],
    current_idx: int,
    current_letter_idx: int,
) -> bool:
    if hints[current_idx][current_letter_idx] != EXACT_MATCH:
        return False
    
    for other_idx in range(current_idx):
        if guesses[other_idx][current_letter_idx] == guesses[current_idx][current_letter_idx]:
            return True
        
    return False


def is_exact_match_with_earlier_letter_match(
    guesses: list[str], 
    hints: list[list[int]],
    current_idx: int,
    current_letter_idx: int,    
) -> bool:
    if hints[current_idx][current_letter_idx] != EXACT_MATCH:
        return False

    for other_idx in range(current_idx):
        for other_letter_idx in range(WORD_LENGTH):
            if (
                guesses[other_idx][other_letter_idx] == guesses[current_idx][current_letter_idx]
                and hints[other_idx][other_letter_idx] == LETTER_MATCH
            ):
                return True
            
    return False


def is_letter_match_with_earlier_letter_match(
    guesses: list[str], 
    hints: list[list[int]],
    current_idx: int,
    current_letter_idx: int,            
) -> bool:
    if hints[current_idx][current_letter_idx] != LETTER_MATCH:
        return False

    for other_idx in range(current_idx):
        if guesses[other_idx][current_letter_idx] == guesses[current_idx][current_letter_idx]:
            return False
        
    for other_idx in range(current_idx):
        for other_letter_idx in range(WORD_LENGTH):
            if (
                guesses[other_idx][other_letter_idx] == guesses[current_idx][current_letter_idx]
                and hints[other_idx][other_letter_idx] == LETTER_MATCH
            ):
                return True
            
    return False


def wrong_letter_repeat(
    guesses: list[str],
    hints: list[list[int]],
    current_idx: int,
    current_letter_idx: int
) -> bool:
    if hints[current_idx][current_letter_idx] != NO_MATCH:
        return False
    
    for other_idx in range(current_idx):
        for other_letter_idx in range(WORD_LENGTH):
            if guesses[other_idx][other_letter_idx] == guesses[current_idx][current_letter_idx]:
                return True
            
    return False


def is_letter_match_same_position(
    guesses: list[str],
    hints: list[list[int]],
    current_idx: int,
    current_letter_idx: int        
) -> bool:
    if hints[current_idx][current_letter_idx] != LETTER_MATCH:
        return False
    
    for other_idx in range(current_idx):
        if guesses[other_idx][current_letter_idx] == guesses[current_idx][current_letter_idx]:
            return True
        
    return False


@torch.no_grad
def generate_samples(policy: StochasticPolicy, vocab: Vocab, tracker: Tracker, batch_size: int) -> list[Sample]:
    samples = []
    env = Environment(vocab)
    reason_counts = collections.defaultdict(int)

    while len(samples) <= batch_size:
        secret = random.choice(vocab.words)
        state = env.reset(secret=secret)
        while not state.terminal:
            action = policy.choose_actions([state])[0]
            state = env.step(action)


        guesses = state.guesses
        hints = state.hints
        print(f"{secret=} {guesses=} {hints=}")

        for idx, guess in enumerate(guesses):
            for letter_idx in range(WORD_LENGTH):
                reason = None
                reward = 0
                if is_exact_match_with_earlier_exact_match(guesses, hints, idx, letter_idx):
                    reason = 0
                    reward = 1.0
                elif is_exact_match_with_earlier_letter_match(guesses, hints, idx, letter_idx):
                    reason = 1
                    reward = 1.0
                elif is_letter_match_with_earlier_letter_match(guesses, hints, idx, letter_idx):
                    reason = 2
                    reward = 1.0
                elif wrong_letter_repeat(guesses, hints, idx, letter_idx):
                    reason = 3
                    reward = -1
                elif is_letter_match_same_position(guesses, hints, idx, letter_idx):
                    reason = 4
                    reward = -1

                if reason is not None and reason_counts[reason] * 4 <= batch_size:
                    reason_counts[reason] += 1
                    samples.append(
                        Sample(
                            state=State(
                                guesses=guesses[:idx] + ([guess[:letter_idx]] if letter_idx > 0 else []), 
                                hints=hints[:idx],
                            ),
                            action=Action(
                                letter=guess[letter_idx], 
                                mask=vocab.get_mask(guess[:letter_idx]),
                                lprobs=None,
                            ),
                            reward=reward,
                        )
                    )

                    if len(samples) == batch_size:
                        break

    for reason in range(5):
        tracker.log_value(f"samples_reason_{reason}_ratio", reason_counts[reason] / len(samples))

    return samples


def evaluate(vocab: Vocab, model: Model, tracker: Tracker, step: int, num_eval_episodes: int) -> None:
    model.eval()
    with tracker.scope("evaluate"):
        roller = BatchRoller(vocab)
        rollouts = roller.run(
            policy=ArgmaxPolicy(model=model, vocab=vocab),
            seeds=[-(idx+1) for idx in range(num_eval_episodes)]
        )
        compute_metrics(rollouts, tracker)

    metrics = {k: v for k, v in tracker.report().items() if k.startswith("eval") and k.endswith("mean")}
    print(f"Evaluation step {step}: {json.dumps(metrics, indent=2)}")


def checkpoint(model: Model, checkpoint_dir: str) -> None:
    model.eval()

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        f.write(model.config.model_dump_json())
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.expanduser("~/data/checkpoints/wordle"),
        help="Checkpoint path",
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=os.path.expanduser("~/code/ml/games/wordle/vocab_easy.txt"),
        help="File containing the list of eligible words",
    )
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps")
    parser.add_argument("--evaluate_every_n_steps", type=int, default=100, help="How often to evaluate")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000, help="Policy checkpointing frequency")
    parser.add_argument("--num_eval_episodes_per_step", type=int, default=100, help="Number of samples to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Gradient clipping norm")
    parser.add_argument("--dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads")
    parser.add_argument("--skip_mask", default=False, action="store_true", help="Skip masks")
    args = parser.parse_args()

    wandb.init(project="wordle", name=args.name, dir="/wandb")
    wandb.config.update(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training but ensures consistency
    torch.use_deterministic_algorithms(True)

    model_config = ModelConfig(layers=args.layers, dim=args.dim, heads=args.heads, share_weights=False)
    model = Model(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), config=model_config)

    scaler = torch.GradScaler(init_scale=2**16)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(args.checkpoint_path, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)

    vocab = Vocab(path=args.vocab_path)
    for step in range(args.steps):
        tracker = Tracker()
        with tracker.timer("t_overall"):
            if step % args.evaluate_every_n_steps == 0:
                evaluate(
                    vocab=vocab, 
                    model=model,
                    tracker=tracker, 
                    step=step, 
                    num_eval_episodes=args.num_eval_episodes_per_step,
                )

            if step % args.checkpoint_every_n_steps == 0:
                checkpoint(model, checkpoint_dir=os.path.join(checkpoint_path, f"{step:05d}"))

            with tracker.scope("generate_samples"):
                model.eval()
                samples = generate_samples(
                    policy=StochasticPolicy(model, vocab),
                    vocab=vocab, 
                    tracker=tracker, 
                    batch_size=args.batch_size,
                )

            positive_samples = []
            negative_samples = []
            for sample in samples:
                if sample.reward > 0:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)

            model.train()
            with tracker.scope("train"):
                optimizer.zero_grad()
                with amp_context():
                    positive_loss = model.supervised_loss(positive_samples) if positive_samples else torch.tensor(0)
                    negative_loss = model.supervised_loss(negative_samples) if negative_samples else torch.tensor(0)
                    loss = (len(positive_samples) * positive_loss + len(negative_samples) * negative_loss) / len(samples)

                tracker.log_value("supervised_positive_loss", positive_loss.item())
                tracker.log_value("supervised_negative_loss", negative_loss.item())
                tracker.log_value("supervised_loss", loss.item())

                loss = scaler.scale(loss)
                loss.backward()
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        metrics = tracker.report()
        wandb.log(metrics, step=step)
        print(
            f"Step: {step}, Loss: {metrics['train/supervised_loss_mean']}, "
            f"Positive Loss: {metrics['train/supervised_positive_loss_mean']}, "
            f"Negative Loss: {metrics['train/supervised_negative_loss_mean']}, "
            f"Total time: {metrics['t_overall_mean']} seconds"
        )

    evaluate(
        vocab=vocab, 
        model=model,
        tracker=tracker, 
        step=args.steps, 
        num_eval_episodes=args.num_eval_episodes_per_step,
    )

    checkpoint(model, checkpoint_dir=os.path.join(checkpoint_path, f"{args.steps:05d}"))


if __name__ == "__main__":
    main()