from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
import random

import numpy as np
import torch
import wandb

from games.ticket2ride.tracker import Tracker
from games.wordle.consts import EXACT_MATCH, MAX_GUESSES, WORD_LENGTH
from games.wordle.environment import BatchRoller, compute_hint
from games.wordle.model import Model, ModelConfig, Sample, amp_context
from games.wordle.policy import ArgmaxPolicy
from games.wordle.state import Action, State
from games.wordle.trainer import compute_metrics
from games.wordle.vocab import Vocab


def generate_samples(vocab: Vocab, tracker: Tracker) -> list[Sample]:
    prefix_length = random.randint(5, MAX_GUESSES - 1)

    secret = random.choice(vocab.words)
    while secret in ["pique", "bayou"]:
        secret = random.choice(vocab.words)

    guesses = []
    hints = []
    steps = list(range(prefix_length))
    random.shuffle(steps)
    for step in steps:
        valid_prefix = False
        while not valid_prefix:
            guess = random.choice(vocab.words)
            while guess[step] != secret[step]:
                guess = random.choice(vocab.words)

            hint = compute_hint(secret, guess)
            if hint != [EXACT_MATCH] * WORD_LENGTH:
                valid_prefix = True

        guesses.append(guess)
        hints.append(hint)

    candidates = []
    for word in vocab.words:
        valid = True
        for guess, hint in zip(guesses, hints):
            if compute_hint(word, guess) != hint:
                valid = False
                break

        if valid:
            candidates.append(word)

    tracker.log_value(f"candidates_at_length_{prefix_length}", len(candidates))

    next_guess = random.choice(candidates)
    samples = []
    for idx, letter in enumerate(next_guess):
        samples.append(
            Sample(
                state=State(guesses=guesses + ([next_guess[:idx]] if idx > 0 else []), hints=hints),
                action=Action(letter=letter, mask=vocab.get_mask(next_guess[:idx]), lprobs=None),
            )
        )

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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--heads", type=int, default=1, help="Number of heads")
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

            samples = []
            with tracker.scope("generate_samples"):
                for _ in range(args.batch_size):
                    samples.extend(generate_samples(vocab, tracker))

            model.train()
            with tracker.scope("train"):
                optimizer.zero_grad()
                with amp_context():
                    loss = model.supervised_loss(samples)

                tracker.log_value("supervised_loss", loss.item())

                loss = scaler.scale(loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()

        metrics = tracker.report()
        wandb.log(metrics, step=step)
        print(
            f"Step: {step}, Loss: {metrics['train/supervised_loss_mean']}, "
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