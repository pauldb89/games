import os
from argparse import ArgumentParser
import random

import numpy as np
import torch
import wandb

from games.wordle.reward import Reward
from games.wordle.trainer import Trainer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment run name")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.expanduser("~/data/checkpoints/wordle"),
        help="Checkpoint path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_episodes_per_epoch", type=int, default=10, help="Number of episodes per epoch")
    parser.add_argument("--num_eval_episodes_per_epoch", type=int, default=10, help="Number of episodes per epoch")
    parser.add_argument("--evaluate_every_n_epochs", type=int, default=5, help="Evaluate every n epochs")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=20, help="Policy checkpointing frequency")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=os.path.expanduser("~/code/ml/games/wordle/vocab_short.txt"),
        help="File containing the list of eligible words",
    )
    parser.add_argument("--no_match_reward", type=float, default=-2, help="Reward for guessing an incorrect letter")
    parser.add_argument(
        "--letter_match_reward",
        type=float,
        default=-1,
        help="Reward for guessing a correct letter, but incorrect location",
    )
    parser.add_argument("--match_reward", type=float, default=0, help="Reward for guessing the correct letter")
    parser.add_argument("--win_reward", type=float, default=100, help="Reward for winning a game")
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

    trainer = Trainer(
        checkpoint_path=os.path.join(args.checkpoint_path, args.name),
        epochs=args.epochs,
        num_episodes_per_epoch=args.num_episodes_per_epoch,
        num_eval_episodes_per_epoch=args.num_eval_episodes_per_epoch,
        evaluate_every_n_epochs=args.evaluate_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vocab_path=args.vocab_path,
        reward_fn=Reward(
            no_match_reward=args.no_match_reward,
            letter_match_reward=args.letter_match_reward,
            exact_match_reward=args.exact_match_reward,
            win_reward=args.win_reward,
        )
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
