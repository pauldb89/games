import os
from argparse import ArgumentParser
import random

import numpy as np
import torch
import wandb

from games.wordle.model import Model, ModelConfig
from games.wordle.reward import Reward
from games.wordle.trainer import Trainer
from games.wordle.vocab import Vocab


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
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--updates_per_batch", type=int, default=1, help="Number of gradient updates per batch")
    parser.add_argument("--num_episodes_per_epoch", type=int, default=100, help="Number of episodes per epoch")
    parser.add_argument("--num_eval_episodes_per_epoch", type=int, default=100, help="Number of episodes per epoch")
    parser.add_argument("--evaluate_every_n_epochs", type=int, default=25, help="Evaluate every n epochs")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=20, help="Policy checkpointing frequency")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=os.path.expanduser("~/code/ml/games/wordle/vocab_easy.txt"),
        help="File containing the list of eligible words",
    )
    parser.add_argument("--no_match_reward", type=float, default=0, help="Reward for guessing an incorrect letter")
    parser.add_argument(
        "--letter_match_reward",
        type=float,
        default=0,
        help="Reward for guessing a correct letter, but incorrect location",
    )
    parser.add_argument("--exact_match_reward", type=float, default=0, help="Reward for guessing the correct letter")
    parser.add_argument("--win_reward", type=float, default=1, help="Reward for winning a game")
    parser.add_argument("--return_discount", type=float, default=0.95, help="Return discount factor (gamma)")
    parser.add_argument("--entropy_loss_weight", type=float, default=0, help="Entropy loss weight")
    parser.add_argument("--dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads")
    parser.add_argument("--skip_mask", default=False, action="store_true", help="Whether to skip masking when sampling")
    parser.add_argument("--resume_step", type=int, default=None, help="Resume step for multi-trajectory rollouts")
    parser.add_argument("--num_expansions", type=int, default=None, help="Number of expansions to resume")
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

    model_config = ModelConfig(
        layers=args.layers, 
        dim=args.dim, 
        heads=args.heads,
    )
    model = Model(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), config=model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(args.checkpoint_path, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        num_episodes_per_epoch=args.num_episodes_per_epoch,
        num_eval_episodes_per_epoch=args.num_eval_episodes_per_epoch,
        evaluate_every_n_epochs=args.evaluate_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        updates_per_batch=args.updates_per_batch,
        lr=args.lr,
        vocab=Vocab(path=args.vocab_path, skip_mask=args.skip_mask),
        reward_fn=Reward(
            no_match_reward=args.no_match_reward,
            letter_match_reward=args.letter_match_reward,
            exact_match_reward=args.exact_match_reward,
            win_reward=args.win_reward,
        ),
        return_discount=args.return_discount,
        entropy_loss_weight=args.entropy_loss_weight,
        resume_step=args.resume_step,
        num_expansions=args.num_expansions,
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
