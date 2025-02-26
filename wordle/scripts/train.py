import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from wordle.distributed import distributed_cleanup, distributed_setup
from wordle.model import (
    MLP,
    AdvantageType,
    AlgoConfig,
    MLPConfig,
    ModelConfig,
    PolicyLossType,
    SamplingType,
    Transformer,
    TransformerConfig,
)
from wordle.reward import Reward
from wordle.trainer import Trainer
from wordle.vocab import Vocab, load_words
from wordle.wandb import wandb_config_update, wandb_finish, wandb_init


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment run name")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.path.expanduser("~/data/checkpoints/wordle"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--initial_checkpoint_path",
        type=str,
        default=None,
        help="Initial checkpoint for model/optimizer"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10_000, help="Number of epochs")
    parser.add_argument("--updates_per_epoch", type=int, default=50, help="Number of gradient updates per batch")
    parser.add_argument("--num_episodes_per_epoch", type=int, default=500, help="Number of episodes per epoch")
    parser.add_argument("--num_eval_episodes_per_epoch", type=int, default=None, help="Number of episodes per epoch")
    parser.add_argument("--evaluate_every_n_epochs", type=int, default=25, help="Evaluate every n epochs")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=20, help="Policy checkpointing frequency")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=os.path.expanduser("~/code/ml/games/wordle/vocab_easy.txt"),
        help="File containing the list of eligible words",
    )
    parser.add_argument("--max_secret_options", type=int, default=None, help="Limit the number of secret options")
    parser.add_argument("--no_match_reward", type=float, default=0, help="Reward for guessing an incorrect letter")
    parser.add_argument(
        "--letter_match_reward",
        type=float,
        default=0,
        help="Reward for guessing a correct letter, but incorrect location",
    )

    parser.add_argument("--exact_match_reward", type=float, default=0, help="Reward for guessing the correct letter")
    parser.add_argument("--win_reward", type=float, default=1, help="Reward for winning a game")
    parser.add_argument(
        "--advantage_type",
        default=AdvantageType.GENERALIZED_ADVANTAGE.value,
        choices=[e.value for e in AdvantageType],
        help="What kind of advantage to use",
    )
    parser.add_argument(
        "--bootstrap_values",
        default=False,
        action="store_true",
        help="Bootstrap value estimates instead of using monte carlo returns as targets",
    )
    parser.add_argument(
        "--policy_loss_type",
        default=PolicyLossType.PPO.value,
        choices=[e.value for e in PolicyLossType],
        help="Policy loss type",
    )
    parser.add_argument("--ppo_clip_coeff", type=float, default=0.1, help="PPO clipping coefficient")
    parser.add_argument("--return_discount", type=float, default=0.95, help="Return discount factor (gamma)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="TD lambda for generalize advantage estimate")
    parser.add_argument("--value_loss_weight", type=float, default=0.05, help="Value loss weight")
    parser.add_argument("--entropy_loss_weight", type=float, default=0.03, help="Entropy loss weight")
    parser.add_argument("--win_loss_weight", type=float, default=0.05, help="Win loss weight")
    parser.add_argument(
        "--model_type", type=str, choices=["transformer", "mlp"], default="mlp", help="Model type"
    )
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads")
    parser.add_argument(
        "--separate_encoder",
        default=False,
        action="store_true",
        help="Do not share encoder params between policy and value function"
    )
    parser.add_argument("--skip_mask", default=False, action="store_true", help="Whether to skip masking when sampling")
    parser.add_argument(
        "--sampling_type",
        type=str,
        default="none",
        choices=[e.value for e in SamplingType],
        help="Logic for picking samples for computing loss"
    )
    parser.add_argument("--sampling_beta", type=float, default=1, help="Exponent for sampling weights")
    args = parser.parse_args()

    rank = distributed_setup()


    wandb_init(project="wordle", name=args.name, dir="/wandb")
    wandb_config_update(args)

    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training but ensures consistency
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.initial_checkpoint_path is not None:
        with open(os.path.join(args.initial_checkpoint_path, "config.json"), "r") as f:
            model_config = ModelConfig.validate_json(f.read())
    elif args.model_type == "transformer":
        model_config = TransformerConfig(
            layers=args.layers,
            dim=args.dim,
            heads=args.heads,
            separate_encoder=args.separate_encoder,
        )
    elif args.model_type == "mlp":
        model_config = MLPConfig(layers=args.layers, dim=args.dim, separate_encoder=args.separate_encoder)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    if model_config.type == "transformer":
        model = Transformer(device=device, config=model_config)
    elif model_config.type == "mlp":
        model = MLP(device=device, config=model_config)
    else:
        raise ValueError(f"Model config type {model_config.type} not supported")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.initial_checkpoint_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.initial_checkpoint_path, "model.pth"), weights_only=True))
        optimizer.load_state_dict(
            torch.load(os.path.join(args.initial_checkpoint_path, "optimizer.pth"), weights_only=True)
        )

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

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
        updates_per_epoch=args.updates_per_epoch,
        lr=args.lr,
        vocab=Vocab(
            words=load_words(args.vocab_path),
            max_secret_options=args.max_secret_options,
            skip_mask=args.skip_mask,
        ),
        reward_fn=Reward(
            no_match_reward=args.no_match_reward,
            letter_match_reward=args.letter_match_reward,
            exact_match_reward=args.exact_match_reward,
            win_reward=args.win_reward,
        ),
        algo_config=AlgoConfig(
            advantage_type=AdvantageType(args.advantage_type),
            boostrap_values=args.bootstrap_values,
            policy_loss_type=PolicyLossType(args.policy_loss_type),
            ppo_clip_coeff=args.ppo_clip_coeff,
            sampling_type=SamplingType(args.sampling_type),
            sampling_beta=args.sampling_beta,
        ),
        return_discount=args.return_discount,
        gae_lambda=args.gae_lambda,
        value_loss_weight=args.value_loss_weight,
        entropy_loss_weight=args.entropy_loss_weight,
        win_loss_weight=args.win_loss_weight,
    )
    trainer.run()

    wandb_finish()
    distributed_cleanup()


if __name__ == "__main__":
    main()
