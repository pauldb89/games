import os.path
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb

from ticket2ride.features import DYNAMIC_EXTRACTORS, STATIC_EXTRACTORS
from ticket2ride.model import Model
from ticket2ride.trainer import PolicyGradientTrainer, Reward


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment run name")
    parser.add_argument(
        "--checkpoint_path",
        default=os.path.expanduser("~/data/checkpoints/ticket2ride"),
        help="Checkpoint path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_players", type=int, default=2, help="Number of players")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_episodes_per_epoch", type=int, default=10, help="Number of episodes per epoch")
    parser.add_argument("--num_eval_episodes_per_epoch", type=int, default=10, help="Number of episodes per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--evaluate_every_n_epochs", type=int, default=5, help="Evaluate every n epochs")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=20, help="Policy checkpointing frequency")
    parser.add_argument("--dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--rel_window", type=int, default=100, help="Relative window size")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda parameter for TD-lambda in GAE")
    parser.add_argument("--reward_discount", type=float, default=0.95, help="Reward discount factor")
    parser.add_argument("--win_reward", type=float, default=0, help="Reward for winning in points")
    parser.add_argument("--initial_draw_card_reward", type=float, default=0.0, help="Initial draw card reward")
    parser.add_argument("--final_draw_card_reward", type=float, default=0.0, help="Final draw card reward")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Rescale rewards for better convergence")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--extractors", type=str, default="dynamic", choices=["dynamic", "static"], help="Feature set")
    parser.add_argument("--value_loss_weight", type=float, default=1.0, help="Loss weight for learning value function")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training but ensures consistency
    torch.use_deterministic_algorithms(True)

    wandb.init(project="ticket2ride", name=args.name, dir="/wandb")
    wandb.config.update(args)

    model = Model(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        extractors=DYNAMIC_EXTRACTORS if args.extractors == "dynamic" else STATIC_EXTRACTORS,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        rel_window=args.rel_window,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO(pauldb): Add win reward and mix with points reward. Keep scales in mind.
    checkpoint_path = os.path.join(args.checkpoint_path, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    trainer = PolicyGradientTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        num_players=args.num_players,
        num_epochs=args.epochs,
        num_episodes_per_epoch=args.num_episodes_per_epoch,
        num_eval_episodes_per_epoch=args.num_eval_episodes_per_epoch,
        batch_size=args.batch_size,
        evaluate_every_n_epochs=args.evaluate_every_n_epochs,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        value_loss_weight=args.value_loss_weight,
        reward_fn=Reward(
            win_reward=args.win_reward,
            initial_draw_card_reward=args.initial_draw_card_reward,
            final_draw_card_reward=args.final_draw_card_reward,
            draw_card_horizon_epochs=args.epochs // 2,
            reward_scale=args.reward_scale,
        ),
        reward_discount=args.reward_discount,
        gae_lambda=args.gae_lambda,
    )
    trainer.execute()
    wandb.finish()


if __name__ == "__main__":
    main()


