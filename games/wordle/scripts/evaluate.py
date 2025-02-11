from argparse import ArgumentParser
import json
import os
import random

import numpy as np
import torch

from games.wordle.environment import BatchRoller
from games.wordle.model import Model
from games.wordle.policy import SamplingPolicy
from games.wordle.tracker import Tracker
from games.wordle.trainer import compute_metrics
from games.wordle.vocab import Vocab


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint to load")
    parser.add_argument(
        "--vocab_path",  
        type=str,       
        default=os.path.expanduser("~/code/ml/games/wordle/vocab_easy.txt"),
        help="File containing the list of eligible words",
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training but ensures consistency
    torch.use_deterministic_algorithms(True)


    model = Model(device=torch.device("cpu"), layers=2, dim=64, heads=1)
    model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    model.to(torch.device("cuda"))

    vocab = Vocab(args.vocab_path)
    roller = BatchRoller(vocab=vocab)
    policy = SamplingPolicy(model=model, vocab=vocab)
    rollouts = roller.run(policy, seeds=list(range(args.num_episodes)))

    for rollout in rollouts:
        print(f"Secret: {rollout.secret}")
        end_state = rollout.transitions[-1].target_state
        print(f"Guesses: {list(zip(end_state.guesses, end_state.hints))}")
    
    tracker = Tracker()
    compute_metrics(rollouts, tracker)

    print(json.dumps(tracker.report(), indent=2))


if __name__ == "__main__":
    main()