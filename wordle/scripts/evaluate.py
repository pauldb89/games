from argparse import ArgumentParser
import json
import os
import random

import numpy as np
import torch

from wordle.environment import BatchRoller
from wordle.model import Model, ModelConfig
from wordle.policy import ArgmaxPolicy
from wordle.tracker import Tracker
from wordle.trainer import compute_metrics
from wordle.vocab import Vocab


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Model checkpoint to load")
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

    with open(os.path.join(args.checkpoint_dir, "config.json"), "r") as f:
        config = ModelConfig.model_validate_json(f.read())

    model = Model(device=torch.device("cpu"), config=config)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pth"), weights_only=True))
    model.to(torch.device("cuda"))

    vocab = Vocab(args.vocab_path)
    roller = BatchRoller(vocab=vocab)
    policy = ArgmaxPolicy(model=model, vocab=vocab)
    with torch.no_grad():
        rollouts = roller.run(policy, seeds=list(range(args.num_episodes)))

    torch.set_printoptions(threshold=100, precision=2)
    for rollout in rollouts:
        print(f"Secret: {rollout.secret}")
        end_state = rollout.transitions[-1].target_state
        print(f"Guesses: {list(zip(end_state.guesses, end_state.hints))}")
    
    tracker = Tracker()
    compute_metrics(rollouts, tracker)

    print(json.dumps(tracker.report(), indent=2))


if __name__ == "__main__":
    main()