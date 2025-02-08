import random
import time
from argparse import ArgumentParser

import torch

from games.ticket2ride.environment import BatchRoller
from games.ticket2ride.features import DYNAMIC_EXTRACTORS
from games.ticket2ride.model import Model
from games.ticket2ride.policies import KeyboardInputPolicy, ArgmaxModelPolicy, UniformRandomPolicy
from games.ticket2ride.render_utils import print_scorecard, print_board
from games.ticket2ride.tracker import Tracker


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--manual', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.manual:
        policies = [KeyboardInputPolicy(), UniformRandomPolicy(seed=0)]
    else:
        model = Model(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            extractors=DYNAMIC_EXTRACTORS,
            layers=2,
            dim=128,
            heads=4,
            rel_window=10,
        )
        policies = [ArgmaxModelPolicy(model=model) for _ in range(2)]

    roller = BatchRoller()
    start_time = time.time()
    transitions = roller.run(
        seeds=[0],
        policies=policies,
        player_policy_ids=[list(range(len(policies)))],
        tracker=Tracker()
    )

    print(f"Running the game took {time.time() - start_time} seconds")
    last_transition = transitions[0][-1]
    print_board(last_transition.target_state.board)
    print_scorecard(last_transition.score.scorecard)


if __name__ == "__main__":
    main()
