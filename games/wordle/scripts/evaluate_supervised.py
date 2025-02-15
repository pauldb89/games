from argparse import ArgumentParser
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from games.wordle.environment import compute_hint
from games.wordle.model import Model, ModelConfig
from games.wordle.state import State
from games.wordle.vocab import Vocab


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

    vocab = Vocab(args.vocab_path)
    model = Model(device=torch.device("cpu"), config=config)
    model.eval()

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pth"), weights_only=True))
    model.to(torch.device("cuda"))

    secret = "sport"
    guesses = ["siren", "upper", "scone", "hairy", "tract"]
    hints = [compute_hint(secret=secret, guess=guess) for guess in guesses]
    for idx in range(5):
        state = State(
            guesses=guesses + ([secret[:idx]] if idx else []), 
            hints=hints,
        )

        logits = model(states=[state], head_masks=vocab.get_mask(secret[:idx]))
        probs = F.softmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        parts = []
        for idx, prob in enumerate(probs):
            letter = chr(ord('a') + idx)
            parts.append(f"{letter}: {prob:.2f}")

        print("  ".join(parts))


if __name__ == "__main__":
    main()