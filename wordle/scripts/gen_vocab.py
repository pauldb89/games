from argparse import ArgumentParser
import itertools
import os

from wordle.consts import WORD_LENGTH
from wordle.vocab import Vocab


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--letters", type=str, required=True, help="Which letters to use")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the new vocabulary")    
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="File containing the list of eligible words",
    )
    args = parser.parse_args()

    words = Vocab(path=args.vocab_path).words if args.vocab_path is not None else []

    with open(args.output_path, "w") as f:
        for letter_group in itertools.product(args.letters, repeat=WORD_LENGTH):
            word = "".join(letter_group)
            if words and word not in words:
                continue

            f.write(word + "\n")


if __name__ == "__main__":
    main()