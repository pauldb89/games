import functools
import random
from dataclasses import dataclass

from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import MAX_GUESSES
from games.wordle.consts import NO_MATCH
from games.wordle.consts import WORD_LENGTH


@functools.cache
def load_vocabulary(path: str) -> list[str]:
    words = []
    with open(path, "r") as f:
        for line in f:
            words.append(line.strip())
    return words


@dataclass
class State:
    guesses: list[str]
    hints: list[list[int]]

    @property
    def terminal(self) -> bool:
        return len(self.hints) == MAX_GUESSES


def compute_hint(secret: str, guess: str) -> list[int]:
    assert len(secret) == len(guess) == WORD_LENGTH
    hint = []
    for secret_letter, guessed_letter in zip(secret, guess):
        if secret_letter == guessed_letter:
            hint.append(EXACT_MATCH)
        elif guessed_letter in secret:
            hint.append(LETTER_MATCH)
        else:
            hint.append(NO_MATCH)

    return hint

class Environment:
    state: State
    secret: str

    def __init__(self, vocab_path: str) -> None:
        self.vocab_path = vocab_path

    def reset(self, seed: int) -> State:
        self.state = State(guesses=[], hints=[])
        self.secret = random.Random(seed).choice(load_vocabulary(self.vocab_path))

    def step(self, letter: str) -> State:
        if not self.state.guesses or len(self.state.guesses[-1]) == WORD_LENGTH:
            guess = ""
        else:
            guess = self.state.guesses.pop()

        guess += letter

        if len(guess) == WORD_LENGTH:
            self.state.hints.append(compute_hint(self.secret, guess))
        self.state.guesses.append(guess)

        return self.state
