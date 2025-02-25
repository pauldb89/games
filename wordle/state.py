from dataclasses import dataclass

import numpy as np

from wordle.consts import EXACT_MATCH, MAX_GUESSES, WORD_LENGTH


@dataclass
class State:
    guesses: list[str]
    hints: list[list[int]]

    @property
    def win(self) -> bool:
        return self.hints and self.hints[-1] == [EXACT_MATCH] * WORD_LENGTH

    @property
    def terminal(self) -> bool:
        return len(self.hints) == MAX_GUESSES or self.win


@dataclass(frozen=True)
class Action:
    letter: str
    value: float
    win_prob: float
    mask: list[int]
    lprobs: np.ndarray | None = None