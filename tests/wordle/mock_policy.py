import numpy as np
from wordle.state import Action, State
from wordle.policy import Policy


class MockPolicy(Policy):
    def __init__(self, letters: list[str]) -> None:
        self.letters = letters
        self.idx = 0

    def choose_actions(self, states: list[State]) -> list[Action]:
        actions = [
            Action(
                letter=self.letters[self.idx],
                mask=[1] * 26,
                lprobs=np.random.randn(26),
                win_prob=0.5,
                value=0.123,
            )
            for _ in states
        ]
        self.idx += 1
        return actions