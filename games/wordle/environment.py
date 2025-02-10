import copy
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
    def win(self) -> bool:
        return self.hints and self.hints[-1] == [EXACT_MATCH] * WORD_LENGTH

    @property
    def terminal(self) -> bool:
        return len(self.hints) == MAX_GUESSES or self.win


@dataclass(frozen=True)
class Action:
    letter: str


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

    def reset(self, seed: int | None = None, secret: str | None = None) -> State:
        assert (secret is not None) != (seed is not None), "Exactly one of secret and seed must be specified"

        if secret is not None:
            self.secret = secret
        else:
            self.secret = random.Random(seed).choice(load_vocabulary(self.vocab_path))

        self.state = State(guesses=[], hints=[])
        return self.state

    def step(self, action: Action) -> State:
        state = copy.deepcopy(self.state)

        if not state.guesses or len(state.guesses[-1]) == WORD_LENGTH:
            guess = ""
        else:
            guess = state.guesses.pop()

        guess += action.letter

        if len(guess) == WORD_LENGTH:
            state.hints.append(compute_hint(self.secret, guess))
        state.guesses.append(guess)

        self.state = state
        return self.state


@dataclass
class Sample:
    state: State
    action: Action
    reward: float = 0


class Policy:
    def choose_actions(self, states: list[State]) -> list[Action]:
        return []


@dataclass(frozen=True)
class Transition:
    source_state: State
    target_state: State
    action: Action


class BatchRoller:
    def __init__(self, vocab_path: str) -> None:
        self.vocab_path = vocab_path

    def run(self, policy: Policy, seeds: list[int]) -> list[list[Transition]]:
        episodes = list(range(len(seeds)))
        envs = [Environment(self.vocab_path) for _ in episodes]
        states = [env.reset(seed) for env, seed in zip(envs, seeds)]

        transitions: list[list[Transition]] = [[] for _ in episodes]
        while episodes:
            actions = policy.choose_actions([states[episode_id] for episode_id in episodes])

            active_episodes = []
            for episode_id, action in zip(episodes, actions):
                next_state = envs[episode_id].step(action)
                transitions[episode_id].append(Transition(states[episode_id], next_state, action))
                states[episode_id] = next_state
                if not next_state.terminal:
                    active_episodes.append(episode_id)

            episodes = active_episodes

        return transitions
