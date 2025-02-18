import abc
import copy
import functools
import random
from dataclasses import dataclass
from typing import Any

from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import NO_MATCH
from games.wordle.consts import WORD_LENGTH
from games.wordle.model import amp_context
from games.wordle.policy import Policy
from games.wordle.state import Action, State
from games.wordle.vocab import Vocab



def compute_hint(secret: str, guess: str) -> list[int]:
    assert len(secret) == len(guess) == WORD_LENGTH
    hint = []
    for secret_letter, guessed_letter in zip(secret, guess):
        if secret_letter == guessed_letter:
            hint.append(EXACT_MATCH)
        else:
            hint.append(NO_MATCH)


    for secret_letter, guessed_letter in zip(secret, guess):
        if secret_letter == guessed_letter:
            continue

        for guess_idx, other_letter in enumerate(guess):
            if hint[guess_idx] == NO_MATCH and secret_letter == other_letter:
                hint[guess_idx] = LETTER_MATCH
                break

    return hint


class Environment:
    state: State
    secret: str

    def __init__(self) -> None:
        self.secret = ""
        self.state = State(guesses=[], hints=[])

    def reset(self, secret: str, state: State | None = None) -> State:
        self.secret = secret
        self.state = copy.deepcopy(state) if state is not None else State(guesses=[], hints=[])
        return self.state

    def step(self, action: Action) -> State:
        assert not self.state.terminal, "Cannot step from a terminal state, reset the environment"
        assert self.secret, "Must reset the environment before stepping"

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
    

@dataclass(frozen=True)
class Transition:
    source_state: State
    target_state: State
    action: Action


@dataclass(frozen=True)
class Rollout:
    secret: str
    transitions: list[Transition]

    @property
    def key(self) -> tuple[str]:
        end_state = self.transitions[-1].target_state
        return (self.secret,) + tuple(guess for guess in end_state.guesses)
    
    def __hash__(self) -> int:
        return hash(self.key)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Rollout):
            return False
        
        return self.key == other.key



class Roller(abc.ABC):
    @abc.abstractmethod
    def run(self, policy: Policy, seeds: list[int]) -> list[Rollout]:
        ...


class BatchRoller(Roller):
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def run(self, policy: Policy, seeds: list[int]) -> list[Rollout]:
        episodes = list(range(len(seeds)))
        envs = [Environment() for _ in episodes]
        states = [env.reset(secret=self.vocab.pick_secret(seed)) for env, seed in zip(envs, seeds)]

        transitions: list[list[Transition]] = [[] for _ in episodes]
        while episodes:
            with amp_context():
                actions = policy.choose_actions([states[episode_id] for episode_id in episodes])

            active_episodes = []
            for episode_id, action in zip(episodes, actions):
                next_state = envs[episode_id].step(action)
                transitions[episode_id].append(Transition(states[episode_id], next_state, action))
                states[episode_id] = next_state
                if not next_state.terminal:
                    active_episodes.append(episode_id)

            episodes = active_episodes

        rollouts = []
        for episode_transitions, env in zip(transitions, envs):
            rollouts.append(Rollout(transitions=episode_transitions, secret=env.secret))
        return rollouts


# class ExpansionRoller(Roller):
#     def __init__(self, vocab: Vocab, resume_step: int, num_expansions: int) -> None:
#         self.vocab = vocab
#         self.resume_step = resume_step
#         self.num_expansions = num_expansions

#     def run(self, policy: Policy, seeds: list[int]) -> list[Rollout]:
#         roller = BatchRoller(vocab=self.vocab)
#         initial_rollouts = roller.run(policy, seeds)

#         filtered_rollouts = [r for r in initial_rollouts if len(r.transitions) > self.resume_step + 1]
#         print(f"Kept {100 * len(filtered_rollouts) / len(initial_rollouts):.2f}% of initial rollouts")

#         envs = []
#         states = []
#         for rollout in filtered_rollouts:
#             for _ in range(self.num_expansions):
#                 env = Environment(vocab=Vocab(words=[rollout.secret]))
#                 state = rollout.transitions[self.resume_step].target_state
#                 states.append(state)

#                 env.reset(seed=0, state=state)
#                 envs.append(env)

#         episodes = list(range(len(envs)))
#         transitions: list[list[Transition]] = [[] for _ in episodes]

#         while episodes:
#             with amp_context():
#                 actions = policy.choose_actions([states[episode_id] for episode_id in episodes])

#             active_episodes = []
#             for episode_id, action in zip(episodes, actions):
#                 next_state = envs[episode_id].step(action)
#                 transitions[episode_id].append(Transition(states[episode_id], next_state, action))
#                 states[episode_id] = next_state
#                 if not next_state.terminal:
#                     active_episodes.append(episode_id)

#             episodes = active_episodes

#         rollouts = []
#         for episode_transitions, env in zip(transitions, envs):
#             end_state = episode_transitions[-1].target_state
#             # print(
#             #     episode_transitions[0].source_state.guesses, 
#             #     episode_transitions[0].target_state.guesses
#             # )
#             # print(end_state.guesses)
#             rollouts.append(Rollout(transitions=episode_transitions, secret=env.secret))

#         unique_rollouts = set(rollouts)
#         print(f"Rollout uniqueness ratio: {100 * len(unique_rollouts) / len(rollouts):.2f}%")

#         return rollouts
