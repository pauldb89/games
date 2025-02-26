from wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH
from wordle.environment import Transition


class Reward:
    def __init__(
        self,
        win_reward: float,
        no_match_reward: float,
        letter_match_reward: float,
        exact_match_reward: float,
    ):
        self.win_reward = win_reward
        self.no_match_reward = no_match_reward
        self.letter_match_reward = letter_match_reward
        self.exact_match_reward = exact_match_reward

    def __call__(self, transitions: list[Transition]) -> list[float]:
        rewards = []
        next_hint = None
        for t in reversed(transitions):
            if len(t.target_state.hints) == len(t.target_state.guesses):
                 next_hint = t.target_state.hints[-1]

            assert next_hint is not None, transitions
            hint = next_hint[len(t.target_state.guesses[-1]) - 1]
            if hint == NO_MATCH:
                reward = self.no_match_reward
            elif hint == LETTER_MATCH:
                reward = self.letter_match_reward
            else:
                assert hint == EXACT_MATCH
                reward = self.exact_match_reward
            if t.target_state.win:
                reward += self.win_reward

            rewards.append(reward)

        return list(reversed(rewards))


