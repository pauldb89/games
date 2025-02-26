import numpy as np

from wordle.environment import Environment, Transition
from wordle.reward import Reward
from wordle.state import Action


def test_reward() -> None:
    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]
    env = Environment()
    state = env.reset(secret="bonus")

    transitions = []
    for letter in letters:
        action = Action(letter=letter, mask=[], lprobs=np.random.randn(26), win_prob=0.5, value=0.123)
        next_state = env.step(action)
        transitions.append(Transition(source_state=state, target_state=next_state, action=action))
        state = next_state

    reward = Reward(win_reward=100, no_match_reward=-2, letter_match_reward=-1, exact_match_reward=0)

    rewards = reward(transitions)
    expected_rewards = [-2, -2, -2, -1, -2, -2, 0, -1, -2, 0, 0, 0, 0, 0, 100]
    assert rewards == expected_rewards
