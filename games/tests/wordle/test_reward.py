import numpy as np
from games.wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH
from games.wordle.environment import Environment, Transition
from games.wordle.reward import Reward
from games.wordle.state import Action
from games.wordle.vocab import Vocab


def test_reward() -> None:
    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]
    vocab = Vocab(words=["bonus"])
    env = Environment(vocab)
    state = env.reset(seed=0)

    transitions = []
    for letter in letters:
        action = Action(letter=letter, mask=[], lprobs=np.random.randn(26))
        next_state = env.step(action)
        transitions.append(Transition(source_state=state, target_state=next_state, action=action))
        state = next_state

    reward = Reward(win_reward=100, no_match_reward=-2, letter_match_reward=-1, exact_match_reward=0)

    rewards = reward(transitions)
    expected_rewards = [-2, -2, -2, -1, -2, -2, 0, -1, -2, 0, 0, 0, 0, 0, 100]
    assert rewards == expected_rewards