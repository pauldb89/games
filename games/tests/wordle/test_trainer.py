import numpy as np
from games.tests.wordle.mock_policy import MockPolicy
from games.wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH
from games.wordle.environment import BatchRoller
from games.wordle.model import Sample
from games.wordle.reward import Reward
from games.wordle.trainer import compute_returns, normalize_returns
from games.wordle.vocab import Vocab


def test_compute_returns():
    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]
    hints = [
        [NO_MATCH, NO_MATCH, NO_MATCH, LETTER_MATCH, NO_MATCH],
        [NO_MATCH, EXACT_MATCH, LETTER_MATCH, NO_MATCH, EXACT_MATCH],
        [EXACT_MATCH, EXACT_MATCH, EXACT_MATCH, EXACT_MATCH, EXACT_MATCH],
    ]

    vocab = Vocab(words=["bonus"])
    roller = BatchRoller(vocab=vocab)
    policy = MockPolicy(letters=letters)
    rollouts = roller.run(policy=policy, seeds=[0])
    transitions = rollouts[0].transitions

    reward = Reward(win_reward=100, no_match_reward=-2, letter_match_reward=-1, exact_match_reward=0)

    rewards = reward(transitions)

    samples = []
    for transition, reward in zip(transitions, rewards):
        samples.append(Sample(state=transition.source_state, action=transition.action, reward=reward))

    samples = normalize_returns(compute_returns(samples, return_discount=0.95))

    expected_returns = [
        -1.57985208, -1.37191614, -1.15303619, -0.92263625, -0.73411696,
        -0.48166864, -0.21593356, -0.04422637,  0.19052501,  0.4916387,
         0.70058653,  0.92053162,  1.15205276,  1.39575923,  1.65229235
    ]
    returns = [s.long_term_return for s in samples]

    np.testing.assert_allclose(returns, expected_returns)