import numpy as np
from tests.wordle.mock_policy import MockPolicy
from wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH
from wordle.environment import BatchRoller
from wordle.model import Sample
from wordle.reward import Reward
from wordle.trainer import compute_returns, normalize_returns
from wordle.vocab import Vocab


def test_compute_returns():
    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]

    vocab = Vocab(words=["bonus"])
    roller = BatchRoller(vocab=vocab)
    policy = MockPolicy(letters=letters)
    rollouts = roller.run(policy=policy, seeds=[0])
    transitions = rollouts[0].transitions

    reward = Reward(win_reward=100, no_match_reward=-2, letter_match_reward=-1, exact_match_reward=0)

    rewards = reward(transitions)

    samples = []
    for transition, reward in zip(transitions, rewards):
        samples.append(Sample(state=transition.source_state, action=transition.action, reward=reward, win=1))

    samples = normalize_returns(compute_returns(samples, return_discount=0.95, gae_lambda=1))

    expected_returns = [
        -1.57985208, -1.37191614, -1.15303619, -0.92263625, -0.73411696,
        -0.48166864, -0.21593356, -0.04422637,  0.19052501,  0.4916387,
         0.70058653,  0.92053162,  1.15205276,  1.39575923,  1.65229235
    ]
    returns = [s.normalized_long_term_return for s in samples]

    np.testing.assert_allclose(returns, expected_returns)