
import more_itertools
import numpy as np

from tests.wordle.mock_policy import MockPolicy
from wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH, WORD_LENGTH
from wordle.environment import BatchRoller, Environment, compute_hint
from wordle.state import Action
from wordle.vocab import Vocab


def make_guesses(letters: list[str]) -> list[list[str]]:
    return ["".join(guess) for guess in more_itertools.chunked(letters, n=WORD_LENGTH)]


def test_compute_hint():
    assert compute_hint("steep", "raise") == [NO_MATCH, NO_MATCH, NO_MATCH, LETTER_MATCH, LETTER_MATCH]
    assert compute_hint("steep", "sleek") == [EXACT_MATCH, NO_MATCH, EXACT_MATCH, EXACT_MATCH, NO_MATCH]
    assert compute_hint("steep", "steep") == [EXACT_MATCH] * WORD_LENGTH
    assert compute_hint("steep", "drool") == [NO_MATCH] * WORD_LENGTH


def test_environment():
    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]
    hints = [
        [NO_MATCH, NO_MATCH, NO_MATCH, LETTER_MATCH, NO_MATCH],
        [NO_MATCH, EXACT_MATCH, LETTER_MATCH, NO_MATCH, EXACT_MATCH],
        [EXACT_MATCH, EXACT_MATCH, EXACT_MATCH, EXACT_MATCH, EXACT_MATCH],
    ]

    env = Environment()
    state = env.reset(secret="bonus")
    assert state.guesses == []
    assert state.hints == []
    for idx, letter in enumerate(letters):
        state = env.step(action=Action(letter=letter, mask=[], lprobs=np.random.randn(26), win_prob=0.5, value=0.123))

        assert state.guesses == make_guesses(letters[:idx+1])
        assert state.hints == hints[:(idx+1) // WORD_LENGTH]


def test_roller():
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

    assert len(rollouts) == 1
    assert rollouts[0].secret == "bonus"
    for idx, transition in enumerate(rollouts[0].transitions):
        assert transition.source_state.guesses == make_guesses(letters[:idx])
        assert transition.source_state.hints == hints[:idx // WORD_LENGTH]
        assert transition.target_state.guesses == make_guesses(letters[:idx+1])
        assert transition.target_state.hints == hints[:(idx+1) // WORD_LENGTH]
        assert transition.action.letter == letters[idx]
