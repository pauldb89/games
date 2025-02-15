
import more_itertools
import numpy as np
from games.tests.wordle.mock_policy import MockPolicy
from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import NO_MATCH
from games.wordle.consts import WORD_LENGTH
from games.wordle.environment import BatchRoller, Environment, compute_hint
from games.wordle.state import Action, State
from games.wordle.vocab import Vocab


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

    vocab = Vocab(words=["bonus"])
    env = Environment(vocab)
    state = env.reset(seed=0)
    assert state.guesses == []
    assert state.hints == []
    for idx, letter in enumerate(letters):
        state = env.step(action=Action(letter=letter, mask=[], lprobs=np.random.randn(26)))

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
