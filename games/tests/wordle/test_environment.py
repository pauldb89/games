
import more_itertools
import numpy as np
from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import NO_MATCH
from games.wordle.consts import WORD_LENGTH
from games.wordle.environment import Environment, compute_hint
from games.wordle.state import Action
from games.wordle.vocab import Vocab


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

        guesses = ["".join(guess) for guess in more_itertools.chunked(letters[:idx+1], n=WORD_LENGTH)]
        assert state.guesses == guesses
        assert state.hints == hints[:(idx+1) // WORD_LENGTH]
