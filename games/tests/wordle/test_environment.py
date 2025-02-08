from games.wordle.consts import EXACT_MATCH
from games.wordle.consts import LETTER_MATCH
from games.wordle.consts import NO_MATCH
from games.wordle.consts import WORD_LENGTH
from games.wordle.environment import compute_hint


def test_compute_hint():
    assert compute_hint("steep", "raise") == [NO_MATCH, NO_MATCH, NO_MATCH, LETTER_MATCH, LETTER_MATCH]
    assert compute_hint("steep", "sleek") == [EXACT_MATCH, NO_MATCH, EXACT_MATCH, EXACT_MATCH, NO_MATCH]
    assert compute_hint("steep", "steep") == [EXACT_MATCH] * WORD_LENGTH
    assert compute_hint("steep", "drool") == [NO_MATCH] * WORD_LENGTH
