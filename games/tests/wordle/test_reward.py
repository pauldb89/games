from games.wordle.consts import EXACT_MATCH, LETTER_MATCH, NO_MATCH
from games.wordle.environment import Action, Environment, Transition
from games.wordle.reward import Reward


def test_reward() -> None:
    env = Environment(vocab_path="")
    env.reset(secret="bonus")

    letters = ["r", "a", "i", "s", "e", "s", "o", "u", "l", "s", "b", "o", "n", "u", "s"]
    env = Environment(vocab_path="")
    state = env.reset(secret="bonus")

    transitions = []
    for letter in letters:
        action = Action(letter=letter)
        next_state = env.step(action)
        transitions.append(Transition(source_state=state, target_state=next_state, action=action))
        state = next_state

    reward = Reward(win_reward=100, no_match_reward=-2, letter_match_reward=-1, exact_match_reward=0)

    rewards = reward(transitions)
    expected_rewards = [-2, -2, -2, -1, -2, -1, 0, -1, -2, 0, 0, 0, 0, 0, 100]
    print(f"{expected_rewards=}")
    assert rewards == expected_rewards