import torch
from games.wordle.model import Model, ModelConfig
from games.wordle.state import State


def test_model() -> None:
    model_config = ModelConfig(layers=1, heads=1, dim=4, share_weights=False)
    model = Model(device=torch.device("cpu"), config=model_config)

    states = [
        State(guesses=["hello", "wor"], hints=[[1, 0, 2, 0, 0]]),
        State(guesses=["h"], hints=[]),
    ]

    model.start_token.data = torch.zeros(4)
    model.letter_embeddings.weight.data = (1 + torch.arange(26, dtype=torch.float32)).unsqueeze(dim=1).repeat(1, 4)
    model.hint_embeddings.weight.data = (-(1 + torch.arange(3, dtype=torch.float32))).unsqueeze(dim=1).repeat(1, 4)

    seqs = model.embed(states)

    expected_values = [[0, 8, -2, 5, -1, 12, -3, 12, -1, 15, -1, 23, 15, 18], [0, 8]]
    for seq, values in zip(seqs, expected_values):
        assert seq.size() == torch.Size([len(values), 4])
        for idx, value in enumerate(values):
            torch.testing.assert_close(seq[idx], torch.tensor([value] * 4, dtype=torch.float32))


    x, src_mask = model.pad(seqs)
    assert x.size() == torch.Size([2, 14, 4])

    new_expected_values = []
    for seq, values in zip(x, expected_values):
        new_values = []
        for idx in range(14):
            value = 0 if idx >= len(values) else values[idx]
            torch.testing.assert_close(seq[idx], torch.tensor([value] * 4, dtype=torch.float32))
            new_values.append(value)

        new_expected_values.append(new_values)

    torch.testing.assert_close(src_mask, torch.tensor([[False] * 14, [False] * 2 + [True] * 12]))