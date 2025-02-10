from dataclasses import dataclass
from torch import nn
import torch

from games.wordle.state import Action, State
from games.wordle.vocab import letter_index


@dataclass
class Sample:
    state: State
    action: Action
    reward: float = 0
    long_term_return: float = 0


class Model(nn.Module):
    def __init__(self, device: torch.device, layers: int, heads: int, dim: int) -> None:
        super().__init__()

        self.layers = layers
        self.heads = heads
        self.dim = dim

        self.letter_embeddings = nn.Embedding(26, dim)
        self.hint_embeddings = nn.Embedding(3, dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=self.heads, 
                dim_feedforward=4 * self.dim,
                batch_first=True,
            ),
            num_layers=layers,
        )
        self.head = nn.Linear(dim, 26)

        self.device = device
        self.to(self.device)

    def embed(self, states: list[State]) -> list[torch.Tensor]:
        seqs = []
        for state in states:
            letters = [letter_index(l) for guess in state.guesses for l in guess]
            hints = [feedback for hint in state.hints for feedback in hint]

            letter_seq = self.letter_embeddings(torch.tensor(letters, device=self.device))
            hint_seq = self.hint_embeddings(torch.tensor(hints, device=self.device))

            seq = torch.empty((len(letters) + len(hints), self.dim), device=self.device)
            idx = torch.arange(len(hints), device=self.device)

            # Create a sequence like letter, hint, letter, hint, ..., letter, hint, letter, letter, ..., letter.
            seq[idx * 2] = letter_seq[idx]
            seq[idx * 2 + 1] = hint_seq
            seq[2 * len(hints):] = letter_seq[len(hints):]

            seqs.append(seq)

        return seqs


    def pad(self, seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

        lengths = torch.tensor([seq.size(0) for seq in seqs], device=self.device)
        mask = torch.arange(x.size(1), device=self.device).expand(len(seqs), -1) >= lengths.unsqueeze(dim=1)

        return x, mask

    def forward(self, states: list[State], masks: list[list[int]]) -> torch.Tensor:
        seqs = self.embed(states)

        x, mask = self.pad(seqs)
        # TODO(pauldb): Add positional encodings.

        x = self.encoder(x, src_key_padding_mask=mask)

        last_token_ids = torch.tensor([seq.size(0) - 1 for seq in seqs], device=self.device)
        x = x[torch.arange(len(seqs), device=self.device), last_token_ids]

        return self.head(x).masked_fill(~torch.tensor(masks, device=self.device), value=float("-inf"))

    def loss(self, samples: list[Sample]) -> torch.Tensor:
        states = []
        masks = []
        targets = []
        weights = []
        for sample in samples:
            states.append(sample.state)
            masks.append(sample.action.mask)
            targets.append(letter_index(sample.action.letter))
            weights.append(sample.long_term_return)

        logits = self(states, masks)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[range(len(samples)), targets].squeeze(dim=1)
        weights = torch.tensor(weights, device=self.device)
        return -(weights * target_log_probs).mean()