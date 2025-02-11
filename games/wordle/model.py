from dataclasses import dataclass
import math
from torch import nn
import torch

from games.wordle.state import Action, State
from games.wordle.vocab import letter_index


def amp_context():
    return torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Sample:
    state: State
    action: Action
    reward: float = 0
    long_term_return: float = 0


class PositionEncodings(nn.Module):
    def __init__(self, dim: int, max_length: int = 200) -> None:
        super().__init__()

        positions = torch.arange(max_length).unsqueeze(dim=1)
        frequencies = torch.exp(torch.arange(0, dim, 2) * math.log(10_000) / dim)

        embeddings = torch.empty(max_length, dim)
        embeddings[:, 0::2] = torch.sin(positions / frequencies)
        embeddings[:, 1::2] = torch.cos(positions / frequencies)
        # For broadcasting along the batch dimension.
        self.register_buffer("embeddings", embeddings.unsqueeze(dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embeddings[:, :x.size(1), :]


class Model(nn.Module):
    def __init__(self, device: torch.device, layers: int, heads: int, dim: int) -> None:
        super().__init__()

        self.layers = layers
        self.heads = heads
        self.dim = dim

        self.start_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.letter_embeddings = nn.Embedding(26, dim)
        self.hint_embeddings = nn.Embedding(3, dim)

        self.positional_encodings = PositionEncodings(dim=dim)

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

        self.to(device)

    def to(self, device: torch.device) -> None:
        self.device = device
        super().to(device)

    def embed(self, states: list[State]) -> list[torch.Tensor]:
        seqs = []
        for state in states:
            letters = [letter_index(l) for guess in state.guesses for l in guess]
            hints = [feedback for hint in state.hints for feedback in hint]

            letter_seq = self.letter_embeddings(torch.tensor(letters, device=self.device, dtype=torch.int64))
            hint_seq = self.hint_embeddings(torch.tensor(hints, device=self.device, dtype=torch.int64))

            seq = torch.empty((len(letters) + len(hints) + 1, self.dim), device=self.device)
            idx = torch.arange(len(hints), device=self.device)

            # Create a sequence like letter, hint, letter, hint, ..., letter, hint, letter, letter, ..., letter.
            seq[0] = self.start_token
            seq[idx * 2 + 1] = letter_seq[idx]
            seq[idx * 2 + 2] = hint_seq
            seq[2 * len(hints)+1:] = letter_seq[len(hints):]

            seqs.append(seq)

        return seqs


    def pad(self, seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

        lengths = torch.tensor([seq.size(0) for seq in seqs], device=self.device)
        mask = torch.arange(x.size(1), device=self.device).expand(len(seqs), -1) >= lengths.unsqueeze(dim=1)

        return x, mask

    def forward(self, states: list[State], head_masks: list[list[int]]) -> torch.Tensor:
        seqs = self.embed(states)

        x, attn_key_mask = self.pad(seqs)

        x = self.positional_encodings(x)

        x = self.encoder(x, src_key_padding_mask=attn_key_mask)

        last_token_ids = torch.tensor([seq.size(0) - 1 for seq in seqs], device=self.device)
        x = x[torch.arange(len(seqs), device=self.device), last_token_ids]

        return self.head(x).masked_fill(~torch.tensor(head_masks, device=self.device), value=float("-inf"))

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
        target_log_probs = log_probs[range(len(samples)), targets]
        weights = torch.tensor(weights, device=self.device)
        return -(weights * target_log_probs).mean()