from dataclasses import dataclass
import math
from pydantic import BaseModel
from torch import nn
import torch
import torch.nn.functional as F

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
    secret: str = ""


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
    

class ModelConfig(BaseModel):
    layers: int
    heads: int
    dim: int


class Model(nn.Module):
    def __init__(self, device: torch.device, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.start_token = nn.Parameter(torch.randn(1, config.dim) * 0.02)
        self.letter_embeddings = nn.Embedding(26, config.dim)
        self.hint_embeddings = nn.Embedding(3, config.dim)

        self.positional_encodings = PositionEncodings(dim=config.dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.heads, 
                dim_feedforward=4 * config.dim,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, 26, bias=False)
        )

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

            seq = torch.empty((len(letters) + len(hints) + 1, self.config.dim), device=self.device)
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

    def loss(self, samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        masks = []
        targets = []
        weights = []
        for sample in samples:
            states.append(sample.state)
            masks.append(sample.action.mask)
            targets.append(letter_index(sample.action.letter))
            weights.append(sample.long_term_return)

        # import ipdb; ipdb.set_trace()

        logits = self(states, masks)
        log_probs = torch.log_softmax(logits, dim=-1)

        target_log_probs = log_probs[range(len(samples)), targets]
        weights = torch.tensor(weights, device=self.device)
        policy_loss = -(weights * target_log_probs).mean()

        masks = torch.tensor(masks, device=self.device)
        masked_log_probs = torch.where(masks, log_probs, 0)
        # entropy_loss is -entropy, since we want to maximize it.
        entropy_loss = (masked_log_probs * torch.exp(masked_log_probs)).sum(dim=1).mean()
        return policy_loss, entropy_loss
        # targets = torch.tensor(targets, device=self.device)
        # weights = torch.tensor(weights, device=self.device)
        # losses = F.cross_entropy(logits, targets, reduction="none")
        # return (losses * weights).mean(), torch.tensor(0.0)        

    def supervised_loss(self, samples: list[Sample]) -> torch.Tensor:
        states = []
        masks = []
        targets = []
        weights = []
        for sample in samples:
            states.append(sample.state)
            masks.append(sample.action.mask)
            targets.append(letter_index(sample.action.letter))
            weights.append(sample.reward)

        logits = self(states, masks)
        targets = torch.tensor(targets, device=self.device)
        weights = torch.tensor(weights, device=self.device)
        losses = F.cross_entropy(logits, targets, reduction="none")
        return (losses * weights).mean()