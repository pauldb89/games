import abc
import collections
import contextlib
import enum
import math
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, TypeAdapter
from torch import nn

from wordle.consts import AMP_ENABLED, EXACT_MATCH, LETTER_MATCH, WORD_LENGTH
from wordle.state import Action, State
from wordle.vocab import letter_index


def amp_context():
    if not AMP_ENABLED:
        return contextlib.nullcontext()

    return torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Sample:
    state: State
    action: Action
    reward: float
    win: int
    long_term_return: float = 0
    estimated_return: float = 0
    advantage: float = 0
    td_error: float = 0

    normalized_long_term_return: float = 0
    normalized_estimated_return: float = 0
    normalized_advantage: float = 0

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
        return x + self.embeddings[:, : x.size(1), :]


class BaseConfig(BaseModel):
    type: Literal["transformer", "mlp"]


class TransformerConfig(BaseConfig):
    type: Literal["transformer"] = "transformer"
    layers: int
    heads: int
    dim: int
    separate_encoder: bool


class MLPConfig(BaseConfig):
    type: Literal["mlp"] = "mlp"
    layers: int
    dim: int
    separate_encoder: bool


ModelConfig = TypeAdapter(Union[TransformerConfig, MLPConfig])


class AdvantageType(enum.StrEnum):
    MONTE_CARLO = "monte_carlo"
    BASELINE_ADJUSTED = "baseline_adjusted"
    GENERALIZED_ADVANTAGE = "generalized_advantage"


class PolicyLossType(enum.StrEnum):
    POLICY_GRADIENT = "policy_gradient"
    PPO = "ppo"


class AlgoConfig(BaseModel):
    advantage_type: AdvantageType
    # Whether to boostrap the value function or to use monte carlo returns.
    boostrap_values: bool
    policy_loss_type: PolicyLossType
    ppo_clip_coeff: float


class Model(nn.Module):
    def to(self, device: torch.device) -> None:
        self.device = device
        super().to(device)

    @abc.abstractmethod
    def compute_logits(self, states: list[State], head_masks: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        samples: list[Sample],
        algo_config: AlgoConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = []
        masks = []
        targets = []
        old_log_probs = []
        wins = []
        policy_gradient_weights = []
        value_targets = []
        for sample in samples:
            states.append(sample.state)
            masks.append(sample.action.mask)
            letter_id = letter_index(sample.action.letter)
            targets.append(letter_id)
            old_log_probs.append(sample.action.lprobs[letter_id])
            wins.append(sample.win)

            if algo_config.advantage_type == AdvantageType.MONTE_CARLO:
                policy_gradient_weights.append(sample.normalized_long_term_return)
            elif algo_config.advantage_type == AdvantageType.BASELINE_ADJUSTED:
                policy_gradient_weights.append(sample.normalized_long_term_return - sample.action.value)
            elif algo_config.advantage_type == AdvantageType.GENERALIZED_ADVANTAGE:
                policy_gradient_weights.append(sample.normalized_advantage)
            else:
                raise ValueError(f"Advantage type {algo_config.advantage_type} not supported")

            if algo_config.boostrap_values:
                value_targets.append(sample.estimated_return)
            else:
                value_targets.append(sample.normalized_long_term_return)

        logits, values, win_probs = self.compute_logits(states, masks)
        log_probs = torch.log_softmax(logits, dim=-1)

        policy_gradient_weights = torch.tensor(policy_gradient_weights, device=self.device, dtype=torch.float32)
        if algo_config.policy_loss_type == PolicyLossType.POLICY_GRADIENT:
            target_log_probs = log_probs[range(len(samples)), targets]
            policy_loss = -(policy_gradient_weights * target_log_probs).mean()
        elif algo_config.policy_loss_type == PolicyLossType.PPO:
            new_log_probs = log_probs[range(len(samples)), targets]
            old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = torch.clip(ratio, 1 - algo_config.ppo_clip_coeff, 1 + algo_config.ppo_clip_coeff)
            policy_loss = -torch.min(policy_gradient_weights * ratio, policy_gradient_weights * clipped_ratio).mean()
        else:
            raise ValueError(f"Policy loss type {algo_config.policy_loss_type} not supported")

        masks = torch.tensor(masks, device=self.device)
        masked_log_probs = torch.where(masks, log_probs, 0)
        # entropy_loss is -entropy, since we want to maximize it.
        entropy_loss = (masked_log_probs * torch.exp(masked_log_probs)).sum(dim=1).mean()

        value_targets = torch.tensor(value_targets, device=self.device, dtype=torch.float32)
        value_loss = F.mse_loss(values, value_targets)

        wins = torch.tensor(wins, device=self.device, dtype=torch.float32)
        win_loss = F.binary_cross_entropy(win_probs, wins)

        return policy_loss, value_loss, entropy_loss, win_loss


class Transformer(Model):
    def __init__(self, device: torch.device, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config

        self.start_token = nn.Parameter(torch.randn(1, config.dim) * 0.02)
        self.letter_embeddings = nn.Embedding(26, config.dim)
        self.hint_embeddings = nn.Embedding(3, config.dim)

        self.positional_encodings = PositionEncodings(dim=config.dim)

        def make_encoder() -> nn.TransformerEncoder:
            return nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=config.dim,
                    nhead=config.heads,
                    dim_feedforward=4 * config.dim,
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=config.layers,
            )

        if config.separate_encoder:
            self.policy_encoder = make_encoder()
            self.value_encoder = make_encoder()
            self.win_encoder = make_encoder()
        else:
            self.encoder = make_encoder()

        self.action_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 26, bias=False))
        self.value_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 1))
        self.win_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 1))

        self.to(device)

    def embed(self, states: list[State]) -> list[torch.Tensor]:
        seqs = []
        for state in states:
            letters = [letter_index(letter) for guess in state.guesses for letter in guess]
            hints = [feedback for hint in state.hints for feedback in hint]

            letter_seq = self.letter_embeddings(torch.tensor(letters, device=self.device, dtype=torch.int64))
            hint_seq = self.hint_embeddings(torch.tensor(hints, device=self.device, dtype=torch.int64))

            seq = torch.empty((len(letters) + len(hints) + 1, self.config.dim), device=self.device)
            idx = torch.arange(len(hints), device=self.device)

            # Create a sequence like letter, hint, letter, hint, ..., letter, hint, letter, letter, ..., letter.
            seq[0] = self.start_token
            seq[idx * 2 + 1] = letter_seq[idx]
            seq[idx * 2 + 2] = hint_seq
            seq[2 * len(hints) + 1 :] = letter_seq[len(hints) :]

            seqs.append(seq)

        return seqs

    def pad(self, seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)

        lengths = torch.tensor([seq.size(0) for seq in seqs], device=self.device)
        mask = torch.arange(x.size(1), device=self.device).expand(len(seqs), -1) >= lengths.unsqueeze(dim=1)

        return x, mask

    def compute_logits(
        self, states: list[State], head_masks: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = self.embed(states)

        features, attn_key_mask = self.pad(seqs)

        features = self.positional_encodings(features)

        if self.config.separate_encoder:
            x = self.policy_encoder(features, src_key_padding_mask=attn_key_mask)
            v = self.value_encoder(features, src_key_padding_mask=attn_key_mask)
            w = self.win_encoder(features, src_key_padding_mask=attn_key_mask)
        else:
            x = v = w = self.encoder(features, src_key_padding_mask=attn_key_mask)

        last_token_ids = torch.tensor([seq.size(0) - 1 for seq in seqs], device=self.device)
        x = x[torch.arange(len(seqs), device=self.device), last_token_ids]
        v = v[torch.arange(len(seqs), device=self.device), last_token_ids]
        w = w[torch.arange(len(seqs), device=self.device), last_token_ids]

        logits = self.action_head(x).masked_fill(~torch.tensor(head_masks, device=self.device), value=float("-inf"))
        values = self.value_head(v).squeeze(dim=1)
        win_probs = F.sigmoid(self.win_head(w)).squeeze(dim=1)
        return logits, values, win_probs


class MLP(Model):
    def __init__(self, device: torch.device, config: MLPConfig) -> None:
        super().__init__()

        self.config = config
        self.input_dim = 1 + 26 + 26 * 5 * 3

        def make_mlp() -> nn.Sequential:
            layers = nn.Sequential(
                nn.Linear(self.input_dim, config.dim),
                nn.LayerNorm(config.dim),
                nn.ReLU(),
            )
            for _ in range(config.layers - 1):
                layers.append(nn.Linear(config.dim, config.dim))
                layers.append(nn.LayerNorm(config.dim))
                layers.append(nn.ReLU())
            return layers

        if config.separate_encoder:
            self.policy_layers = make_mlp()
            self.value_layers = make_mlp()
            self.win_layers = make_mlp()
        else:
            self.layers = make_mlp()

        self.action_heads = nn.ModuleList([nn.Linear(config.dim, 26, bias=False) for _ in range(WORD_LENGTH)])
        self.value_head = nn.Linear(config.dim, 1)
        self.win_head = nn.Linear(config.dim, 1)

        self.to(device)

    def encode(self, states: list[State]) -> np.ndarray:
        # [1, 0, 0] means the letter is a candidate at the given position.
        # [0, 1, 0] means the letter is definitely the correct letter at the given position.
        # [0, 0, 1] means the letter is definitely not the correct letter at the given position.

        def get_slice(letter_index: int, position: int) -> slice:
            offset = 27 + letter_index * WORD_LENGTH * 3 + position * 3
            return slice(offset, offset + 3)

        batch_features = []
        for state in states:
            features = np.array([len(state.hints)] + [0] * 26 + [1, 0, 0] * 26 * WORD_LENGTH, dtype=np.float32)
            for guess, hint in zip(state.guesses, state.hints):
                for position, (letter, feedback) in enumerate(zip(guess, hint)):
                    features[1 + letter_index(letter)] = 1
                    if feedback == EXACT_MATCH:
                        for letter_idx in range(26):
                            features[get_slice(letter_idx, position)] = [0, 0, 1]
                        features[get_slice(letter_index(letter), position)] = [0, 1, 0]
                    elif feedback == LETTER_MATCH:
                        features[get_slice(letter_index(letter), position)] = [0, 0, 1]
                    else:
                        for pos in range(WORD_LENGTH):
                            features[get_slice(letter_index(letter), pos)] = [0, 0, 1]

            batch_features.append(features)

        return np.array(batch_features)

    def compute_logits(
        self, states: list[State], head_masks: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.encode(states), device=self.device)

        if self.config.separate_encoder:
            x = self.policy_layers(features)
            y = self.value_layers(features)
            w = self.win_layers(features)
        else:
            x = y = w = self.layers(features)

        groups = collections.defaultdict(list)
        for idx, state in enumerate(states):
            head_idx = 0 if len(state.hints) == len(state.guesses) else len(state.guesses[-1])
            groups[head_idx].append(idx)

        logits = torch.empty(size=(len(states), 26), device=self.device)
        for head_idx, batch_idxs in groups.items():
            batch_idxs = torch.tensor(batch_idxs, device=self.device)
            logits[batch_idxs] = self.action_heads[head_idx](x[batch_idxs])

        logits = logits.masked_fill(~torch.tensor(head_masks, device=self.device), value=float("-inf"))
        values = self.value_head(y).squeeze(dim=1)
        win_probs = F.sigmoid(self.win_head(w).squeeze(dim=1))

        return logits, values, win_probs
