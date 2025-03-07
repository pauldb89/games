import collections
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ticket2ride.action_utils import (
    BUILD_ROUTE_CLASSES,
    CHOOSE_TICKETS_CLASSES,
    DRAW_CARD_CLASSES,
    PLAN_CLASSES,
)
from ticket2ride.actions import Action, ActionType
from ticket2ride.features import (
    FEATURE_REGISTRY,
    BatchFeatures,
    Extractor,
    Features,
    FeatureType,
)
from ticket2ride.state import ObservedState, Score


@dataclass
class Sample:
    episode_id: int
    state: ObservedState
    action: Action
    score: Score
    reward: float = 0
    long_term_return: float = 0
    estimated_return: float = 0
    advantage: float = 0


class EmbeddingTable(nn.Module):
    def __init__(self, device: torch.device, extractors: list[Extractor], dim: int) -> None:
        super().__init__()

        self.extractors = extractors
        self.device = device

        self.offsets = {}
        size = 1
        for extractor in extractors:
            for feature_type in extractor.feature_types:
                if feature_type not in self.offsets:
                    self.offsets[feature_type] = size
                    size += FEATURE_REGISTRY[feature_type].cardinality

        self.embeddings = nn.Embedding(size, dim, padding_idx=0)

        self.to(device)

    def featurize(self, states: list[ObservedState]) -> BatchFeatures:
        batch_features = []
        for state in states:
            features = []
            for extractor in self.extractors:
                features.extend(extractor.extract(state))
            batch_features.append(features)

        return batch_features

    def compute_indices(self, batch_features: BatchFeatures) -> torch.Tensor:
        batch_indices = []
        for features in batch_features:
            indices = []
            for feature in features:
                indices.append(self.offsets[feature.type] + feature.value)
            batch_indices.append(torch.tensor(indices, device=self.device))

        return torch.nn.utils.rnn.pad_sequence(batch_indices, batch_first=True, padding_value=0)

    def forward(self, states: list[ObservedState]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.featurize(states)
        indices = self.compute_indices(features)
        return self.embeddings(indices), torch.ne(indices, 0)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        self.heads = heads
        self.rel_window = rel_window
        self.rel_proj = rel_proj

        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.scalar_norm = math.sqrt(dim / heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, d = x.size()
        qkv = self.to_qkv(x).view(b, s, self.heads, -1)

        queries, keys, values = qkv.chunk(3, dim=-1)

        queries = torch.permute(queries, dims=[0, 2, 1, 3])
        keys = torch.permute(keys, dims=[0, 2, 1, 3])
        values = torch.permute(values, dims=[0, 2, 1, 3])

        offsets = (
              torch.arange(s, device=x.device).unsqueeze(dim=0)
              - torch.arange(s, device=x.device).unsqueeze(dim=1)
        ).clamp(min=-self.rel_window, max=self.rel_window) + self.rel_window
        offsets = offsets.repeat(b, self.heads, 1, 1)

        content_scores = torch.einsum("bhik,bhjk->bhij", queries, keys) / self.scalar_norm
        position_scores = torch.gather(self.rel_proj(queries), dim=-1, index=offsets)
        scores = content_scores + position_scores

        scores = scores.masked_fill(~mask.view(b, 1, 1, s), float("-inf"))
        attention = torch.softmax(scores, dim=-1)

        ret = torch.einsum("bhij,bhjk->bhik", attention, values)
        return ret.permute(dims=[0, 2, 1, 3]).reshape(b, s, d)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, rel_window: int, rel_proj: nn.Linear) -> None:
        super().__init__()

        self.attn_ln = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RelativeSelfAttention(
            dim=dim,
            heads=heads,
            rel_window=rel_window,
            rel_proj=rel_proj,
        )

        self.mlp_block = Residual(
            module=nn.Sequential(
                nn.LayerNorm(dim, eps=1e-6),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), mask)
        return self.mlp_block(x)


class Model(nn.Module):
    def __init__(
        self,
        device: torch.device,
        extractors: list[Extractor],
        layers: int,
        dim: int,
        heads: int,
        rel_window: int,
    ) -> None:
        super().__init__()

        self.device = device
        self.extractors = extractors

        self.feature_index: dict[FeatureType, int] = {}
        self.embeddings = EmbeddingTable(device, extractors, dim)

        assert (
            dim % heads == 0
        ), f"The hidden dimension {dim} is not visible by the number of heads {heads}"
        rel_proj = nn.Linear(dim // heads, 2 * rel_window + 3)

        blocks = [
            TransformerBlock(dim=dim, heads=heads, rel_window=rel_window, rel_proj=rel_proj)
            for _ in range(layers)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.value_head = nn.Linear(dim, 1)
        self.action_heads = nn.ModuleDict({
            ActionType.PLAN: nn.Linear(dim, len(PLAN_CLASSES)),
            ActionType.DRAW_CARD: nn.Linear(dim, len(DRAW_CARD_CLASSES)),
            ActionType.DRAW_TICKETS: nn.Linear(dim, len(CHOOSE_TICKETS_CLASSES)),
            ActionType.BUILD_ROUTE: nn.Linear(dim, len(BUILD_ROUTE_CLASSES)),
        })

        self.to(device)

    def featurize(self, state: ObservedState) -> Features:
        features = []
        for extractor in self.extractors:
            features.extend(extractor.extract(state))

        return features

    def forward(
        self,
        states: list[ObservedState],
        head: ActionType,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, seq_mask = self.embeddings(states)
        for block in self.blocks:
            x = block(x, seq_mask)

        x = self.norm(x)[:, 0, :]

        logits = self.action_heads[head](x)
        if mask is not None:
            logits = logits.masked_fill(mask.to(self.device) == 0, value=float("-inf"))

        values = self.value_head(x).squeeze(dim=-1)

        return logits, values

    def loss(self, samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
        grouped_samples = collections.defaultdict(list)
        for sample in samples:
            grouped_samples[sample.state.next_action].append(sample)

        policy_loss_terms = []
        values = []
        advantages = []
        returns = []
        for action_type, group in grouped_samples.items():
            states = []
            targets = []
            for sample in group:
                states.append(sample.state)
                advantages.append(sample.advantage)
                returns.append(sample.estimated_return)
                # returns.append(sample.long_term_return)
                assert sample.action.prediction is not None
                targets.append(sample.action.prediction.class_id)

            group_logits, group_values = self(states, head=action_type)
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            # Note(pauldb): The action heads have different number of classes so we can't simply stack the logits
            # and compute the policy loss once like we do for the value loss.
            policy_loss_terms.append(F.cross_entropy(group_logits, targets, reduction="none"))
            values.append(group_values)

        policy_loss_terms = torch.cat(policy_loss_terms, dim=0)
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.half)
        returns = torch.tensor(returns, device=self.device, dtype=torch.half)
        values = torch.cat(values, dim=0)

        return (policy_loss_terms * advantages).mean(), F.mse_loss(values, returns)
