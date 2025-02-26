import abc

import torch

from wordle.consts import WORD_LENGTH
from wordle.model import Model
from wordle.state import Action, State
from wordle.vocab import Vocab, get_letter


class Policy(abc.ABC):
    def choose_actions(self, states: list[State]) -> list[Action]:
        ...


class ModelPolicy(Policy):
    def __init__(self, model: Model, vocab: Vocab) -> None:
        self.model = model
        self.vocab = vocab

    def create_masks(self, states: list[State]) -> list[list[int]]:
        masks = []
        for state in states:
            if not state.guesses or len(state.guesses[-1]) == WORD_LENGTH:
                prefix = ""
            else:
                prefix = state.guesses[-1]

            mask = self.vocab.get_mask(prefix=prefix)
            masks.append(mask)

        return masks

    def choose_actions(self, states: list[State]) -> list[Action]:
        masks = self.create_masks(states)
        logits, values, win_probs = self.model.module.compute_logits(states, masks)
        letter_ids = self.choose_letter_ids(logits)
        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        win_probs = win_probs.detach().cpu().numpy()

        return [
            Action(
                letter=get_letter(letter_id),
                value=value,
                mask=mask,
                lprobs=lprobs,
                win_prob=win_prob
            )
            for letter_id, value, win_prob, mask, lprobs in zip(letter_ids, values, win_probs, masks, log_probs)
        ]

    @abc.abstractmethod
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        ...


class StochasticPolicy(ModelPolicy):
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        return torch.distributions.Categorical(logits=logits).sample().detach().cpu().numpy().tolist()


class ArgmaxPolicy(ModelPolicy):
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        return logits.argmax(dim=-1).detach().cpu().numpy().tolist()