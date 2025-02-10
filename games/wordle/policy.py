import abc
import functools
import torch
from games.wordle.consts import WORD_LENGTH
from games.wordle.model import Model
from games.wordle.state import Action, State
from games.wordle.vocab import get_letter


class Policy(abc.ABC):
    def __init__(self, model: Model, vocab: list[str]) -> None:
        self.model = model
        self.vocab = vocab

    def create_masks(self, states: list[State]) -> list[list[int]]:
        masks = []
        for state in states:
            if not state.guesses or len(state.guesses[-1]) == WORD_LENGTH:
                masks.append(self.vocab.get_mask(prefix=""))
            else:
                masks.append(self.vocab.get_mask(state.guesses[-1]))

        return masks

    def choose_actions(self, states: list[State]) -> list[Action]:
        masks = self.create_masks(states)
        logits = self.model(states, masks)
        letter_ids = self.choose_letter_ids(logits)

        return [
            Action(letter=get_letter(letter_id), mask=mask) 
            for letter_id, mask in zip(letter_ids, masks)
        ]
        
    @abc.abstractmethod
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        ...
        

class SamplingPolicy(Policy):
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        return torch.distributions.Categorical(logits=logits).sample().detach().cpu().numpy().tolist()


class ArgmaxPolicy(Policy):
    def choose_letter_ids(self, logits: torch.Tensor) -> list[int]:
        return torch.argmax(dim=-1).detach().cpu().numpy().tolist()