from dataclasses import dataclass

from ticket2ride.color import Color


@dataclass(frozen=True, order=True)
class Card:
    color: Color

    def __repr__(self) -> str:
        return f"{self.color}"


def render_cards(cards: list[Card]) -> str:
    return ", ".join([repr(card) for card in cards])
