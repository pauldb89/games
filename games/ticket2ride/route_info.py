from dataclasses import dataclass

from games.ticket2ride.color import ANY, Color
from games.ticket2ride.route import ROUTES


@dataclass(order=True)
class RouteInfo:
    route_id: int
    player_id: int
    color: Color
    num_any_cards: int

    def __repr__(self) -> str:
        route = ROUTES[self.route_id]
        return f"{route} with {self.color} cards and {self.num_any_cards} additional {ANY} cards"