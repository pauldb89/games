import itertools

from ticket2ride.actions import (
    ActionType,
    BuildRoute,
    DrawCard,
    DrawTickets,
    Plan,
    Prediction,
)
from ticket2ride.board import Board
from ticket2ride.card import Card
from ticket2ride.color import ANY, COLORS, EXTENDED_COLORS, Color
from ticket2ride.route import ROUTES, Route
from ticket2ride.route_info import RouteInfo
from ticket2ride.state import ObservedState
from ticket2ride.ticket import DrawnTickets, Tickets


def get_valid_actions(state: ObservedState) -> list[ActionType]:
    valid_action_types = []

    if len(state.board.ticket_deck) >= 3:
        valid_action_types.append(ActionType.DRAW_TICKETS)

    if state.turn_id > 0:
        if len(state.board.card_deck) + len(state.board.visible_cards) >= 2:
            valid_action_types.append(ActionType.DRAW_CARD)

        build_route_options = get_build_route_options(state)
        if len(build_route_options) > 0:
            valid_action_types.append(ActionType.BUILD_ROUTE)

    # If the player has no valid actions, they are forced to skip a turn. Setting action_type
    # to PLAN will pass the turn to the next player.
    if not valid_action_types:
        valid_action_types.append(ActionType.PLAN)

    return valid_action_types


def get_draw_card_options(board: Board, consecutive_card_draws: int) -> list[Card | None]:
    card_options: list[Card | None] = []
    if len(board.card_deck) >= 1:
        card_options.append(None)

    for card in board.visible_cards:
        if card in card_options:
            continue

        if card.color == ANY and consecutive_card_draws > 0:
            continue

        card_options.append(card)

    return card_options


def get_build_route_options(state: ObservedState) -> list[RouteInfo]:
    board, player = state.board, state.player

    route_options: list[RouteInfo] = []
    for route in ROUTES:
        # This route has already been built, so it's not a valid option.
        if route.id in board.route_ownership:
            continue

        # The route is too long for the number of train cars the player currently has left.
        if route.length > board.train_cars[player.id]:
            continue

        # Check if we can use locomotive cards alone to build the route.
        if ANY in player.card_counts and player.card_counts[ANY] >= route.length:
            route_options.append(
                RouteInfo(
                    route_id=route.id,
                    player_id=player.id,
                    color=ANY,
                    num_any_cards=route.length,
                )
            )

        color_options = COLORS if route.color == ANY else [route.color]
        for color in color_options:
            if (
                    color in player.card_counts
                    and player.card_counts[color] + player.card_counts.get(ANY, 0) >= route.length
            ):
                route_options.append(
                    RouteInfo(
                        route_id=route.id,
                        player_id=player.id,
                        color=color,
                        # Greedily first use cards of the given color, then use locomotives (ANY).
                        num_any_cards=max(0, route.length - player.card_counts[color]),
                    )
                )

    return route_options


def get_ticket_draw_options(tickets: DrawnTickets, is_initial_turn: bool) -> list[Tickets]:
    start = 1 if is_initial_turn else 0
    
    draw_options: list[Tickets] = []
    for k in range(start, len(tickets)):
        draw_options.extend(itertools.combinations(tickets, k+1))
    return draw_options


def generate_card_classes() -> list[Card | None]:
    return [None] + [Card(c) for c in EXTENDED_COLORS]


def generate_choose_ticket_classes() -> list[tuple[int, ...]]:
    return [x for k in range(3) for x in itertools.combinations(range(3), k+1)]


def generate_build_route_classes() -> list[tuple[Route, Color]]:
    classes = []
    for route in ROUTES:
        colors = [route.color] if route.color != ANY else COLORS
        for color in colors:
            classes.append((route, color))
    return classes


PLAN_CLASSES: list[ActionType] = [
    ActionType.DRAW_CARD,
    ActionType.DRAW_TICKETS,
    ActionType.BUILD_ROUTE,
    ActionType.PLAN,
]
DRAW_CARD_CLASSES = generate_card_classes()
CHOOSE_TICKETS_CLASSES = generate_choose_ticket_classes()
BUILD_ROUTE_CLASSES = generate_build_route_classes()


def create_valid_actions_mask(state: ObservedState) -> list[int]:
    valid_action_types = get_valid_actions(state)
    return [int(action_type in valid_action_types) for action_type in PLAN_CLASSES]


def create_draw_card_mask(state: ObservedState) -> list[int]:
    draw_options = get_draw_card_options(state.board, state.consecutive_card_draws)
    return [int(cls in draw_options) for cls in DRAW_CARD_CLASSES]


def create_draw_tickets_mask(state: ObservedState) -> list[int]:
    mask = []
    for combo in CHOOSE_TICKETS_CLASSES:
        mask.append(1 if len(combo) >= 2 or state.turn_id > 0 else 0)
    return mask


def create_build_route_mask(state: ObservedState) -> list[int]:
    build_options = get_build_route_options(state)

    valid_options = set()
    for route_info in build_options:
        valid_options.add((ROUTES[route_info.route_id], route_info.color))

    mask = []
    for cls in BUILD_ROUTE_CLASSES:
        mask.append(int(cls in valid_options))
    return mask


def create_plan_action(state: ObservedState, prediction: Prediction) -> Plan:
    return Plan(
        player_id=state.player.id,
        action_type=ActionType.PLAN,
        next_action_type=PLAN_CLASSES[prediction.class_id],
        prediction=prediction,
    )


def create_draw_card_action(state: ObservedState, prediction: Prediction) -> DrawCard:
    return DrawCard(
        player_id=state.player.id,
        action_type=ActionType.DRAW_CARD,
        card=DRAW_CARD_CLASSES[prediction.class_id],
        prediction=prediction,
    )


def create_draw_tickets_action(state: ObservedState, prediction: Prediction) -> DrawTickets:
    return DrawTickets(
        player_id=state.player.id,
        action_type=ActionType.DRAW_TICKETS,
        tickets=tuple(state.drawn_tickets[ticket_idx] for ticket_idx in CHOOSE_TICKETS_CLASSES[prediction.class_id]),
        prediction=prediction,
    )


def create_build_route_action(state: ObservedState, prediction: Prediction) -> BuildRoute:
    route, color = BUILD_ROUTE_CLASSES[prediction.class_id]
    return BuildRoute(
        player_id=state.player.id,
        action_type=ActionType.BUILD_ROUTE,
        route_info=RouteInfo(
            route_id=route.id,
            player_id=state.player.id,
            color=color,
            num_any_cards=max(0, route.length - state.player.card_counts[color]),
        ),
        prediction=prediction,
    )
