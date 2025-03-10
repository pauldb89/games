import collections
import random

from ticket2ride.action_utils import get_draw_card_options
from ticket2ride.actions import (
    Action,
    ActionType,
    BuildRoute,
    DrawCard,
    DrawTickets,
    Plan,
)
from ticket2ride.board import Board
from ticket2ride.card import Card
from ticket2ride.color import ANY, COLORS, Color
from ticket2ride.consts import (
    NUM_ANY_CARDS,
    NUM_COLOR_CARDS,
    NUM_INITIAL_PLAYER_CARDS,
    NUM_LAST_TURN_CARS,
)
from ticket2ride.longest_path import find_longest_path
from ticket2ride.player import Player
from ticket2ride.policies import Policy
from ticket2ride.route import ROUTES, Route
from ticket2ride.state import ObservedState, PlayerScore, Score, Transition
from ticket2ride.ticket import DrawnTickets
from ticket2ride.tracker import Tracker


def verify_card_bookkeeping(board: Board, players: list[Player]) -> None:
    card_counts: dict[Color, int] = collections.defaultdict(int)
    for player in players:
        for color, cnt in player.card_counts.items():
            card_counts[color] += cnt

    for card in board.visible_cards:
        card_counts[card.color] += 1

    for card in board.card_deck.deck + board.card_deck.discard_pile:
        card_counts[card.color] += 1

    for color in COLORS:
        assert card_counts[color] == NUM_COLOR_CARDS
    assert card_counts[ANY] == NUM_ANY_CARDS


class Environment:
    current_state: ObservedState
    board: Board
    players: list[Player]

    turn_id: int
    player_id: int
    action_type: ActionType
    game_over: bool

    final_player_id: int | None
    consecutive_card_draws: int
    longest_path_length: int

    scorecard: list[PlayerScore]
    rng: random.Random

    def __init__(self, num_players: int, seed: int = 0) -> None:
        self.num_players = num_players
        self.reset(seed)

    def reset(self, seed: int) -> ObservedState:
        self.board = Board(num_players=self.num_players, rng=random.Random(seed))
        self.players = []
        self.scorecard = []

        for player_id in range(self.num_players):
            player = Player(player_id=player_id)
            for _ in range(NUM_INITIAL_PLAYER_CARDS):
                card = self.board.card_deck.draw()
                player.card_counts[card.color] += 1
            self.players.append(player)
            self.scorecard.append(PlayerScore(player_id=player_id))

        self.turn_id = 0
        self.player_id = 0
        self.action_type = ActionType.PLAN
        self.game_over = False

        self.final_player_id = None
        self.consecutive_card_draws = 0
        self.longest_path_length = 0

        verify_card_bookkeeping(self.board, self.players)

        self.current_state = ObservedState(
            board=self.board,
            player=self.players[self.player_id],
            action_type=ActionType.PLAN,
            # TODO(pauldb): Remember to update the valid action space transition to only include drawing tickets in the first turn of the game.
            turn_id=self.turn_id,
            terminal=self.game_over,
            consecutive_card_draws=self.consecutive_card_draws,
        )
        return self.current_state

    def get_score(self) -> Score:
        if self.action_type not in (ActionType.DRAW_TICKETS, ActionType.BUILD_ROUTE):
            return Score(scorecard=self.scorecard, turn_score=PlayerScore(self.player_id))

        score = PlayerScore(player_id=self.player_id)
        player = self.players[self.player_id]
        for ticket in player.tickets:
            score.total_tickets += 1
            if player.disjoint_sets.are_connected(ticket.source_city, ticket.destination_city):
                score.ticket_points += ticket.value
                score.completed_tickets += 1
            else:
                score.ticket_points -= ticket.value

        owned_routes: list[Route] = []
        for route_info in self.board.route_ownership.values():
            if route_info.player_id == self.player_id:
                route = ROUTES[route_info.route_id]
                owned_routes.append(route)
                score.route_points += route.value
                score.owned_routes_by_length[route.length] += 1

        score.longest_path = find_longest_path(owned_routes)
        if score.longest_path > self.longest_path_length:
            self.longest_path_length = score.longest_path

        score.longest_path_bonus = score.longest_path == self.longest_path_length

        prev_score = self.scorecard[self.player_id]
        self.scorecard[self.player_id] = score
        return Score(
            scorecard=self.scorecard,
            turn_score=PlayerScore(
                player_id=self.player_id,
                route_points=score.route_points - prev_score.route_points,
                ticket_points=score.ticket_points - prev_score.ticket_points,
                completed_tickets=score.completed_tickets - prev_score.completed_tickets,
                longest_path_bonus=False, #score.longest_path_bonus and not prev_score.longest_path_bonus,
                longest_path=score.longest_path - prev_score.longest_path,
            )
        )

    def update_state(
        self,
        next_action_type: ActionType,
        drawn_tickets: DrawnTickets | None = None,
    ) -> ObservedState:
        if next_action_type != ActionType.DRAW_CARD:
            self.consecutive_card_draws = 0

        if next_action_type == ActionType.PLAN:
            if self.final_player_id is not None:
                self.game_over = self.player_id == self.final_player_id
            elif self.board.train_cars[self.player_id] <= NUM_LAST_TURN_CARS:
                self.final_player_id = self.player_id

            if self.player_id + 1 < len(self.players):
                self.player_id += 1
            else:
                self.player_id = 0
                self.turn_id += 1

        self.action_type = next_action_type
        self.current_state = ObservedState(
            board=self.board,
            player=self.players[self.player_id],
            action_type=next_action_type,
            drawn_tickets=drawn_tickets,
            # TODO(pauldb): Remember to update the valid action space transition to only include drawing tickets in the first turn of the game.
            turn_id=self.turn_id,
            terminal=self.game_over,
            consecutive_card_draws=self.consecutive_card_draws,
        )
        return self.current_state

    def transition(
        self,
        action: Action,
        next_action_type: ActionType,
        drawn_tickets: DrawnTickets | None = None,
    ) -> Transition:
        verify_card_bookkeeping(self.board, self.players)

        # We must compute the score and save a copy of the current state before advancing the state to accept the next
        # action.
        source_state = self.current_state
        score = self.get_score()
        target_state = self.update_state(next_action_type, drawn_tickets)
        return Transition(source_state=source_state, target_state=target_state, action=action, score=score)

    def step(self, action: Action) -> Transition:
        assert action.action_type == self.action_type
        assert action.player_id == self.player_id
        assert not self.game_over

        player = self.players[self.player_id]

        if action.action_type == ActionType.PLAN:
            assert isinstance(action, Plan)
            drawn_tickets = None
            if action.next_action_type == ActionType.DRAW_TICKETS:
                drawn_tickets = self.board.ticket_deck.get()

            return self.transition(
                action=action,
                next_action_type=action.next_action_type,
                drawn_tickets=drawn_tickets
            )
        elif action.action_type == ActionType.DRAW_CARD:
            assert isinstance(action, DrawCard)
            card = action.card
            if card is None:
                card = self.board.card_deck.draw()
            else:
                self.board.visible_cards.remove(card)
                self.board.reveal_cards()

            player.card_counts[card.color] += 1
            self.consecutive_card_draws += 1

            if (
                self.consecutive_card_draws <= 1
                and (action.card is None or card.color != ANY)
                and get_draw_card_options(self.board, self.consecutive_card_draws)
            ):
                return self.transition(action=action, next_action_type=ActionType.DRAW_CARD)
            else:
                return self.transition(action=action, next_action_type=ActionType.PLAN)
        elif action.action_type == ActionType.DRAW_TICKETS:
            assert isinstance(action, DrawTickets)
            player.tickets.extend(action.tickets)

            return self.transition(action=action, next_action_type=ActionType.PLAN)
        elif action.action_type == ActionType.BUILD_ROUTE:
            assert isinstance(action, BuildRoute)

            route_info = action.route_info
            assert route_info.route_id not in self.board.route_ownership
            self.board.route_ownership[route_info.route_id] = route_info

            route = ROUTES[route_info.route_id]
            player.build_route(route)
            num_regular_cards = route.length - route_info.num_any_cards
            for _ in range(num_regular_cards):
                player.card_counts[route_info.color] -= 1
                assert player.card_counts[route_info.color] >= 0
                self.board.card_deck.discard(Card(color=route_info.color))

            for _ in range(route_info.num_any_cards):
                player.card_counts[ANY] -= 1
                assert player.card_counts[ANY] >= 0
                self.board.card_deck.discard(Card(color=ANY))

            assert self.board.train_cars[player.id] >= route.length
            self.board.train_cars[player.id] -= route.length

            self.board.route_points[player.id] += route.value

            return self.transition(action=action, next_action_type=ActionType.PLAN)


class BatchRoller:
    def run(
        self,
        seeds: list[int],
        policies: list[Policy],
        player_policy_ids: list[list[int]],
        tracker: Tracker,
    ) -> list[list[Transition]]:
        assert len(seeds) == len(player_policy_ids)

        envs = [Environment(num_players=len(policy_ids)) for policy_ids in player_policy_ids]

        states = []
        episodes = list(range(len(seeds)))
        for env, seed in zip(envs, seeds):
            states.append(env.reset(seed))

        transitions: list[list[Transition]] = [[] for _ in episodes]
        while episodes:
            policies_to_episodes = collections.defaultdict(list)
            for episode_id in episodes:
                player_id = states[episode_id].player.id
                policy_id = player_policy_ids[episode_id][player_id]
                policies_to_episodes[policy_id].append(episode_id)

            episodes = []
            actions = []
            for policy_id, policy_episode_ids in policies_to_episodes.items():
                episodes.extend(policy_episode_ids)
                policy = policies[policy_id]
                with tracker.timer("t_policy_choose_action"):
                    actions.extend(policy.choose_actions([states[episode_id] for episode_id in policy_episode_ids]))

            active_episodes = []
            for episode_id, action in zip(episodes, actions):
                with tracker.timer("t_env_step"):
                    transition = envs[episode_id].step(action)
                    transitions[episode_id].append(transition)

                if not transition.target_state.terminal:
                    active_episodes.append(episode_id)

                states[episode_id] = transition.target_state

            episodes = active_episodes

        return transitions
