import random
from strategies import (
    opening_strategy,
    vuile_was_strategy,
    toep_strategy,
    fold_strategy,
    generate_random_strategy_profile
)

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []              # Cards currently held
        self.points = 0             # Total score (game ends at 15)
        self.in_round = True        # Whether this player is still in the current round
        self.in_game = True         # False if player reaches 15 points
        self.has_folded = False     # Check if player has folded
        self.play_open = False      # Check if player start with open cards
        self.strategies = {}        # Used strategy

    def receive_hand(self, cards):
        self.hand = cards
        self.in_round = True   # Reset for new round

    def assign_random_strategies(self):
        self.strategies = generate_random_strategy_profile()

    def fold(self, round_value):
        """Player folds and takes the current round points."""
        self.points += round_value
        self.in_round = False
        print(f"{self.name} folds and receives {round_value} point(s). Now has {self.points} points.")

    def play_card(self, trick_cards=None, lead_suit=None):
        """Play a card using the assigned opening strategy."""
        if not self.hand:
            return None
        card = opening_strategy(self, lead_suit)
        self.hand.remove(card)
        print(f"{self.name} plays {card}")
        return card

    def toep(self):
        """Stub: decides whether to 'toep' or not. Will be strategy-based later."""
        return False

    def check_eliminated(self, max_points=15):
        if self.points >= max_points:
            self.in_game = False
            print(f"{self.name} has reached {self.points} points and is out of the game.")
            return True
        return False

    def __str__(self):
        return f"{self.name} (Points: {self.points})"

    def __repr__(self):
        return self.__str__()

    def play_card_following_suit(self, lead_suit):
        """Play a card, trying to follow the lead suit if possible."""
        if lead_suit:
            matching_cards = [card for card in self.hand if card.suit == lead_suit]
            if matching_cards:
                card = random.choice(matching_cards)
            else:
                card = random.choice(self.hand)  # No matching suit — throw any card
        else:
            card = random.choice(self.hand)  # First player can play any card

        self.hand.remove(card)
        print(f"{self.name} plays {card}")
        return card
    
    def should_toep(self, round_value):
        """Decide whether to toep based on current round value and hand."""
        return toep_strategy(self, round_value)

    def should_fold(self, round_value, current_turn_owner):
        """Decide whether to fold based on current hand and strategy."""
        return fold_strategy(self, round_value, current_turn_owner)
    
    def has_real_vuile_was(self):
        figure_values = ["J", "Q", "K", "A"]
        figures = [card for card in self.hand if card.rank['rank'] in figure_values]
        sevens = [card for card in self.hand if card.rank['rank'] == "7"]
        eights = [card for card in self.hand if card.rank['rank'] == "8"]
        return (len(figures) == 4) or (len(figures) == 3 and len(sevens) == 1) #or (len(figures) == 3 and len(eights) == 1) or (len(figures) == 2 and len(sevens) == 1 and len(eights) == 1)

    def declare_vuile_was(self):
        decide = vuile_was_strategy(self)
        """Decide whether to call Vuile Was (real or bluff)."""
        return decide
