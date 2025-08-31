import random
from strategies import (
    opening_strategy,
    vuile_was_strategy,
    toep_strategy,
    fold_strategy,
    generate_random_strategy_profile
)
from policies import NeuralQLearningPolicy, ActionEncoder
from rewards import calculate_phase_reward

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
        self.folded_at_value = None # Check at which points is folded
        self.declared_vuile_was = False # Check if player has declared vuile was
        self.toeped = False         # Check if player toeped
        self.is_learning = False    # Check if player is RL

    def receive_hand(self, cards):
        self.hand = cards
        self.in_round = True   # Reset for new round

    def assign_random_strategies(self):
        self.strategies = generate_random_strategy_profile()

    def assign_strategy(self, profile):
        """Assign a fixed strategy profile to the player."""
        self.strategies = profile

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
        return (len(figures) == 4) or (len(figures) == 3 and len(sevens) == 1) 

    def declare_vuile_was(self):
        decide = vuile_was_strategy(self)
        """Decide whether to call Vuile Was (real or bluff)."""
        return decide

### Create the Reinforced Learning Player ###
class RLPlayer(Player):
    def __init__(self, name="RLPlayer", model=NeuralQLearningPolicy):
        super().__init__(name=name)
        # RL models for each phase
        full_deck = get_full_deck()
        self.card_model = NeuralQLearningPolicy(ActionEncoder(full_deck))
        self.vw_model = NeuralQLearningPolicy(ActionEncoder([0, 1]))  # 0=no, 1=yes
        self.toep_model = NeuralQLearningPolicy(ActionEncoder([0, 1]))  # 0=no, 1=yes
        self.fold_model = NeuralQLearningPolicy(ActionEncoder([0, 1]))  # 0=no, 1=yes

        self.experience_buffers = {
            "play_card": [],
            "vuile_was": [],
            "fold": [],
            "toep": [],
            "check": []
        }
        self.model = model
        self.last_obs = None
        self.last_action = None
        self.wins = 0
        self.losses = 0
        self.games_played = 0
        self.declared_vuile_was = False  
        self.is_learning = True
        self.consecutive_folds = 0 # Check amount of consecutive folds, should never exceed 2
        self.exp_path = "rl_experience.pkl"

        # Try to load past experience
        loaded = self.card_model.load_experience(self.exp_path)
        if loaded is not None:
            self.card_model.memory = loaded

    # --- Observation ---
    def get_observation(self, phase, context):
        trick_history = context.get("trick_history", [])
        round_value = context.get("round_value", 1)
        players = context.get("players", [])
        played_cards = [(p.name, c.rank['rank'], c.suit) for t, trick in trick_history for p, c in trick]
        return {
            "phase": phase,
            "points": self.points,
            "hand": [(c.rank['rank'], c.suit) for c in self.hand],
            "open_cards": self.play_open,
            "round_value": round_value,
            "num_figures": sum(c.rank['rank'] in ['J','Q','K','A'] for c in self.hand),
            "num_7s": sum(c.rank['rank']=='7' for c in self.hand),
            "num_8s": sum(c.rank['rank']=='8' for c in self.hand),
            "num_9s": sum(c.rank['rank']=='9' for c in self.hand),
            "played_cards": played_cards,
            "num_other_players": len(players) - 1 if players else 3
        }

    # --- Vuile Was phase ---
    def declare_vuile_was(self, context=None):
        obs = self.get_observation("vuile_was_declaration", context or {})
        action = self.vw_model.predict(obs, [0, 1])
        decision = (action == 1)

        # Track flag so rewards and round logic can read it
        self.declared_vuile_was = decision

        # Save experience
        self.experience_buffers["vuile_was"].append((obs, action, 0.0, None, False))

        # Immediate shaping reward
        self.receive_reward(calculate_phase_reward("vuile_was", self, None), done=False)

        return decision


    # --- Card play ---
    def play_card(self, lead_suit=None, context=None):
        obs = self.get_observation("play_card", context or {})
        legal_cards = self.get_legal_cards(lead_suit)
        self.consecutive_folds = 0
        if not legal_cards:
            return None

        legal_card_strings = [str(c) for c in legal_cards]
        chosen_str = self.card_model.predict(obs, legal_card_strings)

        # Map back to actual Card
        chosen = next(c for c in legal_cards if str(c) == chosen_str)

        # Save experience
        self.experience_buffers["play_card"].append((obs, chosen_str, 0.0, None, False))

        # Shaping reward for playing
        self.receive_reward(calculate_phase_reward("play_card", self, None), done=False)

        # Play the card
        self.hand.remove(chosen)
        print(f"{self.name} plays {chosen}")
        return chosen

    # --- Fold decision ---
    def should_fold(self, round_value, current_player=None):
        obs = self.get_observation("fold", {})
        
        # Check if already folded twice in a row -> cannot fold again
        if self.consecutive_folds >= 2:
            fold = False
        else:
            action = self.fold_model.predict(obs, [0, 1])
            fold = (action == 1)

        if fold:
            self.has_folded = True
            self.folded_at_value = round_value
            self.consecutive_folds += 1
        else:
            # If the player chooses to play instead of folding -> reset streak
            self.consecutive_folds = 0

        # Log experience
        self.experience_buffers["fold"].append((obs, int(fold), 0.0, None, False))
        self.receive_reward(calculate_phase_reward("fold", self, None), done=False)

        return fold



    # --- Toep decision ---
    def should_toep(self, round_value):
        obs = self.get_observation("toep", {})
        action = self.toep_model.predict(obs, [0, 1])
        self.toeped = (action == 1)

        self.experience_buffers["toep"].append((obs, action, 0.0, None, False))
        self.receive_reward(calculate_phase_reward("toep", self, None), done=False)
        return self.toeped

    
    # --- Vuile was Check ---
    def decide_check(self, obs, context=None):
        action_space = ["check", "no_check"]
        action_index = self.check_model.predict(obs, action_space)

        # Store the experience
        self.experience_buffers["check"].append((obs, action_space[action_index], 0, None, False))

        self.last_obs = obs
        self.last_action = action_space[action_index]
        return action_space[action_index] == "check"

    # --- Reward assignment ---
    def receive_reward(self, reward, done):
        if self.last_obs is None or self.last_action is None:
            return
        next_obs = None
        self.experience_buffers["play_card"].append((self.last_obs, self.last_action, reward, next_obs, done))
        self.last_obs = None
        self.last_action = None

    # --- Legal cards helper ---
    def get_legal_cards(self, lead_suit):
        if not self.hand:
            return []
        if not lead_suit:
            return self.hand.copy()
        matching = [c for c in self.hand if c.suit == lead_suit]
        return matching if matching else self.hand.copy()

    def save_experience(self):
        """Delegate saving to one of the models."""
        self.card_model.save_experience(self.card_model.memory, self.exp_path)

    # --- Reset between rounds but keep learned weights ---
    def reset_for_new_round(self):
#        if not isinstance(self.experience_buffers, dict):
#            print("Warning: experience_buffers is not a dict! Reinitializing.")
#            self.experience_buffers = {
#                "play_card": [],
#                "vuile_was": [],
            #     "fold": [],
            #     "toep": [],
            #     "check": []
            # }

        #for k in self.experience_buffers.keys():
        #    self.experience_buffers[k].clear()

        # Clear transient flags
        self.last_obs = None
        self.last_action = None
        self.has_folded = False
        self.folded_at_value = None
        self.declared_vuile_was = False
        self.toeped = False

        # Create new buffers:
        # self.experience_buffers = {
        #     "play_card": [],
        #     "vuile_was": [],
        #     "fold": [],
        #     "toep": [],
        #     "check": []
        # }

        loaded = self.card_model.load_experience()
        if loaded:
            self.experience_buffers = loaded

    def save_progress(self):
        NeuralQLearningPolicy.save_experience(self=self, buffers=self.experience_buffers, path=self.exp_path)
        print(f"[DEBUG] Saved {self.card_model.memory} experiences to {self.exp_path}")
        print(f"[RLPlayer] Saved experience to {self.exp_path}")



def get_full_deck():
    suits = ["hearts", "diamonds", "clubs", "spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    return [f"{rank} of {suit}" for suit in suits for rank in ranks]
