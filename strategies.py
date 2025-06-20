import random

# 1. Opening Strategies
def opening_strategy(player, lead_suit):
    strat = player.strategies.get("opening", "low_first")

    if lead_suit:
        # Must follow suit
        valid_cards = [card for card in player.hand if card.suit == lead_suit]
        if not valid_cards:
            valid_cards = player.hand
    else:
        valid_cards = player.hand

    if strat == "low_first":
        card = min(valid_cards, key=lambda c: c.value)
    elif strat == "high_first":
        card = max(valid_cards, key=lambda c: c.value)
    elif strat == "high_same_suit_first":
        suit_counts = count_suits(player.hand)
        most_common_suit = max(suit_counts, key=suit_counts.get)
        suit_cards = [c for c in valid_cards if c.suit == most_common_suit]
        card = max(suit_cards, key=lambda c: c.value) if suit_cards else random.choice(valid_cards)
    elif strat == "play_suit_first":
        suit_counts = count_suits(player.hand)
        max_suit = max(suit_counts, key=suit_counts.get)
        suit_cards = [c for c in valid_cards if c.suit == max_suit]
        card = min(suit_cards, key=lambda c: c.value) if suit_cards else random.choice(valid_cards)
    elif strat == "play_unique_suit":
        suit_counts = count_suits(player.hand)
        min_suit = min(suit_counts, key=suit_counts.get)
        suit_cards = [c for c in valid_cards if c.suit == min_suit]
        card = min(suit_cards, key=lambda c: c.value) if suit_cards else random.choice(valid_cards)
    else:
        card = random.choice(valid_cards)

    return card


# 2. Vuile Was Strategy
def vuile_was_strategy(player):
    strat = player.strategies.get("vuile_was", "no_bluff")

    has_vw = player.has_real_vuile_was()

    if has_vw:
        return True

    # Handle bluffing
    if strat == "always_bluff":
        return True
    elif strat == "bluff_with_3_figures_and_8":
        fig_count = sum(1 for c in player.hand if c.rank in ["J", "Q", "K", "A"])
        has_8 = any(c.rank == "8" for c in player.hand)
        return fig_count == 3 and has_8
    else:
        return False


# 3. Toep Strategy
def toep_strategy(player, round_value):
    strat = player.strategies.get("toep", "never_toep")

    card_values = sorted([c.rank["value"] for c in player.hand], reverse=True)

    if strat == "10_and_9_or_higher":
        return 8 in card_values and any(v >= 7 for v in card_values if v != 8)
    
    elif strat == "sure_win":
        # naive check: only toep if only one card left and it's the highest
        return len(player.hand) == 1 and player.hand[0].rank["value"] == 8
    
    else:
        return False



# 4. Fold Strategy
def fold_strategy(player, round_value, current_turn_owner):
    strat = player.strategies.get("fold", "never_fold")
    card_values = [c.rank["value"] for c in player.hand]

    if strat == "fold_with_only_Js":
        return all(c.rank == "J" for c in player.hand) and player.name != current_turn_owner

    elif strat == "2_cards_below_K":
        return len(player.hand) == 2 and all(v <= 3 for v in card_values)

    elif strat == "1_card_below_K":
        return len(player.hand) == 1 and card_values[0] <= 3

    else:
        return False


# Helper
def count_suits(cards):
    suits = {}
    for c in cards:
        suits[c.suit] = suits.get(c.suit, 0) + 1
    return suits


# Dictionary of all possible strategies
STRATEGY_POOL = {
    "opening": [
        "low_first",
        "high_first",
        "high_same_suit_first",
        "play_suit_first",
        "play_unique_suit"
    ],
    "vuile_was": [
        "no_bluff",
        "always_bluff",
        "bluff_with_3_figures_and_8"
    ],
    "toep": [
        "never_toep",
        "10_and_9_or_higher",
        "sure_win"
    ],
    "fold": [
        "never_fold",
        "fold_with_only_Js",
        "2_cards_below_K",
        "1_card_below_K"
    ]
}

def generate_random_strategy_profile():
    """Randomly generate a full strategy profile using defined pools."""
    return {
        category: random.choice(options)
        for category, options in STRATEGY_POOL.items()
    }
