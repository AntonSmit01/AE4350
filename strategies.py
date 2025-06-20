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
    #print(f"{player.name} attempting vuile was: has_real={has_vw}, strategy={strat}, hand={player.hand}")

    if has_vw:
        #print(f"{player.name} declares real Vuile Was.")
        return True

    if strat == "bluff_with_3_figures_and_8":
        # Extract rank strings from the dicts
        ranks = [c.rank["rank"] for c in player.hand]

        fig_count = sum(1 for r in ranks if r in ["J", "Q", "K", "A"])
        seven_count = ranks.count("7")
        eight_count = ranks.count("8")
        nine_count = ranks.count("9")
        

        #print(f"{player.name} has {fig_count} figures, {seven_count} 7s, {eight_count} 8s, and {nine_count} 9s.")

        if fig_count == 3 and eight_count >= 1:
            #print(f"{player.name} bluffs with 3 figures + 8")
            return True

        elif fig_count == 2 and seven_count == 1 and eight_count >= 1:
            #print(f"{player.name} bluffs with 2 figures + 7 + 8")
            return True
        
        elif fig_count == 2 and seven_count == 2:
            #print(f"{player.name} bluffs with 2 figures + 2 7s ")
            return True

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


# 5. Check Vuile Was Strategy
def check_strategy(player):
    strat = player.strategies.get("check", "always_check")

    if strat == "always_check":
        if player.points >= 12:
            return False
        else:
            return True
    elif strat == "never_check":
        return False
    else:
        return True  # default fallback


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
    ],
    "check": [
        "always_check",
        "never_check"
    ]
}

def generate_random_strategy_profile():
    """Randomly generate a full strategy profile using defined pools."""
    return {
        category: random.choice(options)
        for category, options in STRATEGY_POOL.items()
    }
