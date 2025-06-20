### This is the main file where you can play the game ###

from deck import Deck
from helper import create_players
from round import Round
from strategies import generate_random_strategy_profile  # <-- new import
import time

# Create players
players = create_players(4)

# Assign and print strategies (except Player A)
print("\n--- Strategy Profiles ---")
for player in players[1:]:  # Skip Player A for learning
    player.strategies = generate_random_strategy_profile()
    #player.strategies["vuile_was"] = "bluff_with_3_figures_and_8"
    #player.strategies["check"] = "always_check"
    print(f"{player.name}'s Strategy:")
    for key, value in player.strategies.items():
        print(f"  {key}: {value}")
    print()

input("Press enter to continue.")

# Player A gets a blank strategy (or learning hook)
players[0].strategies = {}  # Leave empty or fill with RL placeholder if needed

previous_round_winner = None

# Main game loop
while True:
    deck = Deck()
    deck.shuffle()

    round_instance = Round(players, deck, previous_round_winner)
    round_instance.deal_cards()

    for _ in range(4):  # Each round has 4 tricks
        winner = round_instance.play_trick()

    round_instance.apply_end_of_round_scoring()

    if round_instance.check_for_game_end():
        print("🎉 Game over!")
        break

    # Set starter for next round
    previous_round_winner = round_instance.trick_winner
    time.sleep(2)
    print("\n--- Starting next round... ---\n")




## Pseudo Code ##

### This is the main file where you can play the game ###

## Pseudo Code ##

# 1. Initialize game
    # create_deck() -> list of cards
    # create_players(num_players=4) -> list of Player objects

# 2. Shuffle and deal cards to each player

# 3. Optionally: check 'vuile was'
    # if all players agree: allow card exchange

# 4. Loop over rounds
    # For each trick (4 per round):
        # Players play a card in order
        # Track trick winner
    # Determine round winner

# 5. Handle 'toep' (raising)
    # Players can choose to raise
    # Other players decide to follow or fold
    # Update lives or points accordingly

# 6. Update game state
    # track lives / remaining players

# 7. Repeat until game ends (1 player left or a max score/loss reached)

# 8. Integrate RL agent (Player A)
    # Observe state
    # Choose action
    # Receive reward
    # Train/update policy or Q-values


# Use a point tracker to keep data (per round, but also in total to see if Player A is learning)

# First make sure the game works first, after this the RL algorythm comes into play