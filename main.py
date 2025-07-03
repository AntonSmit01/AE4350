### This is the main file where you can play the game ###

from deck import Deck
from helper import create_players
from round import Round
from strategies import generate_random_strategy_profile
from player import Player, RLPlayer
from policies import NeuralQLearningPolicy, ActionEncoder
import time
import pickle

encoder = ActionEncoder()
policy = NeuralQLearningPolicy(action_encoder=encoder)

# Create players
# RLPlayer as Player A
player_a = RLPlayer(name="Player A", model=policy)

# Regular players B, C, D
player_b = Player("Player B")
player_c = Player("Player C")
player_d = Player("Player D")

players = [player_a, player_b, player_c, player_d]

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

if isinstance(players[0], RLPlayer):
    players[0].model.load_experience("experience.pkl")

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

    # TRAIN RL PLAYER HERE
    for player in players:
        if isinstance(player, RLPlayer):
            print(f"📚 Training {player.name} in round...")
            player.model.train_from_buffer(player.experience_buffer)
            player.experience_buffer.clear()

    if round_instance.check_for_game_end():
        print(" Game over!")
    
        for player in players:
            if isinstance(player, RLPlayer):
                player.games_played += 1
                if player.points < 15:
                    player.wins += 1
                else:
                    player.losses += 1

                print(f" {player.name} - Games: {player.games_played}, Wins: {player.wins}, Losses: {player.losses}")

                # Train
                player.model.train_from_buffer(player.experience_buffer)
                player.experience_buffer.clear()
                player.model.save_experience("rl_experience.pkl")

        break


    # Set starter for next round
    previous_round_winner = round_instance.trick_winner
    time.sleep(2)
    print("\n--- Starting next round... ---\n")



