### This is the main file where you can play the game ###

from deck import Deck
from helper import create_players
from round import Round
from strategies import generate_random_strategy_profile
from player import Player, RLPlayer
from policies import NeuralQLearningPolicy, ActionEncoder
import time

policy = NeuralQLearningPolicy(ActionEncoder(["yes", "no"]))

# Create players
# RLPlayer as Player A
player_a = RLPlayer(name="Player A", model=policy)

# Regular players B, C, D
player_b = Player("Player B")
player_c = Player("Player C")
player_d = Player("Player D")

players = [player_a, player_b, player_c, player_d]

strategy_b = {
    "opening": "low_first",
    "vuile_was": "no_bluff",
    "toep": "never_toep",
    "fold": "never_fold",
    "check": "always_check"
}

strategy_c = {
    "opening": "high_first",
    "vuile_was": "bluff_with_3_figures_and_8",
    "toep": "10_and_9_or_higher",
    "fold": "2_cards_below_K",
    "check": "always_check"
}

strategy_d = {
    "opening": "high_same_suit_first",
    "vuile_was": "no_bluff",
    "toep": "sure_win",
    "fold": "1_card_below_K",
    "check": "never_check"
}

# Determine amount of games that is played, not included in the final project
n = 5
game_number = 0

# Assign strategies to B, C, D
players[1].assign_strategy(strategy_b)
players[2].assign_strategy(strategy_c)
players[3].assign_strategy(strategy_d)

# Assign and print strategies (except Player A)
print("\n--- Strategy Profiles ---")
for player in players[1:]:  # Skip Player A for learning
    print(f"{player.name}'s Strategy:")
    for key, value in player.strategies.items():
        print(f"  {key}: {value}")
    print()

input("Press enter to continue.")

# Player A gets a blank strategy
players[0].strategies = {}  # Leave empty 

if isinstance(players[0], RLPlayer):
    players[0].model.load_experience("rl_experience.pkl")

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
            print(f"Training {player.name} with {sum(len(v) for v in player.experience_buffers.values())} transitions")

            # Push episodic buffers into long-term memory
            for buf in player.experience_buffers.values():
                for exp in buf:
                    player.card_model.store_transition(exp)   # store into replay buffer

            # Train from replay buffer
            player.card_model.train_from_buffer(player.experience_buffers)
            player.save_progress()

            # Clear episodic buffers only (not the replay buffer!)
            for k in player.experience_buffers.keys():
                player.experience_buffers[k].clear()
                
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
                player.save_progress()

#        game_number += 1
#        if game_number == n:
        break


    # Set starter for next round
    previous_round_winner = round_instance.trick_winner
    time.sleep(2)
    print("\n--- Starting next round... ---\n")



