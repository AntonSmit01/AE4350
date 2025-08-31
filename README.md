# AE4350
This is my repository for the Bio-inspired Intelligence and Learning for Aerospace Applications Course

You will find a breakdown of all the files and a how to play here.

--- Files ---
Strategies.txt has all strategies that I thought of per phase of the game.
deck.py makes the deck and shuffles the cards.
helper.py is the file with some helping functions.
main.py is the main file where the game can be played.
pickle_file.py can read the pickle file that is made at the end of each game with data in it.
player.py intiates the normal players and the 'learning' player.
policies.py initiates the Q learning and the actionEncoder, where the player really learns using RL.
rewards.py is the file that has the function where the player gets its rewards.
rl_experience.pkl is a pickle file that has the epxerience of the learning player stored.
round.py initiates the round of the game that is played, including all phases.
strategies.py has all pre-set strategies for the non-learning players.
