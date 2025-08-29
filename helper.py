import itertools
from player import Player

def create_players(amount_of_players):
    if not 3 <= amount_of_players <= 5:
        raise ValueError("Toepen supports 3 to 5 players.")

    name_index = {
        0: "Player_A",
        1: "Player_B",
        2: "Player_C",
        3: "Player_D",
        4: "Player_E"
    }

    selected_names = list(itertools.islice(name_index.values(), amount_of_players))
    players = [Player(name) for name in selected_names]

    print(f"Created players: {[p.name for p in players]}")
    return players


