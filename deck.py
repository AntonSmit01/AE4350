import random

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank  # Dict with 'rank' and 'value'

    def __str__(self):
        return f"{self.rank['rank']} of {self.suit}"

    def __repr__(self):
        return self.__str__()

    @property
    def value(self):
        return self.rank["value"]


class Deck:
    def __init__(self):
        self.cards = []
        ranks = [
            {"rank": "10", "value": 8},
            {"rank": "9", "value": 7},
            {"rank": "8", "value": 6},
            {"rank": "7", "value": 5},
            {"rank": "A", "value": 4},
            {"rank": "K", "value": 3},
            {"rank": "Q", "value": 2},
            {"rank": "J", "value": 1}
        ]
        suits = ["spades", "clubs", "hearts", "diamonds"]
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))
    
    def shuffle(self):
        if len(self.cards) > 1:
            random.shuffle(self.cards)
    
    def deal(self, num_players=4, cards_per_player=4):
        hands = [[] for _ in range(num_players)]
        for _ in range(cards_per_player):
            for i in range(num_players):
                hands[i].append(self.cards.pop())
        return hands


