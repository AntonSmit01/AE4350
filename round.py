import time
from strategies import check_strategy

class Round:
    def __init__(self, players, deck, previous_round_winner=None):
        self.players = [p for p in players if p.in_game]
        self.deck = deck
        self.trick_winner = None
        self.previous_round_winner = previous_round_winner
        self.round_value = 1
        self.trick_history = []
        self.active_players = self.players.copy()


    def deal_cards(self, cards_per_player=4):
        hands = self.deck.deal(len(self.players), cards_per_player)
        for player, hand in zip(self.players, hands):
            player.receive_hand(hand)

    def play_trick(self):
        # Handle Vuile Was once per round (before first trick)
        if len(self.trick_history) == 0:
            self.handle_vuile_was_phase()

        # Determine start player (previous trick winner, or previous round winner if it's the first trick)
        if self.trick_winner:
            leader = self.trick_winner
        elif self.previous_round_winner:
            leader = self.previous_round_winner
        else:
            leader = self.active_players[0]  # Fallback

        # Rotate player order so leader goes first
        start_index = self.active_players.index(leader)
        ordered_players = self.active_players[start_index:] + self.active_players[:start_index]

        played_cards = []
        lead_suit = None
        current_turn_owner = leader.name  # For fold strategy logic

        for i, player in enumerate(ordered_players):
            if not player.in_round:
                continue

            # Fold logic
            if player.should_fold(self.round_value, current_turn_owner):
                player.in_round = False
                player.folded = True
                print(f" {player.name} folds and leaves the round.")
                time.sleep(1)
                continue

            # Toep logic
            if player.should_toep(self.round_value):
                print(f" {player.name} toeps! Round value increases from {self.round_value} to {self.round_value + 1}.")
                self.round_value += 1
                time.sleep(1)

            # Card play
            if i == 0:
                card = player.play_card_following_suit(None)
                lead_suit = card.suit
            else:
                card = player.play_card_following_suit(lead_suit)

            played_cards.append((player, card))
            time.sleep(0.5)  # Delay between each play

        # Determine winner among players who followed suit
        valid_cards = [(p, c) for p, c in played_cards if c.suit == lead_suit]
        if valid_cards:
            winner = max(valid_cards, key=lambda x: x[1].value)
            self.trick_winner = winner[0]
            print(f" {self.trick_winner.name} wins the trick!")
        else:
            self.trick_winner = None

        self.trick_history.append((lead_suit, played_cards))
        time.sleep(1)  # Delay after trick
        return self.trick_winner



    def handle_fold(self, player):
        player.fold(self.round_value)
        self.active_players.remove(player)

    def apply_end_of_round_scoring(self):
        for player in self.active_players:
            if player.has_folded:
                player.points += self.round_value
                print(f"{player.name} folded and gets {self.round_value} points, {player.name} now has {player.points}.")
            elif player != self.trick_winner:
                player.points += self.round_value
                print(f"{player.name} loses round and gets {self.round_value} points, {player.name} now has {player.points}.")
            else:
                print(f"{player.name} wins the round and gets 0 points, {player.name} now has {player.points}.")

    def check_for_game_end(self):
        for player in self.players:
            if player.check_eliminated():
                return True
        return False

    def toep_or_fold_phase(self):
        for player in self.active_players.copy():
            if not player.in_round:
                continue

            if player.should_toep(self.round_value):
                self.round_value += 1
                print(f"{player.name} says TOEP! ➜ Round is now worth {self.round_value} points.")
                time.sleep(0.5)

            if player.should_fold(self.round_value):
                self.handle_fold(player)
                print(f"{player.name} folds and receives {self.round_value} points.")
                time.sleep(0.5)

        # Check if only one player remains
        if len([p for p in self.active_players if p.in_round]) == 1:
            winner = [p for p in self.active_players if p.in_round][0]
            print(f"Everyone folded. {winner.name} wins the round!")
            return True  # Round is over
        return False

    def play_full_round(self):
        self.deal_cards()

        if not self.handle_vuile_was_phase():
            print("Game ended due to insufficient cards after vuile was.")
            return

        round_over = self.toep_or_fold_phase()
        if round_over:
            return

        for _ in range(4):
            round_over = self.toep_or_fold_phase()
            if round_over:
                return
            self.play_trick()

        self.apply_end_of_round_scoring()



    def handle_vuile_was_phase(self):
        print("\n=== Vuile Was Phase ===\n")
        time.sleep(1)

        players_calling = []

        # Step 1: Collect players declaring vuile was
        for player in self.active_players:
            if player.declare_vuile_was():
                print(f"{player.name} calls **VUILE WAS**!")
                player.declared_vuile_was = True
                players_calling.append(player)
                time.sleep(1)

        if not players_calling:
                print("No players called vuile was.\n")
                time.sleep(1)
                return True

        # Step 2: Let others decide whether to check
        for caller in players_calling:
            is_real = caller.has_real_vuile_was()
            #print(f"Is real = {is_real}")
            checked = False
            successful_checks = []

            print(f"\nChecking {caller.name}'s vuile was claim...")

            for p in self.active_players:
                if p == caller or not p.in_round:
                    continue

                if check_strategy(p):
                    print(f"{p.name} checks {caller.name}'s vuile was declaration...")
                    time.sleep(1)

                    if is_real:
                        #print(f"{caller.name} had a valid Vuile Was.")
                        successful_checks.append(p)
                    else:
                        print(f"{caller.name} **bluffed!**")
                        print(f"{caller.name} will play with open cards. The cards are: {caller.hand}")
                        caller.play_open = True
                        successful_checks.append(p)
                        checked = True
                        break  # Stop checking once bluff is caught
                else:
                    print(f"{p.name} trusts {caller.name}'s vuile was and does not check.")
                    time.sleep(1)

            # Step 3: Apply consequences
            if is_real:
                print(f"\n{caller.name} had a **real vuile was** with {caller.hand}.")
                self.deck.cards.extend(caller.hand)
                caller.hand = []

                if len(self.deck.cards) < 4:
                    print("Not enough cards left to deal a new hand.")
                    return False

                for _ in range(4):
                    caller.hand.append(self.deck.cards.pop())
                print(f"{caller.name} receives a new hand.")
                time.sleep(1)

                for p in successful_checks:
                    p.points += 1
                    print(f"{p.name} gains 1 point for correctly checking Vuile Was.")
                    time.sleep(1)
            elif not checked:
                print(f"\n{caller.name}'s **bluff was not caught!** No one checked.")
                time.sleep(1)

        print("\n=== End of Vuile Was Phase ===\n")
        time.sleep(1)
        return True


