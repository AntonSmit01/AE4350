import time
from strategies import check_strategy
from player import RLPlayer
from policies import MyPolicy

#players = [
#    RLPlayer("Bot1", model=MyPolicy()),
#    RLPlayer("Bot2", model=MyPolicy()),
#  RLPlayer("Bot3", model=MyPolicy()),
#  RLPlayer("Bot4", model=MyPolicy()),
#]

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
            context = {
                "trick_history": self.trick_history,
                "round_value": self.round_value,
                "players": self.players,
            }
            self.handle_vuile_was_phase()

        # Determine starting player
        if self.trick_winner:
            leader = self.trick_winner
        elif self.previous_round_winner:
            leader = self.previous_round_winner
        else:
            leader = self.active_players[0]  # Fallback

        # Rotate player order so leader starts first
        start_index = self.active_players.index(leader)
        ordered_players = self.active_players[start_index:] + self.active_players[:start_index]

        played_cards = []
        lead_suit = None
        current_turn_owner = leader.name  # Used for fold logic

        for i, player in enumerate(ordered_players):
            if not player.in_round:
                continue

            # FOLD LOGIC
            if i != 0 and player.should_fold(self.round_value, current_turn_owner):
                player.in_round = False
                player.has_folded = True
                player.folded_at_value = self.round_value  # Record when they folded
                print(f" {player.name} folds and leaves the round at {self.round_value} points.")
                time.sleep(1)
                continue

            # TOEP LOGIC
            if player.should_toep(self.round_value):
                print(f" {player.name} toeps! Round value increases from {self.round_value} to {self.round_value + 1}.")
                self.round_value += 1
                time.sleep(1)

            # Prepare context (for RLPlayer)
            context = {
                "trick_history": self.trick_history,
                "round_value": self.round_value,
                "players": self.players,
            }

            # CARD PLAY
            card = player.play_card(lead_suit=lead_suit)#, context=context)

            if i == 0 and card:
                lead_suit = card.suit  # First card defines the suit

            if card:
                played_cards.append((player, card))
            time.sleep(0.5)

        # SAFETY CHECK — no cards played
        if not played_cards:
            print("⚠️ No cards were played this trick (all remaining players folded).")
            self.trick_winner = None
            return None

        print(f"Played cards: {[f'{p.name}: {c}' for p, c in played_cards]}")

        # Determine trick winner based on lead suit
        valid_cards = [(p, c) for p, c in played_cards if c.suit == lead_suit]
        if valid_cards:
            winner = max(valid_cards, key=lambda x: x[1].value)
            self.trick_winner = winner[0]
            print(f"🏆 {self.trick_winner.name} wins the trick!")
        else:
            self.trick_winner = None

        self.trick_history.append((lead_suit, played_cards))
        time.sleep(1)
        return self.trick_winner


    def handle_fold(self, player):
        player.fold(self.round_value)
        self.active_players.remove(player)

    def apply_end_of_round_scoring(self):
        for player in self.active_players:
            reward = 0
            if player.has_folded:
                folded_value = player.folded_at_value or self.round_value
                player.points += folded_value
                print(f"{player.name} folded and gets {folded_value} points, now has {player.points}.")
                reward = -2
            elif player != self.trick_winner:
                player.points += self.round_value
                print(f"{player.name} loses round and gets {self.round_value} points, now has {player.points}.")
                reward = -1
            else:
                print(f"{player.name} wins the round and gets 0 points, now has {player.points}.")
                reward = 2

            # Deliver reward if RLPlayer
            if isinstance(player, RLPlayer):
                done = player.points >= 15
                player.receive_reward(reward, done)

                # Train ONCE at end of round with all experiences
                print(f"Training {player.name} on all phases...")
                player.card_model.train_from_buffer(player.experience_buffers)

                # Clear for next round
                player.reset_for_new_round()



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
        # Reset RL player(s) for a fresh round
        for player in self.players:
            if player.is_learning:   # mark Player A with is_learning=True
                player.reset_for_new_round()

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
                    print(f"{p.name} gains 1 point for checking a correct Vuile Was.")
                    time.sleep(1)
            elif not checked:
                print(f"\n{caller.name}'s **bluff was not caught!** No one checked.")
                time.sleep(1)

        print("\n=== End of Vuile Was Phase ===\n")
        time.sleep(1)
        return True




def calculate_phase_reward(phase, player, round_winner):
    if phase == "play_card":
        # Reward for winning the round
        return 1.0 if player == round_winner else -1.0

    elif phase == "vuile_was":
        # Bluffing reward: negative if declared and lost, positive if succeeded
        if player.declared_vuile_was and player == round_winner:
            return 1.0
        elif player.declared_vuile_was:
            return -1.0
        return 0.0

    elif phase == "fold":
        # Reward if folded before losing big points
        if player.has_folded:
            return -2 if player.points + player.folded_at_value < 15 else -1.0
        else:
            return 0.5 if player.points >= 15 else 0.1

    elif phase == "toep":
        # Reward if toep and win, penalty if toep and lose
        if player.toep:
            return 1.0 if player == round_winner else -1.0
        return 0.0

    return 0.0
