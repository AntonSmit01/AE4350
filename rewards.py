# rewards.py

def calculate_phase_reward(phase, player, round_winner=None):
    """
    Small shaping rewards given at decision time.
    Outcome reward (win/lose/fold) should still be handled in apply_end_of_round_scoring.
    """
    # Flags might not exist on non-RL players, so use getattr with defaults
    has_folded       = getattr(player, "has_folded", False)
    declared_vw      = getattr(player, "declared_vuile_was", False)
    toeped           = getattr(player, "toeped", False)
    consecutive_folds = getattr(player, "consecutive_folds", 0)
    checked_vw       = getattr(player, "checked_vuile_was", False)
    vw_target        = getattr(player, "vuile_was_target", None)  # store who they checked, if applicable

    if phase == "play_card":
        # Encourage engaging with the game instead of folding autopilot
        return 0.2

    elif phase == "vuile_was":
        # Small bonus for taking the risk (bigger outcome reward handled at round end)
        return 0.5 if declared_vw else 0.0

    elif phase == "fold":
        if has_folded:
            # Base penalty for folding
            penalty = -0.3
            # Stronger penalty if folding streaks (encourage breaking the habit)
            if consecutive_folds == 1:
                penalty -= 0.2   # second fold in a row → extra negative
            elif consecutive_folds >= 2:
                penalty -= 0.5   # threefold streak (shouldn't happen often with the hard cap)
            return penalty
        return 0.0

    elif phase == "toep":
        # Encourage aggression a bit
        return 0.5 if toeped else 0.0

    elif phase == "check":
        if getattr(player, "checked_vuile_was", False):
            target = getattr(player, "vuile_was_target", None)
            if target:
                if target.declared_vuile_was and target.has_real_vuile_was():
                    # Player checked a real vuile was → bad check
                    return -1.0
                elif target.declared_vuile_was and not target.has_real_vuile_was():
                    # Player caught a bluff
                    return 1.0
            # Checked but no target info (edge case) → neutral
            return 0.0
        return 0.0

    return 0.0

