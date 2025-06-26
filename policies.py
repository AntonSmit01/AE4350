import random

class MyPolicy:
    def predict(self, observation, action_space):
        # For now, just pick a random legal action
        return random.choice(action_space)

class RandomPolicy:
    def predict(self, obs, action_space):
        return random.choice(action_space)