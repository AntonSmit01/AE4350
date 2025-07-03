import random
import pickle

class MyPolicy:
    def predict(self, observation, action_space):
        # For now, just pick a random legal action
        return random.choice(action_space)

class RandomPolicy:
    def predict(self, obs, action_space):
        return random.choice(action_space)
    

class DebugPolicy:
    def __init__(self):
        self.memory = []

    def predict(self, obs, action_space):
        return random.choice(action_space)

    def train_from_buffer(self, buffer):
        print("Training on", len(buffer), "samples")
        self.memory.extend(buffer)  # Optional: store all experiences

        for obs, action, reward, _, _ in buffer:
            print("→", obs["phase"], "| Action:", action, "| Reward:", reward)

    def save_experience(self, path="rl_experience.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.memory, f)
        print(f"💾 Experience saved to {path}")

    def load_experience(self, path="rl_experience.pkl"):
        try:
            with open(path, "rb") as f:
                self.memory = pickle.load(f)
            print(f"🧠 Experience loaded from {path}")
        except FileNotFoundError:
            print("🧠 No previous experience found.")


### Now the real policy ###

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pickle

# --- Network definition ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Policy using Q-network ---
class NeuralQLearningPolicy:
    def __init__(self, action_encoder, input_dim=10, lr=1e-3, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_encoder = action_encoder  # Encodes actions to/from integers

        self.q_net = QNetwork(input_dim, action_encoder.num_actions)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = []  # For saving/reloading experience

    def _obs_to_tensor(self, obs):
        # Simple flattening of features (you can expand this)
        features = [
            obs["points"],
            obs["round_value"],
            obs["num_figures"],
            obs["num_7s"],
            obs["num_8s"],
            obs["num_9s"],
            obs["num_other_players"],
            int(obs["open_cards"]),
            int(obs["phase"] == "play_card"),
            len(obs["hand"])
        ]
        return torch.tensor(features, dtype=torch.float32)

    def predict(self, obs, action_space):
        state = self._obs_to_tensor(obs).unsqueeze(0)  # [1, input_dim]
        with torch.no_grad():
            q_values = self.q_net(state).squeeze(0)

        # Mask illegal actions
        legal_ids = [self.action_encoder.encode(a) for a in action_space]
        q_values_masked = torch.full_like(q_values, float('-inf'))
        q_values_masked[legal_ids] = q_values[legal_ids]

        # Epsilon-greedy choice
        if random.random() < self.epsilon:
            chosen = random.choice(action_space)
        else:
            best_id = torch.argmax(q_values_masked).item()
            chosen = self.action_encoder.decode(best_id)

        return chosen

    def train_from_buffer(self, buffer):
        print("📚 Training on", len(buffer), "samples")
        for obs, action, reward, next_obs, done in buffer:
            s = self._obs_to_tensor(obs)
            a = self.action_encoder.encode(action)
            target = reward

            if not done and next_obs:
                s_next = self._obs_to_tensor(next_obs)
                with torch.no_grad():
                    q_next = self.q_net(s_next).max().item()
                target += self.gamma * q_next

            pred = self.q_net(s)[a]
            loss = self.loss_fn(pred, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_experience(self, path="rl_experience.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.memory, f)

    def load_experience(self, path="rl_experience.pkl"):
        try:
            with open(path, "rb") as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            print("No saved experience.")


class ActionEncoder:
    def __init__(self):
        self.action_to_id = {}
        self.id_to_action = {}
        self.num_actions = 0

    def encode(self, action):
        if action not in self.action_to_id:
            self.action_to_id[action] = self.num_actions
            self.id_to_action[self.num_actions] = action
            self.num_actions += 1
        return self.action_to_id[action]

    def decode(self, idx):
        return self.id_to_action[idx]
