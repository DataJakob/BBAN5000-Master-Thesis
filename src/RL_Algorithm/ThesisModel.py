import numpy as np

import torch as th
from torch import nn, optim

from stable_baselines3 import PPO



class LSTMPolicy(nn.Module):



    def __init__(self, input_size, output_size):
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, output_size)



    def forward(self, obs):
        lstm_out, _ = self.lstm(obs.unsqueeze(0))
        action = th.softmax(self.fc(lstm_out[:, -1, :]), dim=1)
        return action
    


def train_agent(env, learning_rate=0.001, episodes=100):
    model = LSTMPolicy(input_size=env.observation_space.shape[1], output_size=env.action_space.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(episodes):
        obs, _ = env.reset()
        obs = th.tensor(obs, dtype=th.float32)
        done = False
        episode_rewards = []

        while not done:
            action_probs = model(obs)
            action = action_probs.squeeze().detach().numpy()  # Get the action as a numpy array

            obs, reward, done, _, _ = env.step(action)  # Pass the action directly to the environment
            obs = th.tensor(obs, dtype=th.float32)

            episode_rewards.append(reward)

        # Policy Gradient Update
        total_reward = th.tensor(sum(episode_rewards), requires_grad=True)  # Convert to PyTorch tensor

        loss = -total_reward  # Simple loss function to maximize rewards

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode} - Total Reward: {total_reward.item()}")

    return model



import pandas as pd

def test_agent(env, model, save_path="weights_per_step.csv"):
    weights_per_step_test = []
    time_steps = []

    obs, _ = env.reset()
    obs = th.tensor(obs, dtype=th.float32)
    done = False
    total_reward = 0
    step = 0  # Track the step index for saving purposes

    while not done:
        with th.no_grad():  # Disable gradient calculation for testing
            action_probs = model(obs)

        action = action_probs.squeeze().numpy()
        obs, reward, done, _, _ = env.step(action)
        obs = th.tensor(obs, dtype=th.float32)

        # Store the action probabilities (portfolio weights)
        weights_per_step_test.append(action)
        time_steps.append(step)
        
        total_reward += reward
        step += 1

    # Save weights to a CSV file
    df = pd.DataFrame(weights_per_step_test, columns=[f"Stock_{i}" for i in range(len(action))])
    df.insert(0, "Time_Step", time_steps)
    df.to_csv(save_path, index=False)

    print(f"Total Test Reward: {total_reward}")
    print(f"Weights saved to {save_path}")
    return weights_per_step_test
