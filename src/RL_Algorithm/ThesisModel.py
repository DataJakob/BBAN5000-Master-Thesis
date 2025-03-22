import numpy as np
import pandas as pd

import torch as th
from torch import nn, optim

from stable_baselines3 import PPO



class LSTMPolicy(nn.Module):
    """
    A PyTorch neural network model implementing an LSTM-based policy for portfolio optimization.

    Attributes:
        lstm (nn.LSTM): The LSTM layer for sequential data processing.
        fc (nn.Linear): The fully connected layer mapping LSTM output to action probabilities.
    
    Methods:
        __init__(input_size, output_size): Initializes the model architecture.
        forward(obs): Performs a forward pass through the network.
    """



    def __init__(self, input_size, output_size):
        """
        Initializes the LSTMPolicy model with an LSTM and a fully connected layer.
        
        Args:
            input_size (int): The number of features in the input observation.
            output_size (int): The number of actions (stocks) to produce weights for.
        """
        super(LSTMPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=128)
        self.fc = nn.Linear(128, output_size)



    def forward(self, obs):
        """
        Forward pass through the LSTM and fully connected layers.
        
        Args:
            obs (torch.Tensor): The input observation of shape (sequence_length, input_size).

        Returns:
            torch.Tensor: The action probabilities representing portfolio weights.
        """
        lstm_out, _ = self.lstm(obs.unsqueeze(0))
        action = th.softmax(self.fc(lstm_out[:, -1, :]), dim=1)
        return action


def train_agent(env, learning_rate=0.001, episodes=100):
    """
    Trains the LSTMPolicy model on the provided environment using a simple policy gradient method.
    
    Args:
        env (gym.Env): The portfolio environment to interact with.
        learning_rate (float): The learning rate for the optimizer.
        episodes (int): The number of training episodes.

    Returns:
        LSTMPolicy: The trained model.
    """
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




def test_agent(env, model, save_path="weights_per_step.csv"):
    """
    Tests the trained model and records portfolio weights over time.
    
    Args:
        env (gym.Env): The portfolio environment to test.
        model (LSTMPolicy): The trained model to evaluate.
        save_path (str): The file path to save the recorded weights as a CSV file.

    Returns:
        list: The list of portfolio weights at each time step during testing.
    """
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

