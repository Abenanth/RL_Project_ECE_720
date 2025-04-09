import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# -------------------------------
# Set Random Seeds for Reproducibility
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# Q-Network Architecture (must match training)
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# Initialize Environment and Model
# -------------------------------
# Set render_mode to "human" so you can see the game
env = gym.make("LunarLander-v3", render_mode="human")
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = QNetwork(state_dim, action_dim).to(device)

# Load the saved model weights
model_path = "lunarlander_ddqn_policy_net.pth"
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()  # Set to evaluation mode

# -------------------------------
# Testing Loop
# -------------------------------
num_test_episodes = 10
for episode in range(num_test_episodes):
    state, _ = env.reset(seed=SEED + episode)
    done = False
    episode_reward = 0.0
    
    while not done:
        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        # Greedy action selection (no exploration)
        action = q_values.argmax().item()
        
        # Take action in the environment
        state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        episode_reward += reward

    print(f"Episode {episode+1}: Total Reward = {episode_reward:.2f}")

env.close()



# import gymnasium as gym
# import torch
# import numpy as np
# import random
# import torch.nn as nn
# import torch.nn.functional as F

# # -------------------------------
# # Set Random Seeds for Reproducibility
# # -------------------------------
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # -------------------------------
# # Q-Network Architecture (must match training)
# # -------------------------------
# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_dim)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# # -------------------------------
# # Initialize Environment and Model
# # -------------------------------
# env = gym.make("LunarLander-v3", render_mode="human")
# env.action_space.seed(SEED)
# env.observation_space.seed(SEED)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# policy_net = QNetwork(state_dim, action_dim).to(device)

# # Load the saved model weights
# model_path = "lunarlander_ddqn_policy_net.pth"
# policy_net.load_state_dict(torch.load(model_path, map_location=device))
# policy_net.eval()  # Set the network to evaluation mode

# # -------------------------------
# # Testing Loop
# # -------------------------------
# num_test_episodes = 10
# total_reward = 0.0

# for episode in range(num_test_episodes):
#     state, _ = env.reset(seed=SEED + episode)
#     episode_reward = 0.0
#     done = False
    
#     while not done:
#         # Convert state to tensor
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#         # Get action values from policy network
#         with torch.no_grad():
#             q_values = policy_net(state_tensor)
#         # Select the best action (greedy)
#         action = q_values.argmax().item()
        
#         # Take the action in the environment
#         state, reward, done, truncated, info = env.step(action)
#         done = done or truncated
        
#         episode_reward += reward

#     total_reward += episode_reward
#     print(f"Episode {episode+1}: Reward: {episode_reward}")

# avg_reward = total_reward / num_test_episodes
# print(f"\nAverage Reward over {num_test_episodes} episodes: {avg_reward:.2f}")

# env.close()
