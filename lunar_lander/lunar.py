import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

# -------------------------------
# Set Random Seeds for Reproducibility
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# Create the LunarLander Environment
# -------------------------------
# For training, using render_mode=None speeds up execution. Set to "human" during testing if desired.
env = gym.make("LunarLander-v3", render_mode="human")
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

# -------------------------------
# Replay Buffer for Experience Replay
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), 
                np.array(action), 
                np.array(reward, dtype=np.float32), 
                np.array(next_state), 
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------
# Q-Network Architecture
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
# Double DQN Agent with Entropy Bonus
# -------------------------------
class DDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, 
                 buffer_capacity=100000, target_update_freq=500, entropy_coef=0.01, device="cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.device = device

        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.learn_step = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a mini-batch from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)

        # Current Q values for the actions taken
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # --- Double DQN target calculation ---
        next_q_values_policy = self.policy_net(next_state)
        next_actions = next_q_values_policy.argmax(dim=1)
        next_q_values_target = self.target_net(next_state)
        next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # MSE loss between current Q value and expected Q value
        mse_loss = F.mse_loss(q_value, expected_q_value.detach())

        # --- Entropy Bonus ---
        # Compute the entropy of the softmax over Q-values to encourage exploration.
        q_policy = self.policy_net(state)
        softmax_q = F.softmax(q_policy, dim=1)
        log_softmax_q = F.log_softmax(q_policy, dim=1)
        entropy = - (softmax_q * log_softmax_q).sum(dim=1).mean()

        # Total loss: MSE loss minus entropy bonus (to maximize entropy)
        loss = mse_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

# -------------------------------
# Training Loop with wandb Logging
# -------------------------------
def train():
    # Initialize wandb
    wandb.init(project="lunarlander_ddqn", config={
        "learning_rate": 1e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "entropy_coef": 0.01,  # Increased entropy bonus
        "target_update_freq": 500,
        "num_episodes": 2000,  # Longer training
        "max_steps": 1500,     # Longer episodes
        "epsilon_start": 1.0,
        "epsilon_final": 0.01,
        "epsilon_decay": 1000, # Slower decay for prolonged exploration
    })
    config = wandb.config

    num_episodes = config.num_episodes
    max_steps = config.max_steps
    epsilon_start = config.epsilon_start
    epsilon_final = config.epsilon_final
    epsilon_decay = config.epsilon_decay
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(state_dim, action_dim, gamma=config.gamma, lr=config.learning_rate,
                      batch_size=config.batch_size, target_update_freq=config.target_update_freq,
                      entropy_coef=config.entropy_coef, device=device)

    # Log the model and gradients to wandb
    wandb.watch(agent.policy_net, log="all")

    all_rewards = []
    losses = []
    frame_idx = 0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(max_steps):
            epsilon = epsilon_by_frame(frame_idx)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            done_flag = done or truncated

            # Store the transition in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done_flag)
            state = next_state
            episode_reward += reward
            frame_idx += 1

            # Agent update from replay buffer
            loss_val = agent.update()
            if loss_val is not None:
                losses.append(loss_val)
                wandb.log({"loss": loss_val, "frame": frame_idx, "epsilon": epsilon})
            
            if done_flag:
                break

        all_rewards.append(episode_reward)
        wandb.log({"episode_reward": episode_reward, "episode": episode})
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode+1:4d} | Average Reward: {avg_reward:6.2f} | Epsilon: {epsilon:5.2f}")
            wandb.log({"average_reward_10": avg_reward, "episode": episode})

    # Save the model before finishing training
    torch.save(agent.policy_net.state_dict(), "lunarlander_ddqn_policy_net.pth")
    print("Model saved successfully!")
    wandb.save("lunarlander_ddqn_policy_net.pth")
    wandb.finish()
    return all_rewards, losses

if __name__ == "__main__":
    rewards, losses = train()



# import gymnasium as gym
# import numpy as np
# import random
# from collections import deque
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import wandb

# # -------------------------------
# # Set Random Seeds for Reproducibility
# # -------------------------------
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # -------------------------------
# # Create the LunarLander Environment
# # -------------------------------
# env = gym.make("LunarLander-v3", render_mode="human")
# env.action_space.seed(SEED)
# env.observation_space.seed(SEED)

# # -------------------------------
# # Replay Buffer for Experience Replay
# # -------------------------------
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*batch)
#         return (np.array(state), 
#                 np.array(action), 
#                 np.array(reward, dtype=np.float32), 
#                 np.array(next_state), 
#                 np.array(done, dtype=np.float32))
    
#     def __len__(self):
#         return len(self.buffer)

# # -------------------------------
# # Q-Network Architecture
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
# # Double DQN Agent with Entropy Bonus
# # -------------------------------
# class DDQNAgent:
#     def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, 
#                  buffer_capacity=100000, target_update_freq=1000, entropy_coef=0.001, device="cpu"):
#         self.action_dim = action_dim
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.entropy_coef = entropy_coef
#         self.device = device

#         self.policy_net = QNetwork(state_dim, action_dim).to(device)
#         self.target_net = QNetwork(state_dim, action_dim).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.replay_buffer = ReplayBuffer(buffer_capacity)
#         self.learn_step = 0
#         self.target_update_freq = target_update_freq

#     def select_action(self, state, epsilon):
#         if random.random() < epsilon:
#             return random.randrange(self.action_dim)
#         else:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 q_values = self.policy_net(state_tensor)
#             return q_values.argmax().item()

#     def update(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return None

#         # Sample a mini-batch from replay buffer
#         state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
#         state      = torch.FloatTensor(state).to(self.device)
#         next_state = torch.FloatTensor(next_state).to(self.device)
#         action     = torch.LongTensor(action).to(self.device)
#         reward     = torch.FloatTensor(reward).to(self.device)
#         done       = torch.FloatTensor(done).to(self.device)

#         # Current Q values for the actions taken
#         q_values = self.policy_net(state)
#         q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

#         # --- Double DQN target calculation ---
#         next_q_values_policy = self.policy_net(next_state)
#         next_actions = next_q_values_policy.argmax(dim=1)
#         next_q_values_target = self.target_net(next_state)
#         next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
#         expected_q_value = reward + self.gamma * next_q_value * (1 - done)

#         # MSE loss between current Q value and expected Q value
#         mse_loss = F.mse_loss(q_value, expected_q_value.detach())

#         # --- Entropy Bonus ---
#         # Compute the entropy of the softmax over Q-values to encourage exploration.
#         q_policy = self.policy_net(state)
#         softmax_q = F.softmax(q_policy, dim=1)
#         log_softmax_q = F.log_softmax(q_policy, dim=1)
#         entropy = - (softmax_q * log_softmax_q).sum(dim=1).mean()

#         # Total loss: MSE loss minus entropy bonus (to maximize entropy)
#         loss = mse_loss - self.entropy_coef * entropy

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Update target network periodically
#         self.learn_step += 1
#         if self.learn_step % self.target_update_freq == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())

#         return loss.item()

# # -------------------------------
# # Training Loop with wandb Logging
# # -------------------------------
# def train():
#     # Initialize wandb
#     wandb.init(project="lunarlander_ddqn", config={
#         "learning_rate": 1e-5,
#         "batch_size": 32,
#         "gamma": 0.99,
#         "entropy_coef": 0.001,
#         "target_update_freq": 1000,
#         "num_episodes": 1000,
#         "max_steps": 1000,
#         "epsilon_start": 1.0,
#         "epsilon_final": 0.01,
#         "epsilon_decay": 500,
#     })
#     config = wandb.config

#     num_episodes = config.num_episodes
#     max_steps = config.max_steps
#     epsilon_start = config.epsilon_start
#     epsilon_final = config.epsilon_final
#     epsilon_decay = config.epsilon_decay
#     epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     agent = DDQNAgent(state_dim, action_dim, gamma=config.gamma, lr=config.learning_rate, batch_size=config.batch_size,
#                       target_update_freq=config.target_update_freq, entropy_coef=config.entropy_coef, device=device)

#     # Log the model and gradients
#     wandb.watch(agent.policy_net, log="all")

#     all_rewards = []
#     losses = []
#     frame_idx = 0

#     for episode in range(num_episodes):
#         state, _ = env.reset(seed=SEED + episode)
#         episode_reward = 0

#         for step in range(max_steps):
#             epsilon = epsilon_by_frame(frame_idx)
#             action = agent.select_action(state, epsilon)
#             next_state, reward, done, truncated, info = env.step(action)
#             done_flag = done or truncated

#             # Store the transition in the replay buffer
#             agent.replay_buffer.push(state, action, reward, next_state, done_flag)
#             state = next_state
#             episode_reward += reward
#             frame_idx += 1

#             # Agent update from replay buffer
#             loss_val = agent.update()
#             if loss_val is not None:
#                 losses.append(loss_val)
#                 wandb.log({"loss": loss_val, "frame": frame_idx, "epsilon": epsilon})
            
#             if done_flag:
#                 break

#         all_rewards.append(episode_reward)
#         wandb.log({"episode_reward": episode_reward, "episode": episode})
#         if (episode + 1) % 10 == 0:
#             avg_reward = np.mean(all_rewards[-10:])
#             print(f"Episode {episode+1:4d} | Average Reward: {avg_reward:6.2f} | Epsilon: {epsilon:5.2f}")
#             wandb.log({"average_reward_10": avg_reward, "episode": episode})

#     # Save the model before finishing training
#     torch.save(agent.policy_net.state_dict(), "lunarlander_ddqn_policy_net.pth")
#     print("Model saved successfully!")
#     wandb.save("lunarlander_ddqn_policy_net.pth")
#     wandb.finish()
#     return all_rewards, losses

# if __name__ == "__main__":
#     rewards, losses = train()
