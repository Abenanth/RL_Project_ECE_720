import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

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
env = gym.make("LunarLander-v3", render_mode="human")
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

# -------------------------------
# Actor-Critic Network for PPO
# -------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Actor head: outputs logits for each discrete action
        self.actor = nn.Linear(128, action_dim)
        # Critic head: outputs state-value
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# -------------------------------
# Compute GAE Returns and Advantages
# -------------------------------
def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    # Append a terminal value of 0 (since the episode ends)
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return returns, advantages

# -------------------------------
# PPO Update Function (fixed to use the correct device)
# -------------------------------
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_ratio, ppo_epochs, mini_batch_size, value_loss_coef, entropy_coef):
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Move tensors to the proper device
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    dataset_size = len(states)
    for _ in range(ppo_epochs):
        # Shuffle indices for mini-batch updates
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            end = start + mini_batch_size
            mini_indices = indices[start:end]
            mini_states = states[mini_indices]
            mini_actions = actions[mini_indices]
            mini_old_log_probs = old_log_probs[mini_indices]
            mini_returns = returns[mini_indices]
            mini_advantages = advantages[mini_indices]
            
            logits, values = model(mini_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mini_actions)
            ratio = torch.exp(new_log_probs - mini_old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * mini_advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mini_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), mini_returns)
            
            # Entropy bonus (encourages exploration; set entropy_coef to 0 to disable)
            entropy = dist.entropy().mean()
            
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()

# -------------------------------
# PPO Training Loop with wandb Logging
# -------------------------------
def train():
    # Initialize wandb
    wandb.init(project="lunarlander_ppo", config={
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "ppo_epochs": 4,
        "mini_batch_size": 64,
        "num_episodes": 2000,
        "max_steps": 1500,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.1,  # Set to 0.0 for no entropy bonus
    })
    config = wandb.config

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    all_rewards = []
    total_steps = 0

    for episode in range(config.num_episodes):
        # Storage for on-policy trajectories
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(config.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            done_flag = done or truncated
            
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            state = next_state
            episode_reward += reward
            total_steps += 1

            if done_flag:
                break

        # Compute returns and advantages using GAE
        returns, advantages = compute_gae(rewards, values, config.gamma, config.gae_lambda)
        advantages = np.array(advantages)
        # Normalize advantages for better training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform PPO updates on the collected trajectory
        loss = ppo_update(model, optimizer, states, actions, log_probs, returns, advantages,
                          config.clip_ratio, config.ppo_epochs, config.mini_batch_size,
                          config.value_loss_coef, config.entropy_coef)

        all_rewards.append(episode_reward)
        wandb.log({
            "episode_reward": episode_reward,
            "loss": loss,
            "episode": episode,
            "total_steps": total_steps
        })

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode+1:4d} | Average Reward: {avg_reward:6.2f}")
            wandb.log({"average_reward_10": avg_reward, "episode": episode})
            
    # Save the trained model
    torch.save(model.state_dict(), "lunarlander_ppo_model.pth")
    print("Model saved successfully!")
    wandb.save("lunarlander_ppo_model.pth")
    wandb.finish()
    return all_rewards

if __name__ == "__main__":
    train()
