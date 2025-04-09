import os
import tensorflow as tf
import os

# Monkey-patch: If TensorFlow has no "io" attribute, create a dummy one with a gfile.join method.
if not hasattr(tf, "io"):
    class DummyGFile:
        @staticmethod
        def join(*args):
            return os.path.join(*args)
    dummy_io = type("DummyIO", (), {})()
    dummy_io.gfile = DummyGFile
    tf.io = dummy_io
    print("Monkey-patched tf.io.gfile.join")

import random
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace as OriginalJoypadSpace
from stable_baselines3 import PPO
from wrappers import apply_wrappers  # your custom wrappers (frame skipping, etc.)
import numpy as np
import torch
import wandb

# Initialize Weights & Biases
wandb.init(project="my-mario-project", config={"ent_coef": 0.1, "total_timesteps": 2000000})

# Set global seeds for reproducibility
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Custom JoypadSpace wrapper to remove unsupported keyword arguments ("seed", "options")
class JoypadSpaceCompat(OriginalJoypadSpace):
    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return super().reset(**kwargs)

# Use the official WandbCallback from wandb integration for SB3.
from wandb.integration.sb3 import WandbCallback

def train_model(entropy_value, total_timesteps=250000, model_name="ppo_model", seed=42):
    """
    Train PPO on Super Mario Bros with a given entropy coefficient and log metrics to Weights & Biases.
    
    Args:
      entropy_value (float): ent_coef value. Nonzero value adds an entropy bonus.
      total_timesteps (int): Number of timesteps for training.
      model_name (str): Name used for saving the model.
      seed (int): Random seed for reproducibility.
    """
    # Set random seeds.
    set_global_seed(seed)
    
    # Create the Mario environment with API compatibility.
    env = gym_super_mario_bros.make(
        'SuperMarioBros-1-1-v0',
        render_mode='human',
        apply_api_compatibility=True
    )
    # Set the seed on initial reset.
    env.reset(seed=seed)
    
    # Use our custom JoypadSpace that removes unsupported kwargs.
    env = JoypadSpaceCompat(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    
    # Instantiate PPO with log_interval=1 to log training metrics every rollout update.
    # (Note: The frequency of updates is also affected by the rollout length (n_steps).)
    model = PPO(
        "CnnPolicy", env, ent_coef=entropy_value, verbose=1, seed=seed # Log training metrics at every rollout update.
    )
    
    # Create the official WandbCallback. This callback automatically logs training metrics.
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,    # Optional: log gradients every 100 updates.
        model_save_path=model_name,
        verbose=2,
    )
    
    # Train the model with the wandb callback.
    model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    
    # Save the trained model.
    model.save(model_name)
    env.close()
    print(f"Model saved as {model_name}")

if __name__ == "__main__":
    seed = 42  # Define a random seed for reproducibility.
    
    # Train with entropy bonus (more exploratory).
    train_model(entropy_value=0.5, total_timesteps=2000000, model_name="ppo_mario_with_higher_entropy", seed=seed)
    
    # Train without entropy bonus (more deterministic).
    train_model(entropy_value=0.0, total_timesteps=2000000, model_name="ppo_mario_without_entropy", seed=seed)


# import os
# import tensorflow as tf
# import os

# # Monkey-patch: If TensorFlow has no "io" attribute, create a dummy one with a gfile.join method.
# if not hasattr(tf, "io"):
#     # Create a dummy gfile with a join method.
#     class DummyGFile:
#         @staticmethod
#         def join(*args):
#             return os.path.join(*args)
#     # Create a dummy tf.io object.
#     dummy_io = type("DummyIO", (), {})()
#     dummy_io.gfile = DummyGFile
#     tf.io = dummy_io
#     print("Monkey-patched tf.io.gfile.join")

# import random
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import RIGHT_ONLY
# from nes_py.wrappers import JoypadSpace as OriginalJoypadSpace
# from stable_baselines3 import PPO
# from wrappers import apply_wrappers  # your custom wrappers (frame skipping, etc.)
# import numpy as np
# import torch
# import wandb

# # Initialize Weights & Biases
# wandb.init(project="my-mario-project", config={"ent_coef": 0.1, "total_timesteps": 2000000})

# # Set global seeds for reproducibility
# def set_global_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Custom JoypadSpace wrapper to remove unsupported keyword arguments ("seed", "options")
# class JoypadSpaceCompat(OriginalJoypadSpace):
#     def reset(self, **kwargs):
#         kwargs.pop("seed", None)
#         kwargs.pop("options", None)
#         return super().reset(**kwargs)

# # Updated callback for logging metrics to wandb every 100 timesteps
# from stable_baselines3.common.callbacks import BaseCallback

# class WandbCallback(BaseCallback):
#     def __init__(self, log_freq=100, verbose=0):
#         super(WandbCallback, self).__init__(verbose)
#         self.log_freq = log_freq

#     def _on_step(self) -> bool:
#         # Log every self.log_freq timesteps
#         if self.num_timesteps % self.log_freq == 0:
#             log_data = {"num_timesteps": self.num_timesteps}
#             # List of expected training metric keys
#             keys = [
#                 "loss", "approx_kl", "clip_fraction", "clip_range", "entropy_loss",
#                 "explained_variance", "learning_rate", "policy_gradient_loss", "value_loss", "n_updates"
#             ]
#             for key in keys:
#                 # Log training metrics if present in self.locals
#                 if key in self.locals and self.locals[key] is not None:
#                     try:
#                         log_data[f"train/{key}"] = float(self.locals[key])
#                     except Exception:
#                         log_data[f"train/{key}"] = self.locals[key]
#             # Also log rollout metrics if available
#             if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
#                 ep_rews = [ep_info.get("r", 0) for ep_info in self.model.ep_info_buffer]
#                 ep_lens = [ep_info.get("l", 0) for ep_info in self.model.ep_info_buffer]
#                 log_data["rollout/ep_rew_mean"] = float(np.mean(ep_rews)) if ep_rews else None
#                 log_data["rollout/ep_len_mean"] = float(np.mean(ep_lens)) if ep_lens else None

#             wandb.log(log_data, step=self.num_timesteps)
#             print(f"Logged at step {self.num_timesteps}: {list(log_data.keys())}")
#         return True

# def train_model(entropy_value, total_timesteps=250000, model_name="ppo_model", seed=42):
#     """
#     Train PPO on Super Mario Bros with a given entropy coefficient and log metrics to Weights & Biases.
    
#     Args:
#       entropy_value (float): ent_coef value. Nonzero value adds an entropy bonus.
#       total_timesteps (int): Number of timesteps for training.
#       model_name (str): Name used for saving the model.
#       seed (int): Random seed for reproducibility.
#     """
#     # Set random seeds
#     set_global_seed(seed)
    
#     # Create the Mario environment with API compatibility.
#     env = gym_super_mario_bros.make(
#         'SuperMarioBros-1-1-v0', 
#         render_mode='human', 
#         apply_api_compatibility=True
#     )
#     # Set the seed on initial reset.
#     env.reset(seed=seed)
    
#     # Use our custom JoypadSpace that removes unsupported kwargs.
#     env = JoypadSpaceCompat(env, RIGHT_ONLY)
#     env = apply_wrappers(env)
    
#     # Instantiate PPO (without TensorBoard logging, using wandb instead).
#     model = PPO("CnnPolicy", env, ent_coef=entropy_value, verbose=1, seed=seed)
    
#     # Create the WandbCallback for logging metrics every 100 timesteps.
#     wandb_callback = WandbCallback(log_freq=100)
    
#     # Train the model with the WandbCallback.
#     model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    
#     # Save the trained model.
#     model.save(model_name)
#     env.close()
#     print(f"Model saved as {model_name}")

# if __name__ == "__main__":
#     seed = 42  # Define a random seed for reproducibility.
    
#     # Train with entropy bonus (more exploratory).
#     train_model(entropy_value=0.5, total_timesteps=2000000, model_name="ppo_mario_with_higher_entropy", seed=seed)
    
#     # Train without entropy bonus (more deterministic).
#     train_model(entropy_value=0.0, total_timesteps=2000000, model_name="ppo_mario_without_entropy", seed=seed)
