import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList

TIMESTEPS = 5_000_000

# Base Directory
base_dir = 'C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4TH YEAR PROJECT'
# Directory that the logs are assigned
logdir = "logs"



# Function to evaluate the model
def evaluate_model(model, env, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()  # Reset the environment
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)  # Get action from the model
            obs, reward, done, _ = env.step(action)  # Take action in the environment
            total_reward += reward  # Accumulate total reward for the episode
        episode_rewards.append(total_reward)  # Append total reward of the episode
    mean_reward = np.mean(episode_rewards)  # Calculate mean reward across episodes
    return mean_reward, episode_rewards

# Function to train the model
def train_model(algorithm, env, model_dir, tb_log_name, total_timesteps):
    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(base_dir, model_dir + '_best'),
                                 log_path=logdir, eval_freq=10000, n_eval_episodes=10,
                                 deterministic=True, render=False, callback_after_eval=reward_threshold_callback)
    model = algorithm("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    callback = CallbackList([eval_callback])
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name, callback=callback)
    return model

# Define your bipedal walker environment
env = gym.make("BipedalWalker-v3")  # Create environment
env = DummyVecEnv([lambda: env])  # Wrap the environment in a vectorized environment

max_stages = 2  # Maximum number of stages

# TensorBoard logging directory
tensorboard_log_dir = "tensorboard_logs_CL"

# Calculate the timesteps per stage
timesteps_per_stage = TIMESTEPS // max_stages

# Initialize PPO model with TensorBoard logging
model = PPO(
    policy='MlpPolicy',  # Policy architecture for the model
    env=env,  # Environment for the model
    tensorboard_log=tensorboard_log_dir  # Specify the directory for TensorBoard logs
)

# Curriculum learning loop
for stage in range(max_stages):
    print(f"Training Stage {stage + 1}")
    
    # Train the model for the current stage
    model = train_model(PPO, env, f"model_stage_{stage}", f"Stage_{stage}", timesteps_per_stage)
    
    # Evaluate performance on the current stage
    mean_reward, _ = evaluate_model(model, env, num_episodes=5)
    print(f"Mean reward on Stage {stage + 1}: {mean_reward}")
    
    # Load model parameters for the next stage
    if stage < max_stages - 1:
        model = PPO.load(f"C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_stage_0_best/best_model", env=env)
    else:
        # Enable hardcore mode for the last stage
        env = gym.make('BipedalWalker-v3', hardcore=True)
        env = DummyVecEnv([lambda: env])


