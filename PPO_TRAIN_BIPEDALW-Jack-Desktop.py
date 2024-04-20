# Import necessary libraries
import gymnasium as gym  # OpenAI Gym library for creating environments
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor  # Monitor for monitoring environment
from stable_baselines3.common.vec_env import DummyVecEnv  # Vectorized environment
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluation function
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm

# Create the BipedalWalker environment with hardcore mode
n_envs = 1  # Number of parallel environments
save_freq = 10000  # Frequency to save checkpoints

# Define your BipedalWalker environment
env = gym.make("BipedalWalker-v3", hardcore=True)
env = DummyVecEnv([lambda: env])  # Vectorize the environment for parallelization

# TensorBoard logging directory
tensorboard_log_dir = "BIPEDALW_tensorboard_logs_PPO (Final version)"

# Initialize the PPO model
model = PPO(
    policy='MlpPolicy',  # Specify the policy architecture
    env=env,  # Pass the environment to the model
    tensorboard_log=tensorboard_log_dir  # Specify TensorBoard log directory
)

# Define the reward threshold callback to stop training when the threshold is reached
reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

# Define the evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./model_logs_bw/",  # Directory to save the best model
    log_path="./eval_logs_bw/",  # Directory to save evaluation logs
    eval_freq=10000,  # Frequency of evaluation
    deterministic=True,  # Use deterministic actions for evaluation
    render=False,  # Do not render evaluation episodes
    callback_on_new_best=reward_threshold_callback  # Callback to stop training when a new best reward is achieved
)

# Train the model
total_timesteps = 10_000_000  # Total number of training timesteps

checkpoint_callback = CheckpointCallback(
    save_freq=max(save_freq // n_envs, 1),  # Adjust save frequency for parallel environments
    save_path="./model_logs_bw/",  # Directory to save checkpoints
    name_prefix="rl_model"  # Prefix for checkpoint filenames
)

# Perform the training with callbacks for checkpointing and evaluation
model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, eval_callback],  # Use both checkpoint and evaluation callbacks
    progress_bar=True  # Show a progress bar during training
)

# Save the trained model
model.save("ppo-BipedalWalker-v3")

# Evaluate the trained model
eval_env = Monitor(gym.make("BipedalWalker-v3"))  # Create an environment monitor for evaluation
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)  # Evaluate the model
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")  # Print the mean and standard deviation of the rewards
