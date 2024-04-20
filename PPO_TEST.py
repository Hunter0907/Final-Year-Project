import gymnasium as gym  # Importing the Gym library for reinforcement learning environments
from stable_baselines3 import PPO  # Importing the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3

# Load the BipedalWalker-v3 environment from Gym with hardcore mode and human render mode
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

# Load the trained PPO model from the specified path - STANDARD PPO TRAINED FILES
model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_logs/best_model.zip")


# Load the trained PPO model from the specified path - STANDARD PPO TRAINED FILES
model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_logs/best_model.zip")
#model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_logs_bw/rl_model_90000_steps.zip")
#model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_logs/rl_model_1000000_steps.zip")
#model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_logs/rl_model_500000_steps.zip")


# Load the trained PPO model from the specified path - CL TRAINED FILES
#model = PPO.load("C:/Users/jackc/OneDrive/UCC/Year4_S1/Project/GYM/G19 Computer/4th Year Project/model_stage_1_best/best_model.zip")

# Loop for running the environment for 1000 episodes
for _ in range(1000):
    # Reset the environment for each episode, returning the initial observation and info
    observation, info = env.reset()

    truncated = False  # Initialize the 'truncated' flag to False
    terminated = False  # Initialize the 'terminated' flag to False

    # Loop for interacting with the environment until termination or truncation
    while not truncated and not terminated:
        # Predict an action using the trained model and the current observation, with deterministic behavior
        action, _state = model.predict(observation, deterministic=True)
        
        # Take a step in the environment using the predicted action, returning the new observation, reward, and info
        observation, reward, terminated, truncated, info = env.step(action)
        
    # If the episode terminates or is truncated, reset the environment for the next episode
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment after finishing all episodes
env.close()
