import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import env as envd

data = pd.read_csv(r"data\AAPL_MSFT_GOOG_AMZN_NVDA_TSLAmonthlycompounded.csv")

# Create the environment
env = envd.PortfolioAgentEnv(data)  # Ensure this matches your custom env

# Wrap the environment for vectorization (optional but recommended for Stable-Baselines3)
vec_env = make_vec_env(lambda: env, n_envs=1)  # n_envs=1 for single environment

# Set up the PPO model
model = PPO(
    "MlpPolicy",  # Use an MLP policy
    vec_env,  # The environment
    verbose=1,  # Set verbose to 1 for training logs
    learning_rate=3e-4,  # Default learning rate; adjust if needed
    n_steps=2048,  # Number of steps to collect for each update
    batch_size=64,  # Batch size for PPO
    n_epochs=10,  # Number of epochs per policy update
    gamma=0.99,  # Discount factor
    clip_range=0.2,  # Clipping range for PPO
    ent_coef=0.01,  # Entropy coefficient
    seed=42,  # Set a random seed for reproducibility
)

# Set up evaluation callback
eval_env = envd.PortfolioAgentEnv(data)  # A separate evaluation environment
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./best_model", 
    log_path="./logs", 
    eval_freq=5000, 
    deterministic=True, 
    render=False,
)

# Train the agent
model.learn(total_timesteps=100_000, callback=eval_callback)

# Save the trained model
model.save("ppo_financial_model")

# Load and test the trained model
trained_model = PPO.load("ppo_financial_model")

# Test the model
obs = env.reset()
for _ in range(1000):  # Run for 1000 timesteps
    action, _states = trained_model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

# Print final portfolio value
print("Final portfolio value:", info.get("portfolio_value", "Unknown"))
