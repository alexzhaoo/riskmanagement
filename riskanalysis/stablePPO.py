import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import stableenv as envd

data = pd.read_csv(r"data\AAPL_MSFT_GOOG_AMZN_NVDA_TSLAmonthlycompounded.csv")


env = envd.PortfolioAgentEnv(data)  


vec_env = make_vec_env(lambda: env, n_envs=1)  

model = PPO(
    "MlpPolicy",  
    vec_env,  
    verbose=1,  
    learning_rate=3e-4,  
    n_steps=2048, 
    batch_size=64,  
    n_epochs=10, 
    gamma=0.99,
    clip_range=0.2, 
    ent_coef=0.01, 
    seed=42, 
)


eval_env = envd.PortfolioAgentEnv(data) 
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./best_model", 
    log_path="./logs", 
    eval_freq=5000, 
    deterministic=True, 
    render=False,
)

model.learn(total_timesteps=100_000, callback=eval_callback)



model.save("ppo_financial_model")


trained_model = PPO.load("best_model/best_model")


obs = env.reset()
for _ in range(1000): 
    action, _states = trained_model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()


