import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
import stableenv as envd
import warnings
import numpy as np
from data import yfinanceuse




def main():
    warnings.filterwarnings("ignore")
    tickers = yfinanceuse.get_tickers()
    data = pd.read_csv(rf"data\{tickers}monthlycompounded.csv")
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


    trained_model = PPO.load("best_model/best_model")


    obs = env.reset()


    final_weights = None

    for _ in range(len(data) // (env.num_stocks * env.window_size)):  
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        final_weights = action  
        if done:
            break 
    
    final_weights = final_weights / np.sum(final_weights)

    np.savetxt(rf"weights/{tickers}weights.csv", final_weights, delimiter=",")

    print("Normalized Portfolio Weights:", final_weights)


    print('Done')

main()