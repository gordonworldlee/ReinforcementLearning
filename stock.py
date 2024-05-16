import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('AAPLprices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'], inplace=True)

# env = gym.make('stocks-v0', df=df, frame_bound=(10,200), window_size=5)

# state = env.reset()
# while True:
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     # env.render()
#     if done:
#         print("info:", info)
#         break


# plt.figure(figsize=(10,6))
# plt.cla()
# env.unwrapped.render_all()
# plt.show()

env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C('MlpPolicy', env, verbose=1) 
model.learn(total_timesteps=1000000)