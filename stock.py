import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from finta import TA

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('AAPLprices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', ascending=True, inplace=True)
df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))

df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)
df.set_index(df['Date'], inplace=True)





#explained variance needs to be high
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,200), window_size=5)
env = DummyVecEnv([env_maker])

model = PPO('MlpPolicy', env, verbose=1) 
model.learn(total_timesteps=1000)




env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs, info = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        print("info", info)
        break

#plt.figure(figsize=(15,6))
plt.cla()
env.unwrapped.render_all()
plt.show()