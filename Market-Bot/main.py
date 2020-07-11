import gym
import json
import datetime as dt
import pandas as pd
import time
import sys

from env.StockTradingEnv import x
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

#---------------------------------------------------------------------
#Load data
df = pd.read_csv('data/calculated.csv')
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

#Learn
model = PPO2(MlpPolicy, env, verbose=1, ent_coef=0.004)
model.learn(total_timesteps=200_000)

#save and load model
model.save("ppo2_cartpole")
del model
model = PPO2.load("ppo2_cartpole")

#---------------------------------------------------------------------
#Test the agent
df = pd.read_csv('data/calculated1.csv')
df = df.sort_values('Date')
env = DummyVecEnv([lambda: StockTradingEnv(df)])

x()
obs = env.reset()

for i in range(20_000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
#---------------------------------------------------------------------



