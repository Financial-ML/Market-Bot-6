import gym
import json
import datetime as dt
import pandas as pd
import time
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from data.Features import *
from env.StockTradingEnv import StockTradingEnv



model = PPO2.load("ppo2_cartpole")


#---------------------------------------------------------------------
#df = pd.read_csv('data/calculated1.csv') insted add Applayer from MA3
#df = df.sort_values('Date')
#env = DummyVecEnv([lambda: StockTradingEnv(df)])

obs = env.reset()

#action, _states = model.predict(obs)
#obs, rewards, done, info = env.step(action)  translated to same model in MA3
#---------------------------------------------------------------------