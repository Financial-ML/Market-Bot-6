import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import time
from random import randint

MAX_SHARE_PRICE = 5000
disaple = []

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    #initialize the shabe go enviroment
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.env_num = 0
        self.sum = 0

        # Actions of the format Buy Sell Hold
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last Bar
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(75,1), dtype=np.float16)
    #get the next state
    def _next_observation(self):
        # Get the stock data points for the last Bar and scale to between 0-1
        frame = np.array([

            [self.df.loc[self.current_step, 'momentum3close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'momentum4close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'momentum5close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'momentum8close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'momentum9close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'momentum10close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch3K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch3D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch4K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch4D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch5K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch5D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch8K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch8D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch9K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch9D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch10K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch10D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'will6R'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'will7R'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'will8R'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'will9R'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'will10R'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'proc12close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'proc13close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'proc14close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'proc15close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'wadl15close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch10K'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'stoch10D'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'adosc2AD'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'adosc3AD'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'adosc4AD'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'adosc5AD'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'macd1530'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'cci15close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'bollinger15upper'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'bollinger15mid'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'bollinger15lower'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'paverage2open'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'paverage2high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'paverage2low'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'paverage2close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope3high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope4high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope5high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope10high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope20high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'slope30high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier10a0'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier10a1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier10b1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier10w'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier20a0'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier20a1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier20b1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier20w'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier30a0'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier30a1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier30b1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'fourier30w'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine5a0'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine5b1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine5w'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine6a0'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine6b1'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'sine6w'] / MAX_SHARE_PRICE],

            [self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE],
            [self.df.loc[self.current_step, 'volume'] / MAX_SHARE_PRICE],
            #get feed back of True valus
            [self.balance / 999999],
            [self.reward / 999999],
            [self.sum / 999999],
        ])
        return frame
    #exicute the state (taking the action)
    def _take_action(self, action):
        # Set the current price to the time step
        current_price = self.df.loc[self.current_step , 'close']
        #-----------------------------------------------
        #calculate if True
        reward_cur = 0
        if action == 0:#sell
            reward_cur = current_price - self.df.loc[self.current_step + 20 , 'close'] - 0.0003
        if action == 1:#buy
            reward_cur = self.df.loc[self.current_step + 20 , 'close'] - current_price - 0.0003

        self.sum = self.sum + reward_cur * 1000

        self.pre_balance = self.balance
        if reward_cur > 0:
            self.balance  = self.balance  + 1
        if reward_cur < 0:
            self.balance  = self.balance - 1

        if self.pre_balance != self.balance:
            self.count += 1
            
        if self.count == 0:
            self.count = 1
        #-----------------------------------------------
    #calculate the order and move
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        win = self.count / 2 + self.balance / 2
        win_rate = win / self.count
        self.reward =  self.balance - (self.count /2)
        
        done = False

        if self.sum <= -1000:
            print('Broke')
            
        if self.current_step > self.current_step_end:
            xx = 1
            for i in range(0,len(disaple)):
                if disaple[i] == self.env_num:
                    xx = 0
            if xx == 1:
                print('----------------------')
                print(f'Value True: {self.balance}')
                print(f'Number of orders: {self.count}')
                print(f'reward : {round(self.reward,2)}')
                print(f'env_num : {self.env_num}')
                print(f'Profit : {self.sum}')
                print('----------------------')
                #time.sleep(0.4)
            disaple.insert(len(disaple),self.env_num)
            done = True

        obs = self._next_observation()
        
        return obs, self.reward, done, {}
    # Reset the state of the environment to an initial state
    def reset(self):
        #-------------------
        #reset time_step to random epsiode each epsiode has 1000 time_step widow size
        window = len(self.df) / 1000
        window = int(window)
        current_pos = randint(0,window-1)
        
        self.env_num = current_pos
        self.current_step = current_pos * 1000
        self.current_step_end = self.current_step + 1000 - 26
        #-------------------
        self.balance = 0
        self.count = 1
        self.reward = 0
        self.sum = 0
        
        return self._next_observation()

def x():
    disaple.clear()