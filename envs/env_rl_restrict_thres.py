import pickle
import gymnasium as gym
import numpy as np
import pandas as pd

from envs.mock_trading import TradingSystem
from utils.logger import logger

def read_best_params():
    with open('result/gridsearch/best_res.pickle', 'rb') as pk:
        _, best_params = pickle.load(pk)
    return best_params

class RL_Restrict_TradeEnv(gym.Env):
    def __init__(self, df, model='', tc=0.0002, cash=1.0, fixed_amt=0.1, verbose=0):
        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Discrete(3), # {0, 1, 2}
                    #   Position 0: shorting leg_0 -> longing leg_1
                    #   Position 1:         empty holding
                    #   Position 2: longing leg_0 <- shorting leg_1
            'zone':  gym.spaces.Discrete(5), # {0, 1, 2, 3, 4}
                    # The zscore comes from price0-price1, zone0 stands for price0 way higher than price1
                    #   Zone 0 (Should be position 0)
                    # ---------- + OPEN_THRES ----------
                    #   Zone 1 (Should be position 0, 1)
                    # ---------- + CLOS_THRES ----------
                    #   Zone 2 (Should be position 1)
                    # ----------   ZSCORE = 0 ----------
                    #   Zone 2 (Should be position 1)
                    # ---------- - CLOS_THRES ----------
                    #   Zone 3 (Should be position 1, 2)
                    # ---------- - OPEN_THRES ----------
                    #   Zone 4 (Should be position 2)
            'zscore': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64)
        })
        self.action_space = gym.spaces.Discrete(3) # {0: "short leg0 long leg1", 1: "close positions", 2: "long leg0 short leg1"}

        self.verbose = verbose
        self.cash, self.networth = cash, cash
        self.fixed_amt = fixed_amt
        self.df = df
        self.model = model
        self.best_params = read_best_params()
        self.holdings = [0, 0] #[400, -300] That means we have 400 unit of leg0 and -300 unit of leg1

    def _get_obs(self):
        zscore = self.df.iloc[self.trade_step]['zscore']

        if zscore > self.best_params['OPEN_THRE']:
            zone = 0
        elif zscore > self.best_params['CLOS_THRE']:
            zone = 1
        elif zscore < -self.best_params['OPEN_THRE']:
            zone = 4
        elif zscore < -self.best_params['CLOS_THRE']:
            zone = 3
        else:
            zone = 2

        obs = {
            'position': self.position,
            'zone': zone,
            'zscore': np.array([zscore])
        }

        return obs
    
    def _get_reward(self, prev_networth):
        act_rwd = 1
        act_pun = 0.1
        action_reward = act_rwd - act_pun
        
        if self.signal['zone']==0 and self.signal['position']==0:
            reward = action_reward if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==1:
            reward = action_reward if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==2:
            reward = action_reward if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==0:
            reward = action_reward if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==1:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==1 and self.signal['position']==2:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==0:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==1:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==2:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==0:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==1:
            reward = action_reward if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==2:
            reward = action_reward if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==0:
            reward = action_reward if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==1:
            reward = action_reward if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==2:
            reward = action_reward if self.action==2 else 0

        # reward += self.networth - prev_networth
        reward = reward * 10

        return reward

    def _take_action(self):
        sys=TradingSystem(self.df, self.holdings, self.trade_step, cash=self.cash, amt=self.fixed_amt)

        if self.position==0 and self.action==0:
            # Do nothing
            pass
        elif self.position==0 and self.action==1:
            # Close position
            self.cash, self.holdings = sys.close_position()
        elif self.position==0 and self.action==2:
            # Long leg0 short leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==1 and self.action==0:
            # Short leg0 long leg1
            self.cash, self.holdings = sys.open_position(0)
        elif self.position==1 and self.action==1:
            # Do nothing
            pass
        elif self.position==1 and self.action==2:
            # Long leg0 short leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==2 and self.action==0:
            # Short leg0 long leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==2 and self.action==1:
            # Close position
            self.cash, self.holdings = sys.close_position()
        elif self.position==2 and self.action==2:
            # Do nothing
            pass
        
        self.networth = sys.get_networth()
        self.position = self.action

    def reset(self, seed=None):
        self.position = 1
        self.trade_step = self.best_params['period']
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.action = action
        self.signal = self.observation
        prev_networth = self.networth
        self._take_action()
        self.trade_step += 1
        self.observation = self._get_obs()
        terminated = self.trade_step >= len(self.df)
        truncated = False
        self.reward = self._get_reward(prev_networth)

        if self.verbose==1:
            curr_df = self.df.iloc[self.trade_step]
            logger(self.model, curr_df['datetime'], self.networth, self.action, curr_df['zscore'], self.position, curr_df['close0'], curr_df['close1'])

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{round(self.reward, 3)}, networth: {round(self.networth, 4)}")

    def close(self):
        print("Finished")
        print(f"networth: {self.networth}")