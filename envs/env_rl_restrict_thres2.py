import pickle
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

class RL_Restrict_TradeEnv(gym.Env):
    def __init__(self, df, tc=0.0002):
        with open('result/gridsearch/best_res.pickle', 'rb') as pk:
            _, best_params = pickle.load(pk)

        self.observation_space = gym.spaces.MultiBinary(2) # {0, 1}
        self.action_space = gym.spaces.Discrete(2) # {0, 1}
        self.df = df
        self.best_params = best_params
        self.period = best_params['period']

    def _get_obs(self):
        
        thresholds = {
            "zone0": np.array([1, 0, 0 ,0, 0]).astype(np.int8),
            "zone1": np.array([0, 1, 0 ,0, 0]).astype(np.int8),
            "zone2": np.array([0, 0, 1, 0, 0]).astype(np.int8),
            "zone3": np.array([0, 0, 0, 1, 0]).astype(np.int8),
            "zone4": np.array([0, 0, 0, 0, 1]).astype(np.int8),
        }

        zscore = self.df.iloc[self.step]['zscore']

        if zscore > best_params['OPEN_THRE']:
            threshold = self.thresholds["zone0"]
        elif zscore > best_params['CLOS_THRE']:
            threshold = self.thresholds["zone1"]
        elif zscore < -best_params['OPEN_THRE']:
            threshold = self.thresholds["zone2"]
        elif zscore < -best_params['CLOS_THRE']:
            threshold = self.thresholds["zone3"]
        else:
            threshold = self.thresholds["zone4"]

        return threshold

    def reset(self, seed=None):
        # super().reset(seed=seed)
        self.step = self.best_params['period']+1
        self.observation = self._get_obs()
        info = {}
        return self.observation, info

    def step(self, action):
        self.signal = self.observation
        self.action = action
        self.observation = self._get_obs()
        terminated = False
        truncated = False
        self.reward = 1 if self.action==1 else 0
        info = {}

        return self.observation, self.reward, terminated, truncated, info
    
    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{self.reward}")