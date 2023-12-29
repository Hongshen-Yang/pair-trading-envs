import pickle
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

class RL_Restrict_TradeEnv(gym.Env):
    def __init__(self, df, tc=0.0002):
        with open('result/gridsearch/best_res.pickle', 'rb') as pk:
            _, best_params = pickle.load(pk)

        self.observation_space = gym.spaces.Discrete(2) # {0, 1}
        self.action_space = gym.spaces.Discrete(2) # {0, 1}
        self.period = best_params['period']

    def _get_obs(self):
        return np.random.randint(2) # {0, 1}
    
    def reset(self, seed=None):
        # super().reset(seed=seed)
        self.observation = self._get_obs()
        info = {}
        return self.observation, info

    def step(self, action):
        self.signal = self.observation
        self.action = action
        self.observation = self._get_obs()
        terminated = False
        truncated = False
        self.reward = 1 if self.action==self.signal else 0
        info = {}

        return self.observation, self.reward, terminated, truncated, info
    
    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{self.reward}")