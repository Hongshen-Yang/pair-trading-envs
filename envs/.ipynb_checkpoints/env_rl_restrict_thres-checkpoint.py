import pickle
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

def read_best_params():
    with open('result/gridsearch/best_res.pickle', 'rb') as pk:
        _, best_params = pickle.load(pk)
    return best_params

class RL_Restrict_TradeEnv(gym.Env):
    def __init__(self, df, tc=0.0002):
        self.observation_space = gym.spaces.Discrete(2) # {0, 1}
        self.action_space = gym.spaces.Discrete(2) # {0, 1}
        # self.df = df
        # self.best_params = read_best_params()

    def _get_obs(self):

        # thresholds = {
        #     "zone0": np.array([1, 0, 0 ,0, 0]).astype(np.int8),
        #     "zone1": np.array([0, 1, 0 ,0, 0]).astype(np.int8),
        #     "zone2": np.array([0, 0, 1, 0, 0]).astype(np.int8),
        #     "zone3": np.array([0, 0, 0, 1, 0]).astype(np.int8),
        #     "zone4": np.array([0, 0, 0, 0, 1]).astype(np.int8),
        # }

        # zscore = self.df.iloc[self.step]['zscore']

        # if zscore > self.best_params['OPEN_THRE']:
        #     obs = thresholds["zone0"]
        # elif zscore > self.best_params['CLOS_THRE']:
        #     obs = thresholds["zone1"]
        # elif zscore < -self.best_params['OPEN_THRE']:
        #     obs = thresholds["zone2"]
        # elif zscore < -self.best_params['CLOS_THRE']:
        #     obs = thresholds["zone3"]
        # else:
        #     obs = thresholds["zone4"]

        return np.random.randint(2)

    def reset(self, seed=None):
        # super().reset(seed=seed)
        # self.trade_step = self.best_params['period']+1
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.signal = self.observation
        self.action = action
        self.observation = self._get_obs()
        terminated = False
        truncated = False
        self.reward = 1 if self.action==self.signal else 0

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{self.reward}")