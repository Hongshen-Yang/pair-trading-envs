import os
import csv
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from envs.env_gridsearch import kellycriterion

PERIOD = 1000 # Only look at the current price
CASH = 1
ISKELLY = False
OPEN_THRE = 1.8
CLOS_THRE = 0.4
FIX_AMT = 0.1

class PairTradingEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    # for pair trading, we need to feed in two OHLCV dataframes
    def __init__(self, df0, df1, tc=0.0002, period=PERIOD, cash=CASH, isKelly=ISKELLY, fixed_amt=FIX_AMT, verbose=0, noThres=False, model=""):
        super().__init__()

        if not df0['time'].equals(df1['time']):
            raise ValueError("Two dataframe must have same time index")

        self.cash = cash
        self.period = period
        self.model = model
        self.fixed_amt = fixed_amt
        self.verbose=verbose
        self.noThres = noThres

        # transaction cost
        self.tc = tc
        # Whether to use Kelly or not
        self.isKelly = isKelly

        self.df0 = df0[['close', 'datetime']]
        self.df1 = df1[['close', 'datetime']]

        self.reward_range = (-np.inf, np.inf)

        # Baseline 3 does not support Dict/Tuple action spaces....only Box Discrete MultiDiscrete MultiBinary
        self.action_space = spaces.Discrete(4) # actions: {0: short p0 long p1, 1: close, 2: long p0 short p1, 3: do nothing}

        if self.noThres:
            self.observation_space = spaces.Dict({
                "zscore":     spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "position":   spaces.MultiBinary(3), # {0: short leg0 long leg1, 1: none, 2: long leg0 short leg1}
            })
        else:
            self.observation_space = spaces.Dict({
                "threshold": spaces.MultiBinary(5), 
                # {0: above positive open thres, 1: between positive open thres and close thres, 
                # 2: between two close thres, 3 between negative open thres and close thres, 4: below negative open thres}
                "zscore":     spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "position":   spaces.MultiBinary(3), # {0: short leg0 long leg1, 1: none, 2: long leg0 short leg1}
            })

        self.position_action_mapping = [[1, 2, 3], [0, 2, 3], [0, 1, 3]]
    
        # if the length is 35, then the index shall be 0~34
        self.max_steps = len(df0)-1
    
    def _kellycriterion(self, direct):
        # direct is +1 or -1
        spreads = pd.Series(self.distance[-self.period:-1]) * direct   
        kc_f = kellycriterion(spreads)

        return kc_f

    def _next_observation(self):
        # The current step is always higher than the PERIOD as defined in the 

        prices0 = self.df0['close'].iloc[self.current_step-self.period: self.current_step]
        prices1 = self.df1['close'].iloc[self.current_step-self.period: self.current_step]

        self.distance = [x - y for x, y in zip(prices0, prices1)]
        self.zscore = (self.distance[-1] - np.mean(self.distance)) / np.std(self.distance)
        
        if self.noThres:
            obs = {
                "zscore": np.array([self.zscore]),
                "position": self.position,
            }

        else:
            '''The OPEN_THRE and CLOS_THRE comes from trade_gridsearch'''
            open_thre = OPEN_THRE
            clos_thre = CLOS_THRE

            if self.zscore > open_thre:
                threshold = 0
            elif self.zscore > clos_thre:
                threshold = 1
            elif self.zscore < -open_thre:
                threshold = 4
            elif self.zscore < -clos_thre:
                threshold = 3
            else:
                threshold = 2
            
            obs = {
                "threshold": int(threshold),
                "zscore": np.array([self.zscore]),
                "position": self.position,
            }
        
        return obs

    def _close_position(self):
        order_amount0 = -self.holding0
        order_amount1 = -self.holding1

        order_value0 = order_amount0 * self.curr_price0
        order_value1 = order_amount1 * self.curr_price1
        tc_cost = (abs(order_value0) + abs(order_value1)) * self.tc

        self.cash -= order_value0 + order_value1 + tc_cost
        self.holding0 = 0
        self.holding1 = 0
        self.position = 1

        self.order_amount0 = order_amount0
        self.order_amount1 = order_amount1

    def _open_position(self):

        # evaluate purchasing power 
        max_amount0 = (self.fixed_amt if self.fixed_amt else self.cash)/self.curr_price0
        max_amount1 = (self.fixed_amt if self.fixed_amt else self.cash)/self.curr_price1

        direction = self.action-1
        kc = self._kellycriterion(direct=direction) if self.isKelly else 1
        order_amount0 = direction * max_amount0 * kc
        order_amount1 = -direction * max_amount1 * kc

        order_value0 = order_amount0 * self.curr_price0
        order_value1 = order_amount1 * self.curr_price1
        tc_cost = (np.abs(order_value0) + np.abs(order_value1)) * self.tc

        # Open a new position
        self.cash -= order_value0 + order_value1 + tc_cost
        self.holding0 = order_amount0
        self.holding1 = order_amount1
        self.position = 1 if kc==0 else self.action

        self.kc = kc
        self.order_amount0 = order_amount0
        self.order_amount1 = order_amount1

    def _reverse_position(self):
        max_amount0 = (self.fixed_amt if self.fixed_amt else self.cash)/self.curr_price0
        max_amount1 = (self.fixed_amt if self.fixed_amt else self.cash)/self.curr_price1

        direction = self.action-1
        kc = self._kellycriterion(direct=direction) if self.isKelly else 1
        order_amount0 = direction * max_amount0 * kc - self.holding0
        order_amount1 = -direction * max_amount1 * kc - self.holding1

        order_value0 = order_amount0 * self.curr_price0
        order_value1 = order_amount1 * self.curr_price1
        tc_cost = (np.abs(order_value0) + np.abs(order_value1)) * self.tc

        self.cash -= order_value0 + order_value1 + tc_cost
        self.holding0 += order_amount0
        self.holding1 += order_amount1
        self.position = 1 if kc==0 else self.action

        # for debugging
        self.kc = kc
        self.order_amount0 = order_amount0
        self.order_amount1 = order_amount1

    def _take_action(self, action):

        # Record current net_worth to prev_net_worth
        self.prev_net_worth = self.net_worth

        self.curr_price0 = self.df0['close'].iloc[self.current_step]
        self.curr_price1 = self.df1['close'].iloc[self.current_step]

        self.action = action
        
        if self.action == 1:
            self._close_position()

        elif self.action == 0:
            if self.position  == 1:
                self._open_position()
            
            elif self.position == 2:
                self._reverse_position()
    
        elif self.action == 2:
            if self.position == 0:
                self._reverse_position()

            elif self.position == 1:
                self._open_position()

        # We record the net_worth from previous period to prev_net_worth
        self.net_worth = self.cash + self.holding0 * self.curr_price0 + self.holding1*self.curr_price1

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        self.observation = self._next_observation()

        if action in self.position_action_mapping[int(self.position)]:
            reward = self.net_worth - self.prev_net_worth
            reward = reward * 2 if reward < 0 else reward
        else:
            # Invalid action, penalize the agent
            reward = -1.0
        
        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(self.net_worth <= 0)
        info = {}

        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        np.random.seed(seed)
        
        self.cash = self.cash
        self.net_worth = self.cash
        self.prev_net_worth = self.cash
        self.position = 0
        self.render_step = 0
        
        self.holding0 = 0
        self.holding1 = 0
        self.order_amount0 = 0
        self.order_amount1 = 0
        self.kc = 0

        # self.current_step = np.random.randint(self.period, self.max_steps)
        self.current_step = self.period + 1
        
        return self._next_observation(), {}
    
    def render(self):
        profit = self.net_worth - self.cash

        if self.verbose == 1:        
            print(
                # f"direction: {self.action-1} "
                f"networth: {self.net_worth}, " 
                f"action: {self.action}, position: {self.position}, zscore: {self.zscore} "
                f"order_amount0: {self.order_amount0}, order_amount1: {self.order_amount1} "
                f"holding0: {self.holding0}, holding1: {self.holding1} "
                f"cash: {self.cash}, curr_price0: {self.curr_price0}, curr_price1: {self.curr_price1} "
            )
            
        with open(f"{self.model}", mode='a+', newline='') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                [self.df0['datetime'].iloc[self.current_step], 
                self.net_worth,
                self.action,
                self.zscore,
                self.position]
            )