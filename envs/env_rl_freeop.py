import os
import csv
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from envs.env_gridsearch import kellycriterion

# The lookback period for the observation space
PERIOD = 900 # Only look at the current price
CASH = 1
ISKELLY = False
OPEN_THRE = 1.6
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
        self.verbose = verbose
        self.noThres = noThres

        # transaction cost
        self.tc = tc
        # Whether to use Kelly or not
        self.isKelly = isKelly

        self.df0 = df0[['close', 'datetime']]
        self.df1 = df1[['close', 'datetime']]

        self.reward_range = (-np.inf, np.inf)

        # Baseline 3 does not support Dict/Tuple action spaces....only Box Discrete MultiDiscrete MultiBinary
        self.action_space = spaces.Discrete(3) # actions: {0: short p0 long p1, 1: do nothing, 2: long p0 short p1}

        if self.noThres:
            self.observation_space = spaces.Dict({
                "zscore":     spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "position":   spaces.Box(low=-1.0, high=1.0, dtype=np.float64), # [-1: short leg0 long leg1, 0: none, 1: long leg0 short leg1]
            })
        else:
            self.observation_space = spaces.Dict({
                "threshold": spaces.MultiBinary(5), 
                # {0: above positive open thres, 1: between positive open thres and close thres, 
                # 2: between two close thres, 3 between negative open thres and close thres, 4: below negative open thres}
                "zscore":     spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
                "position":   spaces.Box(low=-1.0, high=1.0, dtype=np.float64), # [-1: short leg0 long leg1, 0: none, 1: long leg0 short leg1]
            })

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
                "position": np.array([self.position]),
            }
        else:
            '''The OPEN_THRE and CLOS_THRE comes from trade_gridsearch'''
            open_thre = OPEN_THRE
            clos_thre = CLOS_THRE

            if self.zscore > open_thre:
                threshold = np.array([1, 0, 0 ,0, 0]).astype(np.int8)
            elif self.zscore > clos_thre:
                threshold = np.array([0, 1, 0, 0, 0]).astype(np.int8)
            elif self.zscore < -open_thre:
                threshold = np.array([0, 0, 1, 0, 0]).astype(np.int8)
            elif self.zscore < -clos_thre:
                threshold = np.array([0, 0, 0, 1, 0]).astype(np.int8)
            else:
                threshold = np.array([0, 0, 0, 0, 1]).astype(np.int8)
                
            obs = {
                "threshold": threshold,
                "zscore": np.array([self.zscore]),
                "position": np.array([self.position]),
            }
        
        return obs

    def _take_action(self, action):
        
        self.action = action
        # Record current net_worth to prev_net_worth
        self.prev_net_worth = self.net_worth

        curr_price0 = self.df0['close'].iloc[self.current_step]
        curr_price1 = self.df1['close'].iloc[self.current_step]

        self.curr_price0 = curr_price0
        self.curr_price1 = curr_price1

        max_amount0 = (self.fixed_amt if self.fixed_amt else self.cash)/curr_price0
        max_amount1 = (self.fixed_amt if self.fixed_amt else self.cash)/curr_price1

        direction = self.action-1
        kc = self._kellycriterion(direct=direction) if self.isKelly else 1

        # Leg0
        if abs(self.holding0 + max_amount0 * kc * direction) > max_amount0:
            order_amount = max_amount0 - self.holding0
            self.holding0 += order_amount

            order_value0 = order_amount * curr_price0
            tc_cost = abs(order_value0) * self.tc
            self.cash -= order_value0 + tc_cost

        else:
            order_amount = max_amount0 * kc * direction
            self.holding0 += order_amount

            order_value0 = order_amount * curr_price0
            tc_cost = abs(order_value0) * self.tc
            self.cash -= order_value0 + tc_cost
        
        # Leg1
        if abs(self.holding1 + max_amount1 * kc * -direction) > max_amount1:
            order_amount = -max_amount1 - self.holding1
            self.holding1 += order_amount

            order_value1 = order_amount * curr_price0
            tc_cost = abs(order_value1) * self.tc
            self.cash -= order_value1 + tc_cost

        else:
            order_amount = max_amount1 * kc * -direction
            self.holding1 += order_amount

            order_value1 = order_amount * curr_price0
            tc_cost = abs(order_value1) * self.tc
            self.cash -= order_value1 + tc_cost

        position0 = self.holding0/max_amount0
        position1 = self.holding1/max_amount1

        if position0 == 0:
            self.position = 0
        else:
            self.position = position0/abs(position0) * max(abs(position0), abs(position1))

        self.kc = kc
        # We record the net_worth from previous period to prev_net_worth
        self.net_worth = self.cash + self.holding0 * curr_price0 + self.holding1 * curr_price1

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        self.observation = self._next_observation()
        reward = self.net_worth - self.prev_net_worth
        terminated = bool(self.current_step >= self.max_steps)
        truncated = bool(self.net_worth <= 0)
        info = {}

        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        np.random.seed(seed)
        
        self.cash = self.cash
        self.net_worth = self.cash
        self.prev_net_worth = self.cash
        self.position = 1
        self.holding0 = 0
        self.holding1 = 0
        self.render_step = 0

        # self.current_step = np.random.randint(self.period, self.max_steps)
        self.current_step = self.period + 1

        return self._next_observation(), {}
    
    def render(self):
        profit = self.net_worth - self.cash

        if self.verbose == 1:
            print(f"networth {self.net_worth}, action {self.action}, zscore {self.zscore}, pos {self.position}, holding0 {self.holding0}, holding1 {self.holding1}")

        with open(f"{self.model}", mode='a+', newline='') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(
                [self.df0['datetime'].iloc[self.current_step], 
                self.net_worth,
                self.action,
                self.zscore,
                self.position,
                self.curr_price0,
                self.curr_price1]
            )