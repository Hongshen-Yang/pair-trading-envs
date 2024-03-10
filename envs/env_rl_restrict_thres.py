import pickle
import gymnasium as gym
import numpy as np
import pandas as pd

from envs.mock_trading import TradingSystem, TradingSystemFreeAmt
from utils.logger import logger

def read_best_params():
    with open('result/gridsearch/best_res.pickle', 'rb') as pk:
        _, best_params = pickle.load(pk)
    return best_params

class RL_Restrict_TradeEnv(gym.Env):
    def __init__(self, df, model='', tc=0.002, cash=1.0, fixed_amt=0.1, verbose=0):
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
        
        if self.signal['zone']==0 and self.signal['position']==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==1:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==0 and self.signal['position']==2:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==1 and self.signal['position']==2:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==0:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==2 and self.signal['position']==2:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==0:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==1:
            reward = act_rwd if self.action==1 else 0
        elif self.signal['zone']==3 and self.signal['position']==2:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==0:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==1:
            reward = act_rwd if self.action==2 else 0
        elif self.signal['zone']==4 and self.signal['position']==2:
            reward = act_rwd if self.action==2 else 0

        reward += (self.networth - prev_networth)*100
        return reward

    def _take_action(self):
        sys=TradingSystem(self.df, self.holdings, self.trade_step, cash=self.cash, amt=self.fixed_amt)

        if self.position==0 and self.action==0:
            # Do nothing
            pass
        elif self.position==0 and self.action==1:
            # Close position
            self.cash, self.holdings = sys.close_position()
            self.networth = sys.get_networth()
        elif self.position==0 and self.action==2:
            # Long leg0 short leg1
            self.cash, self.holdings = sys.open_position(self.action)
        elif self.position==1 and self.action==0:
            # Short leg0 long leg1
            self.cash, self.holdings = sys.open_position(self.action)
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
            self.networth = sys.get_networth()
        elif self.position==2 and self.action==2:
            # Do nothing
            pass

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

class RL_RestrictFreeAmt_TradeEnv(gym.Env):
    def __init__(self, df, model='', tc=0.002, cash=1.0, verbose=0): # act_pun is action punishment
        self.observation_space = gym.spaces.Dict({
            'holdings': gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32), # {0, 1, 2}
                    #   Position=0: position closed
                    #   Position>0: longing
                    #   Position<0: shorting
            'zone':  gym.spaces.Discrete(5), # {0, 1, 2, 3, 4}
                    # The zscore comes from price0-price1, zone0 stands for price0 way higher than price1
                    #   Zone 0
                    # ---------- + OPEN_THRES ----------
                    #   Zone 1
                    # ---------- + CLOS_THRES ----------
                    #   Zone 2
                    # ----------   ZSCORE = 0 ----------
                    #   Zone 2
                    # ---------- - CLOS_THRES ----------
                    #   Zone 3
                    # ---------- - OPEN_THRES ----------
                    #   Zone 4
            'zscore': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64)
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float64)
        # {[-1, 0]: "short leg0 long leg1", 0: "close positions", [0,1]: "long leg0 short leg1"}

        self.verbose = verbose
        self.cash, self.networth = cash, cash
        self.df = df
        self.model = model
        self.best_params = read_best_params()
        self.holdings = np.array([0, 0], dtype=np.float32) #[1, -1] That means we have 1 unit of leg0 and -1 unit of leg1
        self.units = np.array([0, 0], dtype=np.float32) # Holding but in units
        self.tc = tc

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
        
        df_current = self.df.iloc[self.trade_step]
        price0 = df_current['close0']
        value0= self.units[0]*price0
        perc = value0/self.networth
        self.holdings = np.array([perc, -perc], dtype=np.float32)

        obs = {
            'holdings': self.holdings,
            'zone': zone,
            'zscore': np.array([zscore], dtype=np.float32)
        }

        return obs
    
    def _get_reward(self, prev_networth):

        act_rwd = 1
        act_rwd_lvl1 = 1 # Close a position in the right time
        act_rwd_lvl2 = 0.7 # Open a position in the right time
        act_rwd_lvl3 = 0.5 # Do nothing in the right time
        
        # if   self.signal['zone']==0 and self.signal['holdings'][0]<0:
        #     reward = act_rwd_lvl3 if self.action<0 else 0
        # elif self.signal['zone']==0 and self.signal['holdings'][0]==0:
        #     reward = act_rwd_lvl2 if self.action<0 else 0
        # elif self.signal['zone']==0 and self.signal['holdings'][0]>0:
        #     reward = act_rwd_lvl1 if self.action<0 else 0
        # elif self.signal['zone']==1 and self.signal['holdings'][0]<0:
        #     reward = act_rwd_lvl3 if self.action<0 else 0
        # elif self.signal['zone']==1 and self.signal['holdings'][0]==0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==1 and self.signal['holdings'][0]>0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==2 and self.signal['holdings'][0]<0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==2 and self.signal['holdings'][0]==0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==2 and self.signal['holdings'][0]>0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==3 and self.signal['holdings'][0]<0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==3 and self.signal['holdings'][0]==0:
        #     reward = act_rwd_lvl1 if self.action==0 else 0
        # elif self.signal['zone']==3 and self.signal['holdings'][0]>0:
        #     reward = act_rwd_lvl3 if self.action>0 else 0
        # elif self.signal['zone']==4 and self.signal['holdings'][0]<0:
        #     reward = act_rwd_lvl1 if self.action>0 else 0
        # elif self.signal['zone']==4 and self.signal['holdings'][0]==0:
        #     reward = act_rwd_lvl2 if self.action>0 else 0
        # elif self.signal['zone']==4 and self.signal['holdings'][0]>0:
        #     reward = act_rwd_lvl3 if self.action>0 else 0

        if   self.signal['zone']==0 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<=0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>=0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>0 else 0

        # Transaction Fee Punishment
        # act_pnm = abs(self.action-self.signal['holdings'][0])/10
        # reward -= act_pnm

        reward += (self.networth-prev_networth)*100

        reward += 0.1 if self.action==0 else 0
        
        return reward

    def _take_action(self):

        sys=TradingSystemFreeAmt(self.df, self.units, self.trade_step, self.cash, self.tc)

        if   self.holdings[0]<0 and self.action<0:
            self.cash, self.units = sys.adjust_position(self.action)
        elif self.holdings[0]<0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
        elif self.holdings[0]<0 and self.action>0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action<0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action==0:
            # Do nothing
            pass
        elif self.holdings[0]==0 and self.action>0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action<0:
            # Close and open position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
        elif self.holdings[0]>0 and self.action>0:
            self.cash, self.units = sys.adjust_position(self.action)

        # self.cash, self.units = sys.adjust_position(self.action)
        # self.networth = sys.get_networth()
        
    def reset(self, seed=None):
        self.holdings = np.array([0, 0], dtype=np.float32)
        self.units = np.array([0, 0], dtype=np.float32)
        self.trade_step = self.best_params['period']
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.action = action[0]
        # Separate Signal as previous observation
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
            logger(self.model, curr_df['datetime'], self.networth, self.action, curr_df['zscore'], self.holdings, curr_df['close0'], curr_df['close1'])

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{round(self.reward, 3)}, networth: {round(self.networth, 4)}")

    def close(self):
        print("Finished")
        print(f"networth: {self.networth}")

class RL_RestrictFreeAmtTalib_TradeEnv(gym.Env):
    def __init__(self, df, model='', tc=0.002, cash=1.0, verbose=0): # act_pun is action punishment
        self.observation_space = gym.spaces.Dict({
            'holdings': gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32), # {0, 1, 2}
                    #   Position=0: position closed
                    #   Position>0: longing
                    #   Position<0: shorting
            'zone':  gym.spaces.Discrete(5), # {0, 1, 2, 3, 4}
                    # The zscore comes from price0-price1, zone0 stands for price0 way higher than price1
                    #   Zone 0
                    # ---------- + OPEN_THRES ----------
                    #   Zone 1
                    # ---------- + CLOS_THRES ----------
                    #   Zone 2
                    # ----------   ZSCORE = 0 ----------
                    #   Zone 2
                    # ---------- - CLOS_THRES ----------
                    #   Zone 3
                    # ---------- - OPEN_THRES ----------
                    #   Zone 4
            'zscore': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'close0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'close1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'volume0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'volume1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'high0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'high1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'low0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'low1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'upperband0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'upperband1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'adx0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'adx1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'ad0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'ad1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'atr0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'atr1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'ht_dcperiod0': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'ht_dcperiod1': gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
        })

        self.action_space = gym.spaces.Box(low=-1, high=1, dtype=np.float64)
        # {[-1, 0]: "short leg0 long leg1", 0: "close positions", [0,1]: "long leg0 short leg1"}

        self.verbose = verbose
        self.cash, self.networth = cash, cash
        self.df = df
        self.model = model
        self.best_params = read_best_params()
        self.holdings = np.array([0, 0], dtype=np.float32) #[1, -1] That means we have 1 unit of leg0 and -1 unit of leg1
        self.units = np.array([0, 0], dtype=np.float32) # Holding but in units
        self.tc = tc

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
        
        df_current = self.df.iloc[self.trade_step]
        price0 = df_current['close0']
        value0= self.units[0]*price0
        perc = value0/self.networth
        self.holdings = np.array([perc, -perc], dtype=np.float32)

        obs = {
            'holdings': self.holdings,
            'zone': zone,
            'zscore': np.array([zscore], dtype=np.float32)
        }

        return obs
    
    def _get_reward(self, prev_networth):

        act_rwd = 1
        act_rwd_lvl1 = 1 # Close a position in the right time
        act_rwd_lvl2 = 0.7 # Open a position in the right time
        act_rwd_lvl3 = 0.5 # Do nothing in the right time

        if   self.signal['zone']==0 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==0 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action<0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action<=0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==1 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==2 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action==0 else 0
        elif self.signal['zone']==3 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>=0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]<0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]==0:
            reward = act_rwd if self.action>0 else 0
        elif self.signal['zone']==4 and self.signal['holdings'][0]>0:
            reward = act_rwd if self.action>0 else 0

        # Transaction Fee Punishment
        # act_pnm = abs(self.action-self.signal['holdings'][0])/10
        # reward -= act_pnm

        reward += (self.networth-prev_networth)*100

        reward += 0.1 if self.action==0 else 0
        
        return reward

    def _take_action(self):

        sys=TradingSystemFreeAmt(self.df, self.units, self.trade_step, self.cash, self.tc)

        if   self.holdings[0]<0 and self.action<0:
            self.cash, self.units = sys.adjust_position(self.action)
        elif self.holdings[0]<0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
        elif self.holdings[0]<0 and self.action>0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action<0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]==0 and self.action==0:
            # Do nothing
            pass
        elif self.holdings[0]==0 and self.action>0:
            # Open position
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action<0:
            # Close and open position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
            self.cash, self.units = sys.open_position(self.action)
        elif self.holdings[0]>0 and self.action==0:
            # Close position
            self.cash, self.units = sys.close_position()
            self.networth = sys.get_networth()
        elif self.holdings[0]>0 and self.action>0:
            self.cash, self.units = sys.adjust_position(self.action)

        # self.cash, self.units = sys.adjust_position(self.action)
        # self.networth = sys.get_networth()
        
    def reset(self, seed=None):
        self.holdings = np.array([0, 0], dtype=np.float32)
        self.units = np.array([0, 0], dtype=np.float32)
        self.trade_step = self.best_params['period']
        self.observation = self._get_obs()
        return self.observation, {}

    def step(self, action):
        self.action = action[0]
        # Separate Signal as previous observation
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
            logger(self.model, curr_df['datetime'], self.networth, self.action, curr_df['zscore'], self.holdings, curr_df['close0'], curr_df['close1'])

        return self.observation, self.reward, terminated, truncated, {}

    def render(self):
        print(f"signal: {self.signal}, action: {self.action}, reward:{round(self.reward, 3)}, networth: {round(self.networth, 4)}")

    def close(self):
        print("Finished")
        print(f"networth: {self.networth}")