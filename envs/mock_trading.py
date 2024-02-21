import pandas as pd
import numpy as np

class TradingSystem():
    def __init__(self, df, holdings, trade_step, cash, amt=0, tc=0.002):
        self.cash = cash
        self.df = df.iloc[trade_step] # columns: time, close0, itvl, datetime, close1, spread, zscore
        self.tc = tc # transaction cost
        self.holdings = holdings #[400, -300] That means we have 400 unit of leg0 and -300 unit of leg1
        self.amt = amt
        self.trade_step = trade_step

    def close_position(self):
        self.cash += (self.holdings[0]*self.df['close0'] + self.holdings[1]*self.df['close1']) * (1-self.tc)
        self.holdings = [0, 0]

        return self.cash, self.holdings

    def open_position(self, action):
        self.close_position()

        units_leg0 = self.amt/self.df['close0'] if self.amt else self.cash/self.df['close0']
        units_leg1 = self.amt/self.df['close1'] if self.amt else self.cash/self.df['close1']

        if action == 0:
            self.holdings = [-units_leg0*(1-self.tc), units_leg1*(1-self.tc)]
        elif action == 2:
            self.holdings = [units_leg0*(1-self.tc), -units_leg1*(1-self.tc)]
        else:
            self.holdings = [0, 0]

        return self.cash, self.holdings

    def get_holdings(self):
        return self.holdings

    def get_networth(self):
        return self.cash + self.holdings[0]*self.df['close0'] + self.holdings[1]*self.df['close1']

class TradingSystemFreeAmt():
    def __init__(self, df, units, trade_step, cash, tc=0.002):
        self.cash = cash
        self.df = df.iloc[trade_step] # columns: time, close0, itvl, datetime, close1, spread, zscore
        self.tc = tc # transaction cost
        self.units = units #[400, -300] That means we have 400 unit of leg0 and -300 unit of leg1
        self.trade_step = trade_step
        self.networth = cash + units[0]*self.df['close0'] + units[1]*self.df['close1']

    # def close_position(self):
    #     self.cash += (self.units[0]*self.df['close0'] + self.units[1]*self.df['close1']) * (1-self.tc)
    #     self.units = np.array([0, 0])

    #     return self.cash, self.units

    # def open_position(self, action):
    #     self.cash, self.units = self.close_position()
    #     units_leg0 = action*self.cash/self.df['close0']
    #     units_leg1 = action*self.cash/self.df['close1']
    #     self.units = [units_leg0*(1-self.tc), -units_leg1*(1-self.tc)]

    #     return self.cash, self.units

    def adjust_position(self, action):
        # How much we buy in units
        units_leg0 = self.networth*action/self.df['close0']
        units_leg1 = self.networth*-action/self.df['close1']
        # Transaction Fee
        ttl_leg0 = abs(self.units[0]-units_leg0)*self.df['close0']
        ttl_leg1 = abs(self.units[1]-units_leg1)*self.df['close1']
        tc = (ttl_leg0 + ttl_leg1)*self.tc
        # self.cash -= tc
        # Update to the newest units
        self.units = np.array([units_leg0, units_leg1], dtype=np.float32)

        return self.cash, self.units

    def get_units(self):
        return self.units

    def get_networth(self):
        return self.networth
