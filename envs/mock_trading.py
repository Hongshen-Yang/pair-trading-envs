import pandas as pd

class tradingSystem():
    def __init__(self, df, positions, tc=0.0002, cash=1.0):
        self.cash = cash
        self.df = df
        self.tc = tc
        self.positions = positions

    def order(self, action):
        pass

    def check_positions(self):
        return self.positions
