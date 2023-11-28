import os
import csv
import datetime
import backtrader as bt
import numpy as np

# Define Trading Strategy

'''
### Custom Sizer
Firstly define a sizer based on [Kelly Criterion](https://www.wikiwand.com/en/Kelly_criterion)

*Not in Use*

# Seems that Sizer can only be executed when self.buy(size=None). 
# We need to purchase amount in a certain ratio in Pair Trading.
# Therefore the Sizer is hard to implemented.
'''

def kellycriterion(distances):
    '''
    We had discussion on the Kelly Criterion on 30-Oct-2023
    The Kelly Criterion should be based on the change of the difference between the long leg or the short leg
    Let's say BTCUSDT is [50, 51, 53, 61, 54, 52], BTCUSDC is [48, 52, 50, 61, 53, 50]
    Then BTCUSDT - BTCUSDC is [2, -1, 3, 0, 1, 2], which means we long BTCUSDT and short BTCUSDC
    Then the Kelly Criterion assumes that we are holding the difference between BTCUSDT and BTCUSDC
    '''

    kc_p = len(distances[distances > 0]) / len(distances)
    kc_q = 1 - kc_p
    kc_a = -(distances[distances < 0].mean()) if len(distances[distances < 0]) > 0 else 1e-5
    kc_b = (distances[distances > 0].mean()) if len(distances[distances > 0]) > 0 else 1e-5
    f = min(max((kc_p/kc_a - kc_q/kc_b), 0), 1)

    return f


class KellyCriterionSizer(bt.Sizer):
    params = (('period', 30),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data).size

        close_prices = pd.Series(data.close.get(size=self.p.period))
        f = kellycriterion(close_prices)
        
        size = cash * f / data.close[0] if isbuy else position * f

        return size

'''
### Custom Indicator
Define a custom indicator for [Kelly Criterion](https://www.wikiwand.com/en/Kelly_criterion) & Zscore
Define custom CommissionInfo (*Not in use*)
'''

class KellyCriterionIndicator(bt.indicators.PeriodN):
    _mindatas = 1

    packages = (('pandas', 'pd'),)
    lines = ('kc_f',)
    params = (('period', 30),)

    # Return a single data point use `next``
    def next(self):
        spreads = pd.Series(self.data.get(size=self.p.period))
        # Because we expect the return to be close up, hence a gain means smaller spread
        kc_f = kellycriterion(spreads)
        
        self.lines.kc_f[0] = kc_f

# The official documents do not have 2 data scenario
# I can only imitate from source code
# https://github.com/mementum/backtrader/blob/master/backtrader/indicators/ols.py
# 2023-Oct-08, Gave up on using custom ZscoreIndicator, mismatch between docs & source code

class ZscoreIndicator(bt.indicators.PeriodN):
    # ensure at least 2 data feeds are passed
    _mindatas = 2
    packages = (
        ('pandas', 'pd'),
        ('statsmodels.api', 'sm'),
        ('scipy.stats', 'stats'),
    )
    lines = ('spread', 'zscore',)
    params = (('period', 30),)

    def next(self):
        p0, p1 = (pd.Series(data.get(size=self.p.period)) for data in self.datas)
        distance = pd.Series(p0-p1)
        zscores = stats.zscore(distance)
        self.lines.spread[0] = distance[-1]
        self.lines.zscore[0] = zscores[-1]

class PairTradingCommInfo(bt.CommInfoBase):
    params = (
        ('commission', 0.0), ('mult', 10), ('margin', 1000),
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', True),
    )

strategytxt = f"result/gridsearch/strategy.txt"
os.remove(strategytxt) if os.path.exists(strategytxt) else None

class PairTrading(bt.Strategy):
    params = dict(
        OPEN_THRE=2,
        CLOS_THRE=0.1,
        period=30,
        verbose = 0, # 0 for only recording final result, 1 for writing the process in file, 2 for writing the process in file + printing final result
        kellycriterion = True,
        prefix = None,
        fixed_amount = 0,
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]

        # # Calculate zscore of the ratio
        # transform = bt.indicators.OLS_TransformationN(self.data0, self.data1, period=self.p.period)
        # # A positive spread means data0 is greater than data1
        # # https://github.com/mementum/backtrader/blob/b853d7c90b6721476eb5a5ea3135224e33db1f14/backtrader/indicators/ols.py#L70C24-L79C1
        # self.spread = transform.spread
        # self.zscore = transform.zscore

        self.spread = self.data0 - self.data1
        stddev = bt.indicators.StandardDeviation(self.spread, period=self.p.period)
        self.zscore = (self.spread - bt.indicators.MovingAverageSimple(self.spread, period=self.p.period)) / stddev

        self.storagetxt = (
            f"result/gridsearch/{self.p.prefix}_{self.data0._name}_{self.data1._name}_"
            f"O{int(self.p.OPEN_THRE*10)}C{int(self.p.CLOS_THRE*10)}P{self.p.period}.csv"
        )
        '''
        2023 Oct 18
        Although the ideal case is to write separate pandas dataframes for each execution
        However, we are agnostic about whether the order hits a "margin" or a sucess execution
        Hence I have to wait for some results from `notify_order`
        '''
        # df_trading = pd.Dataframe()
        
        # 2023-Oct-08, Gave up on using custom Indicator, mismatch between docs & source code
        # spread, zscore = ZscoreIndicator(self.data0, self.data1, period=self.p.period)

        # We should have separate kelly criteria calculation for different direction
        self.kc_f = KellyCriterionIndicator(self.spread, period=self.p.period)

        # Actually it is better to use `notify_trade`, but I can't really find much doc on it
        # def notify_order(self, order):
        #     if self.p.verbose:
        #         with open(self.storagetxt, "a") as f:
        #             if order.status in [order.Submitted, order.Accepted]:
        #                 return
        #             elif order.status == order.Completed:
        #                 if order.isbuy():
        #                     f.write(f"Buy {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
        #                 else:
        #                     f.write(f"Sell {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
        #             elif order.status in [order.Expired, order.Canceled, order.Margin]:
        #                 # f.write(f"{order.Status[order.status]}" + "\n")
        #                 return

        #         f.close()
    
        # no documentation on notify_trade!!!! so hard to crawl through source code for APIs !!!!
        # def notify_trade(self, trade):
        #     if trade.status == 2:
        #         print(trade.pnlcomm, trade.data._name)

    def next(self):
        # Time management in backtrader
        # https://www.backtrader.com/docu/timemgmt/
        current_time = self.data1.datetime.datetime()

        # Calculate the ratio between the 2 assets
        ratio = self.data0.close[0] / self.data1.close[0]
        cash = self.broker.get_cash()
        position = self.broker.getposition(self.data0).size + self.broker.getposition(self.data1).size

        # Whether to activate kelly criterion or not
        kc = self.kc_f[0] if self.p.kellycriterion else 1
        order_amount0 = (self.p.fixed_amount if self.p.fixed_amount else cash) / self.data0.close[0] * kc
        order_amount1 = (self.p.fixed_amount if self.p.fixed_amount else cash) / self.data1.close[0] * kc

        # actions: {0: short p0 long p1, 1: close, 2: long p0 short p1, 3: do nothing}
        if abs(self.zscore[0]) <= self.p.CLOS_THRE and position != 0:
            self.close(data=self.data0)
            self.close(data=self.data1)
            action = 1
    
        elif self.zscore[0] <= -self.p.OPEN_THRE and position == 0 and kc!=0:
            # purchase with Kelly Criterion
 
            self.buy(data=self.data0, size=order_amount0)
            self.sell(data=self.data1, size=order_amount1)
            action = 2

        elif self.zscore[0] >= self.p.OPEN_THRE and position == 0 and kc!=0:
            # purchase with Kelly Criterion
            purchase_amount = self.broker.get_cash()/self.data1.close[0] * kc

            self.sell(data=self.data0, size=order_amount0)
            self.buy(data=self.data1, size=order_amount1)
            action = 0
        
        else:
            action = 3

        with open(self.storagetxt, mode='a+', newline='') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow([
                current_time, 
                self.broker.get_value(),
                action
            ])

        csv_f.close()

    def stop(self):
        self.close(data=self.data0)
        self.close(data=self.data1)

        if self.p.verbose == 2:
            print(f"==================================================\n")
            print(f'Open Threshold:{self.params.OPEN_THRE}, Close Threshold:{self.params.CLOS_THRE}, period: {self.params.period}\n')
            print('Starting Value - %.2f\n' % self.broker.startingcash)
            print('Ending   Value - %.2f\n' % self.broker.getvalue())
            print(f"==================================================\n")