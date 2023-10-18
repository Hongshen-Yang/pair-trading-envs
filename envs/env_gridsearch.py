import os
import datetime
import backtrader as bt

# Define Trading Strategy

'''
### Custom Sizer
Firstly define a sizer based on [Kelly Criterion](https://www.wikiwand.com/en/Kelly_criterion)

*Not in Use*

# Seems that Sizer can only be executed when self.buy(size=None). 
# We need to purchase amount in a certain ratio in Pair Trading.
# Therefore the Sizer is hard to implemented.
'''

class KellyCriterionSizer(bt.Sizer):
    params = (('period', 30),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data).size

        close_prices = data.close.get(size=self.p.period)
        returns = np.log(close_prices / close_prices.shift(1)).dropna() * -1

        kc_p = len(returns[returns > 0]) / len(returns)
        kc_q = 1 - kc_p
        kc_a = -(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1e-5
        kc_b = (returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 1e-5

        f = min(max((kc_p/kc_a - kc_q/kc_b), 0), 1)
        
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
        returns = spreads.pct_change() * -1

        kc_p = len(returns[returns > 0]) / len(returns)
        kc_q = 1 - kc_p
        kc_a = -(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1e-5
        kc_b = (returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 1e-5
        
        kc_f = min(max((kc_p/kc_a - kc_q/kc_b), 0), 1)
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
        residuals = sm.OLS(p0, sm.add_constant(p1)).fit().resid
        zscores = stats.zscore(residuals)
        self.lines.spread[0] = 2
        self.lines.zscore[0] = 1

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
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]

        # Calculate zscore of the ratio
        transform = bt.indicators.OLS_TransformationN(self.data1, self.data0, period=self.p.period)
        self.spread = transform.spread
        self.zscore = transform.zscore

        self.storagetxt =  f"result/gridsearch/{self.p.prefix}_{self.data0._name}_{self.data1._name}_O{int(self.p.OPEN_THRE*10)}C{int(self.p.CLOS_THRE*10)}P{self.p.period}.txt"
        os.remove(self.storagetxt) if os.path.exists(self.storagetxt) else None
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
    def notify_order(self, order):
        if self.p.verbose:
            with open(self.storagetxt, "a") as f:
                if order.status in [order.Submitted, order.Accepted]:
                    return
                elif order.status == order.Completed:
                    if order.isbuy():
                        f.write(f"Buy {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
                    else:
                        f.write(f"Sell {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
                elif order.status in [order.Expired, order.Canceled, order.Margin]:
                    # f.write(f"{order.Status[order.status]}" + "\n")
                    return
            f.close()
    
    # no documentation on notify_trade!!!! so hard to crawl through source code for APIs !!!!
    # def notify_trade(self, trade):
    #     if trade.status == 2:
    #         print(trade.pnlcomm, trade.data._name)

    def next(self):
        # Time management in backtrader
        # https://www.backtrader.com/docu/timemgmt/
        current_time = self.data1.datetime.datetime()

        # Calculate the ratio between the 2 assets
        ratio = self.data1.close[0] / self.data0.close[0]
        cash = self.broker.get_cash()
        position = self.broker.getposition(self.data0).size + self.broker.getposition(self.data1).size
        # Whether to activate kelly criterion or not
        kc = self.kc_f[0] if self.p.kellycriterion else 1

        with open(self.storagetxt, "a") as f:
            if abs(self.zscore[0]) <= self.p.CLOS_THRE and position != 0:
                if self.p.verbose:
                    f.write(f"---- Close Position @ {current_time} ----" + "\n")

                self.close(data=self.data0)
                self.close(data=self.data1)
        
            elif self.zscore[0] <= -self.p.OPEN_THRE and position == 0 and kc!=0:
                if self.p.verbose:
                    f.write(f"---- Open Position @ {current_time} ----" + "\n")

                # purchase with Kelly Criterion
                purchase_amount = self.broker.get_cash()/self.data0.close[0] * kc

                self.sell(data=self.data1, size=purchase_amount/ratio)
                self.buy(data=self.data0, size=purchase_amount)

            elif self.zscore[0] >= self.p.OPEN_THRE and position == 0 and kc!=0:
                if self.p.verbose:
                    f.write(f"---- Open Position @ {current_time} ----\n")
                
                # purchase with Kelly Criterion
                purchase_amount = self.broker.get_cash()/self.data1.close[0] * kc

                self.sell(data=self.data0, size=purchase_amount*ratio)
                self.buy(data=self.data1, size=purchase_amount)

        f.close()

    def stop(self):
        self.close(data=self.data0)
        self.close(data=self.data1)
        
        with open(self.storagetxt, "a") as f:
            f.write(f"==================================================\n")
            f.write(f'Open Threshold:{self.params.OPEN_THRE}, Close Threshold:{self.params.CLOS_THRE}, period: {self.params.period}\n')
            f.write('Starting Value - %.2f\n' % self.broker.startingcash)
            f.write('Ending   Value - %.2f\n' % self.broker.getvalue())
            f.write(f"==================================================\n")
        f.close()

        if self.p.verbose == 2:
            print(f"==================================================\n")
            print(f'Open Threshold:{self.params.OPEN_THRE}, Close Threshold:{self.params.CLOS_THRE}, period: {self.params.period}\n')
            print('Starting Value - %.2f\n' % self.broker.startingcash)
            print('Ending   Value - %.2f\n' % self.broker.getvalue())
            print(f"==================================================\n")