import os
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
        returns = np.log(close_prices / close_prices.shift(1)).dropna()

        kc_p = len(returns[returns > 0]) / len(returns)
        kc_a = (returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1.0
        kc_b = (-returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 1.0
        kc_q = 1 - kc_p

        f = min(max((kc_p/kc_a - kc_q/kc_b), 0), 1)

        if isbuy:
            size = cash * f / data.close[0]
        else:
            size = position * f

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
        returns = spreads.pct_change()

        kc_p = len(returns[returns > 0]) / len(returns)
        kc_q = 1 - kc_p
        kc_a = (-returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1
        kc_b = (returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 1
        
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

strategy_txt = f"result/gridsearch/strategy.txt"
os.remove(strategy_txt) if os.path.exists(strategy_txt) else None

class PairTrading(bt.Strategy):
    params = dict(
        OPEN_THRE=2,
        CLOS_THRE=0.1,
        period=30
    )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status == order.Completed:
            with open(strategy_txt, "a") as f:
                if order.isbuy():
                    f.write(f"Buy {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
                else:
                    f.write(f"Sell {order.data._name} @ price: {order.executed.price} for Qty: {order.executed.size}" + "\n")
            f.close()
        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            with open(strategy_txt, "a") as f:
                f.write(f"{order.Status[order.status]}" + "\n")
            f.close()

    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]

        # Calculate zscore of the ratio
        transform = bt.indicators.OLS_TransformationN(self.data1, self.data0, period=self.p.period)
        self.spread = transform.spread
        self.zscore = transform.zscore
        
        # 2023-Oct-08, Gave up on using custom Indicator, mismatch between docs & source code
        # spread, zscore = ZscoreIndicator(self.data0, self.data1, period=self.p.period)

        # We should have separate kelly criteria calculation for different direction
        self.kc_f_pos = KellyCriterionIndicator(self.spread, period=self.p.period)
        self.kc_f_neg = KellyCriterionIndicator(-self.spread, period=self.p.period)

    def next(self):
        # Calculate the ratio between the 2 assets
        ratio = self.data1.close[0] / self.data0.close[0]
        cash = self.broker.get_cash()
        position = self.broker.getposition(self.data0).size + self.broker.getposition(self.data1).size
        
        if abs(self.zscore[0]) < self.p.CLOS_THRE and position != 0:
            with open(strategy_txt, "a") as f:
                f.write(f"------" + "\n")
                f.write(f"close position @ {self.data1.datetime[0]}\n")
            f.close()
            self.close(data=self.data0)
            self.close(data=self.data1)
    
        elif self.zscore[0] < -self.p.OPEN_THRE and position == 0 and self.kc_f_neg[0]!=0:
            with open(strategy_txt, "a") as f:
                f.write(f"------" + "\n")
                f.write(f"open position @ {self.data1.datetime[0]}\n")
            f.close()

            # purchase with Kelly Criterion
            purchase_amount = self.broker.get_cash()/self.data0.close[0]*self.kc_f_neg[0]

            self.sell(data=self.data1, size=purchase_amount/ratio)
            self.buy(data=self.data0, size=purchase_amount)

        elif self.zscore[0] > self.p.OPEN_THRE and position == 0 and self.kc_f_pos[0]!=0:
            with open(strategy_txt, "a") as f:
                f.write(f"------\n")
                f.write(f"open position @ {self.data1.datetime[0]}\n")
            f.close()
            
            # purchase with Kelly Criterion
            purchase_amount = self.broker.get_cash()/self.data1.close[0]*self.kc_f_pos[0]

            self.sell(data=self.data0, size=purchase_amount*ratio)
            self.buy(data=self.data1, size=purchase_amount)

    def stop(self):
        self.close(data=self.data0)
        self.close(data=self.data1)
        
        print('==================================================')
        print(f'Open Threshold:{self.params.OPEN_THRE}, Close Threshold:{self.params.CLOS_THRE}, period: {self.params.period}')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

        with open(strategy_txt, "a") as f:
            f.write(f"==================================================\n")
            f.write(f'Open Threshold:{self.params.OPEN_THRE}, Close Threshold:{self.params.CLOS_THRE}, period: {self.params.period}\n')
            f.write('Starting Value - %.2f\n' % self.broker.startingcash)
            f.write('Ending   Value - %.2f\n' % self.broker.getvalue())
            f.write(f"==================================================\n")
        f.close()