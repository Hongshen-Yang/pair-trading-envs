import pandas as pd
import numpy as np

def get_return(networthcsv):
    returns = pd.read_csv(networthcsv, names=['datetime', 'values', 'action', 'zscore', 'position', 'price0', 'price1'])
    # Remove non actions
    returns = returns[returns['action'] != 3]
    returns['datetime'] = pd.to_datetime(returns['datetime'], format='mixed')
    returns.set_index('datetime', inplace=True)

    returns['pnl'] = returns['values'].diff()
    returns['returns'] = returns['values'].pct_change()
    returns = returns.dropna()

    return returns

def get_metrics(best_return):
    total_return = (best_return['values'][-1])/best_return['values'][0]
    total_year = (best_return.index[-1]-best_return.index[0]).days/365.25 # Yearly return
    cagr = total_return**(1/total_year)-1
    total_orders_count = best_return.shape[0] # Calculate total orders count
    long_action_count = best_return[best_return['action'] == 2].shape[0]
    short_action_count = best_return[best_return['action'] == 0].shape[0]
    won_orders_count = best_return[best_return['pnl'] > 0].shape[0] # Calculate won orders count
    lost_orders_count = best_return[best_return['pnl'] < 0].shape[0] # Calculate lost orders count
    max_win = best_return[best_return['pnl'] > 0].max()['pnl']
    max_loss = best_return[best_return['pnl'] < 0].min()['pnl']
    win_loss_order_ratio = won_orders_count / lost_orders_count if lost_orders_count != 0 else np.inf # Calculate Win/Loss order ratio
    avg_order_pnl = best_return['pnl'].mean() # Calculate Avg order pnl
    avg_order_pnl_won = best_return[best_return['pnl'] > 0]['pnl'].mean() # Calculate Avg order pnl won
    avg_order_pnl_lost = best_return[best_return['pnl'] < 0]['pnl'].mean() # Calculate Avg order pnl lost
    avg_long_order_pnl = best_return[(best_return.shift(1)['action']==2)&(best_return['action']==1)]['pnl'].mean() # Calculate Avg long order pnl
    avg_short_order_pnl = best_return[(best_return.shift(1)['action']==0)&(best_return['action']==1)]['pnl'].mean() # Calculate Avg short order pnl

    # Print the calculated indices
    print("Compound annual growth rate:", format(cagr, ".00%"))
    print("Total orders count:", total_orders_count)
    print("Total long action:", long_action_count)
    print("Total short action:", short_action_count)
    print("Won orders count:", won_orders_count)
    print("Lost orders count:", lost_orders_count)
    print("Win/Loss order ratio:", win_loss_order_ratio)
    print("Max win:", max_win)
    print("Max Loss:", max_loss)
    print("Avg order pnl:", avg_order_pnl)
    print("Avg order pnl won:", avg_order_pnl_won)
    print("Avg order pnl lost:", avg_order_pnl_lost)
    print("Avg long order pnl:", avg_long_order_pnl)
    print("Avg short order pnl:", avg_short_order_pnl)