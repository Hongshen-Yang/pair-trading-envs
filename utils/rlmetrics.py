import pandas as pd
import numpy as np

def get_return(networthcsv):
    returns = pd.read_csv(networthcsv, names=['datetime', 'values', 'action', 'zscore', 'position'])
    # Remove non actions
    returns = returns[returns['action'] != 3]
    returns['datetime'] = pd.to_datetime(returns['datetime'], format='mixed')
    returns.set_index('datetime', inplace=True)

    returns['pnl'] = returns['values'] - returns['values'].shift(-1)
    returns['returns'] = returns['values'].pct_change()
    returns = returns.dropna()

    return returns

def get_metrics(best_return):
    
    # Yearly return
    total_return = (best_return['values'][-1])/best_return['values'][0]
    total_year = (best_return.index[-1]-best_return.index[0]).days/365.25
    cagr = total_return**(1/total_year)-1

    # Calculate total orders count
    total_orders_count = best_return.shape[0]

    long_action_count = best_return[best_return['action'] == 2].shape[0]
    short_action_count = best_return[best_return['action'] == 0].shape[0]

    # Calculate won orders count
    won_orders_count = best_return[best_return['pnl'] > 0].shape[0]

    # Calculate lost orders count
    lost_orders_count = best_return[best_return['pnl'] < 0].shape[0]

    max_win = best_return[best_return['pnl'] > 0].max()

    max_loss = best_return[best_return['pnl'] < 0].min()

    # Calculate Win/Loss order ratio
    win_loss_order_ratio = won_orders_count / lost_orders_count if lost_orders_count != 0 else np.inf

    # Calculate Avg order pnl
    avg_order_pnl = best_return['pnl'].mean()

    # Calculate Avg order pnl won
    avg_order_pnl_won = best_return[best_return['returns'] > 0]['pnl'].mean()

    # Calculate Avg order pnl lost
    avg_order_pnl_lost = best_return[best_return['returns'] < 0]['pnl'].mean()

    # Calculate Avg long order pnl
    avg_long_order_pnl = best_return[best_return['action'] == 2]['pnl'].mean()

    # Calculate Avg short order pnl
    avg_short_order_pnl = best_return[best_return['action'] == 0]['pnl'].mean()

    # Print the calculated indices
    print("Compound annual growth rate:", format(cagr, ".00%"))
    print("Total orders count:", total_orders_count)
    print("Total long action:", won_orders_count)
    print("Total short action:", lost_orders_count)
    print("Won orders count:", won_orders_count)
    print("Lost orders count:", lost_orders_count)
    print("Win/Loss order ratio:", win_loss_order_ratio)
    print("Avg order pnl:", avg_order_pnl)
    print("Avg order pnl won:", avg_order_pnl_won)
    print("Avg order pnl lost:", avg_order_pnl_lost)
    print("Avg long order pnl:", avg_long_order_pnl)
    print("Avg short order pnl:", avg_short_order_pnl)