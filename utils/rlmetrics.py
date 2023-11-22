import pandas as pd

def get_return(networthcsv):
    returns = pd.read_csv(networthcsv, names=['datetime', 'values', 'action'])
    returns['datetime'] = pd.to_datetime(returns['datetime'], format='mixed')
    returns.set_index('datetime', inplace=True)

    returns['returns'] = returns['values'].pct_change()
    returns = returns.dropna()

    return returns

def get_metrics(best_return):
    # Calculate total orders count
    total_orders_count = best_return.shape[0]

    long_action_count = best_return[best_return['action'] == 2].shape[0]
    short_action_count = best_return[best_return['action'] == 0].shape[0]

    # Calculate won orders count
    won_orders_count = best_return[best_return['returns'] > 0].shape[0]

    # Calculate lost orders count
    lost_orders_count = best_return[best_return['returns'] < 0].shape[0]

    # Calculate Win/Loss order ratio
    win_loss_order_ratio = won_orders_count / lost_orders_count if lost_orders_count != 0 else np.inf

    # Calculate Avg order pnl
    avg_order_pnl = best_return['returns'].mean()

    # Calculate Avg order pnl won
    avg_order_pnl_won = best_return[best_return['returns'] > 0]['returns'].mean()

    # Calculate Avg order pnl lost
    avg_order_pnl_lost = best_return[best_return['returns'] < 0]['returns'].mean()

    # Calculate Avg long order pnl
    avg_long_order_pnl = best_return[best_return['action'] == 2]['returns'].mean()

    # Calculate Avg short order pnl
    avg_short_order_pnl = best_return[best_return['action'] == 0]['returns'].mean()

    # Print the calculated indices
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