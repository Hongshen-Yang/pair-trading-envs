import os
import zipfile
import pandas as pd

def read2df(symbols, freqs, marketType="spot"):
    # List to store individual DataFrames
    dfs = []

    if symbols is None:
        klines_path = os.path.abspath(f'./binance-public-data/python/data/{marketType}/monthly/klines/')
        symbols = [folder for folder in os.listdir(klines_path)]

    # Loop through each freq
    for freq in freqs.keys():
        # Loop through each symbol
        rawdfs = []
        
        for symbol in symbols:
            directory = os.path.abspath(f'./binance-public-data/python/data/{marketType}/monthly/klines/{symbol}/{freq}/')
            
            # Loop through each zip file in the directory
            for file_name in os.listdir(directory):
                if file_name.endswith('.zip'):
                    zip_file_path = os.path.join(directory, file_name)

                    if os.path.exists(zip_file_path):
                        with zipfile.ZipFile(os.path.join(directory, file_name), 'r') as zip_ref:
                            # only one CSV file in each zip archive
                            csv_file = zip_ref.namelist()[0]
                            with zip_ref.open(csv_file) as csv_fp:
                                # Read the CSV data into a DataFrame
                                temp_df = pd.read_csv(csv_fp, header=None)
                                temp_df.columns = [
                                    'open_time', 'open', 'high', 'low', 'close', 'volume', 
                                    'close_time', 'quote_asset_volume', 'number_of_trades', 
                                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                                ]
                                temp_df = temp_df.rename(columns={"close_time": "time"})
                                temp_df['tic'] = symbol
                                temp_df['itvl'] = freq
                                rawdfs.append(temp_df[['time', 'open', 'high', 'low', 'close', 'volume', 'tic', 'itvl']])

        # Concatenate all symbols into a single DataFrame
        rawdf = pd.concat(rawdfs, ignore_index=True)

        # Count the number of unique 'tic' values per date
        tic_counts = rawdf.groupby('time')['tic'].nunique()

        # Filter the DataFrame to keep only rows where all 'tic' values participate
        df = rawdf[rawdf['time'].isin(tic_counts[tic_counts == len(rawdf['tic'].unique())].index)]
        # Only wanted columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'tic', 'itvl']]
        df = df[df['time']!='close_time']

        df['datetime'] = pd.to_datetime(df['time'], unit='ms')

        numeric_columns = df.columns.difference(['datetime', 'tic', 'itvl'])
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        df = df.sort_values(['time', 'tic', 'itvl'],ignore_index=True)
        df = df.drop_duplicates()
        
        dfs.append(df)
    
    return dfs

def unify_dfs(dfs, symbols, period):
    dfs[0]['close'] = dfs[0]['close'].apply(lambda x: 1/x)
    
    df0 = dfs[0][dfs[0]['tic']==symbols[0]].reset_index(drop=True)
    df1 = dfs[0][dfs[0]['tic']==symbols[1]].reset_index(drop=True)

    df0 = df0[['time', 'close', 'tic', 'itvl', 'datetime']]
    df1 = df1[['time', 'close', 'tic', 'itvl', 'datetime']]

    tic0, tic1 = df0['tic'][0], df1['tic'][0]
    df = pd.merge(df0, df1, on=['time', 'itvl', 'datetime'], suffixes=(f"0", f"1"))
    df = df.drop([f"tic0", f"tic1"], axis=1)
    df['spread'] = df[f'close0'] - df[f'close1']

    zscore = []

    for index, row in df.iterrows():
        if index <= period:
            zscore.append(0)
        else:
            df_tmp = df.iloc[index-period:index]
            zscore.append((row['spread']-df_tmp['spread'].mean())/df_tmp['spread'].std())

    df['zscore'] = zscore

    return [tic0, tic1], df

if __name__ == '__main__':
    print(len(read2df(None, {'1d': 1440})))