import os
import zipfile
import pandas as pd

def read2df(symbols, freqs):
    # List to store individual DataFrames
    dfs = []

    if symbols is None:
        klines_path = os.path.abspath(f'./binance-public-data/python/data/spot/monthly/klines/')
        symbols = [folder for folder in os.listdir(klines_path)]

    # Loop through each freq
    for freq in freqs.keys():
        # Loop through each symbol
        rawdfs = []
        
        for symbol in symbols:
            directory = os.path.abspath(f'./binance-public-data/python/data/spot/monthly/klines/{symbol}/{freq}/')
            
            # Loop through each zip file in the directory
            for file_name in os.listdir(directory):
                if file_name.endswith('.zip'):
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
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')

        df = df.sort_values(['time', 'tic', 'itvl'],ignore_index=True)

        dfs.append(df)
    
    return dfs

if __name__ == '__main__':
    print(len(read2df(None, {'1d': 1440})))