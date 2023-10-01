from itertools import combinations
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd

def cointncorr(dfs, freqs):
    daily_minutes = 1440
    dfs_res = {}
    for i, (freq, f) in enumerate(freqs.items()):
        batch_size = int(daily_minutes / f)
        df_pivot = dfs[i].pivot(index='time', columns='tic', values='close').dropna()

        asset_pairs = list(combinations(df_pivot.columns.tolist(), 2))

        result_dict = {}
        for asset_pair in asset_pairs:
            result_dict[f"{asset_pair[0]}-{asset_pair[1]}"] = {'coint': [], 'corr': []}

        for j in range(0, len(df_pivot), batch_size):
            batch_df = df_pivot.iloc[j:j+batch_size]

            for asset_pair in asset_pairs:
                
                first_ele = batch_df[asset_pair[0]]
                second_ele = batch_df[asset_pair[1]]

                _, pvalue, _ = coint(first_ele, second_ele)
                corr = np.corrcoef(first_ele, second_ele)[0][1]

                assetpair = f"{asset_pair[0]}-{asset_pair[1]}"
                result_dict[assetpair]['coint'].append(pvalue < 0.05)
                result_dict[assetpair]['corr'].append(corr)

        for asset_pair in asset_pairs:
            assetpair = f"{asset_pair[0]}-{asset_pair[1]}"
            result_dict[assetpair]['coint'] = np.average(result_dict[assetpair]['coint'])
            result_dict[assetpair]['corr'] = np.average(result_dict[assetpair]['corr'])

        dfs_res[freq] = result_dict

        # print(f"{df_pivot.columns[0]} & {df_pivot.columns[1]} freqs: {freq}, the coint% is {coint_avg}, the avg corr is {corr_avg}")

    return dfs_res