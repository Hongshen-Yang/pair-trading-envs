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

class CointnCorr():
    def __init__(self, dfs, freqs):
        self.dfs = dfs
        self.freqs = freqs
        self.daily_minutes = 1440

    def cointncorr(self):
        pairs = list(combinations(self.dfs[0]['tic'].unique(), 2))
        dfs_res_det = {"_".join(pair):{key: {'coint':[], 'corr':[]} for key in self.freqs} for pair in pairs}
        dfs_res = {"_".join(pair):{key: {} for key in self.freqs} for pair in pairs}

        for i, (freq, f) in enumerate(self.freqs.items()):
            print(f"calculating {freq}")
            batch_size = int(self.daily_minutes / f)
            df_pivot = self.dfs[i].pivot(index='time', columns='tic', values='close').dropna()

            for j in tqdm(range(batch_size, len(df_pivot), 1), desc=f"Calculating {freq}"):
                batch_df = df_pivot.iloc[j-batch_size:j]

                for pair in pairs:
                    
                    first_ele = batch_df[pair[0]]
                    second_ele = batch_df[pair[1]]

                    _, pvalue, _ = coint(first_ele, second_ele)
                    corr = np.corrcoef(first_ele, second_ele)[0][1]

                    pair_ = "_".join(pair)
                    dfs_res_det[pair_][freq]['coint'].append(pvalue < 0.1)
                    dfs_res_det[pair_][freq]['corr'].append(corr)

            for pair in pairs:
                pair_ = "_".join(pair)
                dfs_res[pair_][freq] = {
                    'coint': np.average(dfs_res_det[pair_][freq]['coint']),
                    'corr': np.average(dfs_res_det[pair_][freq]['corr'])
                }

            # print(f"{df_pivot.columns[0]} & {df_pivot.columns[1]} freqs: {freq}, the coint% is {coint_avg}, the avg corr is {corr_avg}")

        return dfs_res

    def tabulate(self):
        dfs_res = self.cointncorr()
        tables = {pair: pd.DataFrame(dfs_res[pair]) for pair in dfs_res}
        return tables