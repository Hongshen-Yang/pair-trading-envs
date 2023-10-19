import pandas as pd

def pyfolio_process(res):
    res_pyfolio = res[0].analyzers.pyfolio.get_analysis()
    res_pyfolio = pd.Series(res_pyfolio['returns'])
    res_pyfolio.index = pd.to_datetime(res_pyfolio.index)
    res_pyfolio = res_pyfolio.astype('float32')
    return res_pyfolio