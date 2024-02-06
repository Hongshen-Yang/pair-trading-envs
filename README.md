Still under development...

# Objective
This is a project to construct a trading bot with with `[backtrader](https://www.backtrader.com/)` framework with multi-leg Pair Trading.
It follows Gatev's work on Pair Trading [1] by constructed a customized Backtrader env (/envs/env_gridsearch.py)

Then a series of follow-ups on Reinforcement Learning is experimented, similar rules but on customised [gymnasium](https://gymnasium.farama.org/index.html) environment (\envs\env_rl_*)


## Python environments
Are stored in `environment.yml`

## Trading entries
Start from `params.py` for configurations.

Then `trade_gridsearch.ipynb` for grid search the most suitable hyperparams.

Lastly, `trade_RL_*` for reinforcement learning notebooks

## References

[1] Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), 797-827.