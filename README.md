# Reinforcement Learning Pair Trading

This repository contains the open-source code for the paper Reinforcement Learning Pair Trading: A Dynamic Scaling approach[^1] 

### Rule-based Pair Trading Environment
Rule-based pair trading environments with [backtrader](https://www.backtrader.com/) framework (`env_gridsearch.ipynb`)

### Gymnasium-based Pair Trading Environment
Reinforcement Learning based environment with [gymnasium](https://gymnasium.farama.org/index.html)  (`env_rl.ipynb`)
* Fixed Amount: The bet for each trading is fixed at a certain number.
* Free Amount: The bet is dynamically measured by the each trading opportunity.

[^1]: Yang, H., & Malik, A. (2024). Reinforcement learning pair trading: A dynamic scaling approach (arXiv:2407.16103). arXiv. https://doi.org/10.48550/arXiv.2407.16103
