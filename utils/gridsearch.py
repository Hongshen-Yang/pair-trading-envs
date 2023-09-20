import math, random
from itertools import product

param_grid = {
    "param1": random.sample(range(-50, 50), 5),
    "param2": random.sample(range(-50, 50), 10),
    "param3": random.sample(range(-50, 50), 6),
}

def est(param1, param2, param3):
    val1 = math.sin(param1)
    val2 = math.cos(param2)
    val3 = math.sin(param3)
    return val1 + val2 + val3

# Try to follow the practice of sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
def gridsearch(estimator, param_grid):
    best_score = None
    best_params = None

    # Generate all possible combinations of parameters
    param_combinations = product(*param_grid.values())

    for params in param_combinations:
        param_dict = {key: val for key, val in zip(param_grid.keys(), params)}
        score = estimator(**param_dict)
        if best_score is None or score > best_score:
            best_score = score
            best_params = param_dict
    
    return best_score, best_params