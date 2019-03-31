# Homework 3
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

import logging
from lenskit.algorithms import Predictor
from lenskit.algorithms.basic import UnratedItemCandidateSelector
from hwk3_util import my_clone as my_clone
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from functools import reduce
_logger = logging.getLogger(__name__)


class WeightedHybrid(Predictor):
    """

    """

    # HOMEWORK 3 TODO: Follow the constructor for Fallback, which can be found at
    # https: // github.com / lenskit / lkpy / blob / master / lenskit / algorithms / basic.py
    # Note that you will need to
    # -- Check for agreement between the set of weights and the number of algorithms supplied.
    # -- You should clone the algorithms with hwk3_util.my_clone() and store the cloned version.
    # -- You should normalize the weights so they sum to 1.
    # -- Keep the line that set the `selector` function.

    algorithms = []
    weights = []

    def __init__(self, algorithms, weights):
        """
        Args:
            algorithms: a list of component algorithms.  Each one will be trained.
            weights: weights for each component to combine predictions.
        """
        # HWK 3: Code here
        # print(f"Number of Algorithms: {len(algorithms)}, Number of Algorithms: {len(weights)}")
        # print(f"Weights {weights}")
        if len(algorithms) == len(weights):
            self.algorithms = [my_clone(algo) for algo in algorithms]
            # algorithms.append(my_clone(self.algorithms))
            self.weights = np.abs(weights / np.sum(weights))
            # print(self.weights)
            
        self.selector = UnratedItemCandidateSelector()

    def clone(self):
        return WeightedHybrid(self.algorithms, self.weights)

    # HOMEWORK 3 TODO: Complete this implementation
    # Will be similar to Fallback. Must also call self.selector.fit()
    def fit(self, ratings, *args, **kwargs):

        # HWK 3: Code here
        for algo in self.algorithms:
            algo.fit(ratings, *args, **kwargs)
        
        self.selector.fit(ratings)
        
        return self

    def candidates(self, user, ratings):
        return self.selector.candidates(user, ratings)

    # HOMEWORK 3 TODO: Complete this implementation
    # Computes the weighted average of the predictions from the component algorithms
    def predict_for_user(self, user, items, ratings=None):
        
        # make predictions list for the prediction of each algorithm
        preds = [algo.predict_for_user(user, items, ratings=ratings) for algo in self.algorithms]        
        
        # pair up weights and predictions as tuple pairs to sum up later
        pairs = list(zip(preds, self.weights))
        
        # user reduce function to sum linear combination of weights and predictions 
        preds = reduce(lambda x, y: x.add(y), [i[0]*i[1] for i in pairs])       
        
        return preds

        

    def __str__(self):
        return 'Weighted([{}])'.format(', '.join(self.algorithms))
