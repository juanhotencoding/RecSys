# Homework 3
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

import logging
from lenskit.algorithms import Predictor
from lenskit.algorithms.basic import UnratedItemCandidateSelector
from hwk3_util import my_clone
import numpy as np
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
        # -- Check for agreement between the set of weights and the number of algorithms supplied.
        if len(algorithms) == len(weights):
            self.algorithms = algorithms
            self.weights = weights
        # -- You should clone the algorithms with hwk3_util.my_clone() and store the cloned version.
        # -- You should normalize the weights so they sum to 1.
        self.weights = np.linalg.norm(weights)
        # -- Keep the line that set the `selector` function.
       
        # self.algorithms = algorithms
        # self.weights = weights
        self.selector = UnratedItemCandidateSelector()

    def clone(self):
        return WeightedHybrid(self.algorithms, self.weights)

    # HOMEWORK 3 TODO: Complete this implementation
    # Will be similar to Fallback. Must also call self.selector.fit()
    def fit(self, ratings, *args, **kwargs):

        # HWK 3: Code here
        for algo in self.algorithms:
            algo.fit(ratings, *args, *kwargs)
        return self

    def candidates(self, user, ratings):
        return self.selector.candidates(user, ratings)

    # HOMEWORK 3 TODO: Complete this implementation
    # Computes the weighted average of the predictions from the component algorithms
    def predict_for_user(self, user, items, ratings=None):
        preds = None
        # HWK 3: Code here

        return preds

    def __str__(self):
        return 'Weighted([{}])'.format(', '.join(self.algorithms))
