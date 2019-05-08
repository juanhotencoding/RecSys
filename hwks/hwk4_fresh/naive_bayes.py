# Homework 4
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder
# Edited by Juan Vargas-Murillo

from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import UnratedItemCandidateSelector

import numpy as np
import pandas as pd


class NaiveBayesRecommender(Recommender):

    # _count_tables = {}
    _item_features = None
    _nb_table = None
    _min_float = np.power(2.0, -149)

    def __init__(self, item_features=None, thresh=2.9, alpha=0.01, beta=0.01):
        self._item_features = item_features
        self.selector = UnratedItemCandidateSelector()
        self._nb_table = NaiveBayesTable(thresh, alpha, beta)

    # TODO: HOMEWORK 4
    def fit(self, ratings, *args, **kwargs):
        # Must fit the selector
        self.selector.fit(ratings)

        self._nb_table.reset()
        # For each rating
            # Get associated item features
            # Update NBTable
        # ratings.apply(lambda x: self._nb_table.process_rating(x['user'], x['rating'], self.get_features_list(x['item'])))
        for i in ratings.index:
            self._nb_table.process_rating(ratings.at[i, 'user'], ratings.at[i, 'rating'], self.get_features_list(ratings.at[i, 'item']))
            # self._nb_table.process_rating(ratings[i, 'user'], ratings[i, 'rating'], self.get_features_list(ratings[i, 'item']))

    # TODO: HOMEWORK 4
    # Should return ordered data frame with items and score
    def recommend(self, user, n=None, candidates=None, ratings=None):
        # n is None or zero, return DataFrame with an empty item column
        if n is None or n == 0:
            return pd.DataFrame({'item': []})

        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        # Initialize scores
        scores = []
        # for each candidate
        for candidate in candidates:
            # Score the candidate for the user
            score = self.score_item(user, candidate)
            # Build list of candidate, score pairs
            scores.append((candidate, score))
        # Turn result into data frame
        recs_list = pd.DataFrame(scores, columns = ['item', 'score'])
        # Retain n largest scoring rows (nlargest)
        recs_list = recs_list.nlargest(n, columns=['item', 'score'])
        # Sort by score (sort_values)
        recs_list = recs_list.sort_values(by='score', ascending=False)
        # return data frame
        return recs_list

    # TODO: HOMEWORK 4
    # Helper function to return a list of features for an item from features data frame
    def get_features_list(self, item):

        return self._item_features[self._item_features['item'] == item]['feature'].tolist()

    # TODO: HOMEWORK 4
    def score_item(self, user, item):
        # get the features
        features = self.get_features_list(item)
        # initialize the liked and nliked scores with the base probability
        liked, nliked = self._nb_table.user_prob(user, liked=True), self._nb_table.user_prob(user, liked=False)
        # for each feature
            # update scores by multiplying with conditional probability
        for feature in features:
            liked *= self._nb_table.user_feature_prob(user, feature, liked=True)
            nliked *= self._nb_table.user_feature_prob(user, feature, liked=False)
        # Handle the case when scores go to zero.
        liked = self.ensure_minimum_score(liked)
        nliked = self.ensure_minimum_score(nliked)
        # Compute log-likelihood
        score = np.log(liked/nliked)
        # Handle zero again
        score = self.ensure_minimum_score(score)
        # Return result
        return score

    # DO NOT ALTER
    def get_params(self, deep=True):

        return {'item_features': self._item_features,
                'thresh': self._nb_table.thresh,
                'alpha': self._nb_table.alpha,
                'beta': self._nb_table.beta}

    # DO NOT ALTER
    def ensure_minimum_score(self, val):
        if val == 0.0:
            return self._min_float
        else:
            return val


# TODO: HOMEWORK 4
# Helper class
class NaiveBayesTable:
    liked_cond_table = {}
    nliked_cond_table = {}
    liked_table = {}
    nliked_table = {}
    thresh = 0
    alpha = 0.01
    beta = 0.01

    # TODO: HOMEWORK 4
    def __init__(self, thresh=2.9, alpha=0.01, beta=0.01):
        self.thresh = thresh
        self.alpha = alpha
        self.beta = beta 

    # TODO: HOMEWORK 4
    # Reset all the tables
    def reset(self):
        self.liked_cond_table.clear()
        self.nliked_cond_table.clear()
        self.liked_table.clear()
        self.nliked_table.clear()

    # TODO: HOMEWORK 4
    # Return the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table: return 0 in these cases
    def user_feature_count(self, user, feature, liked=True):
        if liked:
            if self.liked_cond_table.get(user, 0) == 0:
                return 0
            if self.liked_cond_table[user].get(feature, 0) == 0:
                return 0
            return self.liked_cond_table[user][feature]
        else:
            if self.nliked_cond_table.get(user, 0) == 0:
                return 0
            if self.nliked_cond_table[user].get(feature, 0) == 0:
                return 0
            return self.nliked_cond_table[user][feature]


    # TODO: HOMEWORK 4
    # Sets the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table. Create appropriate entry or entries
    def set_user_feature_count(self, user, feature, count, liked=True):
        if liked:
            if self.liked_cond_table.get(user, 0) == 0:
                self.liked_cond_table[user] = {}
            self.liked_cond_table[user][feature] = count
        else:
            if self.nliked_cond_table.get(user, 0) == 0:
                self.nliked_cond_table[user] = {}
            self.nliked_cond_table[user][feature] = count
        


    def incr_user_feature_count(self, user, feature, liked=True):
        val = self.user_feature_count(user, feature, liked)
        self.set_user_feature_count(user, feature, val+1, liked)

    # TODO: HOMEWORK 4
    # Computes P(f|L) or P(f|~L) as the observed ratio of features and total likes/dislikes
    # Should include smooting with beta value
    def user_feature_prob(self, user, feature, liked=True):
        return (self.user_feature_count(user, feature, liked) + self.beta) / (self.user_count(user, liked) + (2 * self.beta)) if liked else (self.user_feature_count(user, feature, liked) + self.beta) / (self.user_count(user, liked) + (2 * self.beta))

    # TODO: HOMEWORK 4
    # Return the liked (disiked) count for a user (
    # Should be robust if the user is not currently in table: return 0 in this cases
    def user_count(self, user, liked=True):
        if liked:
            if self.liked_table.get(user, 0):
                return self.liked_table[user]
        else:
            if self.nliked_table.get(user, 0):
                return self.nliked_table[user]
        return 0

    # TODO: HOMEWORK 4
    # Sets the liked/disliked count for a user
    # Should be robust if the user is not currently in table. Create appropriate entry
    def set_user_count(self, user, value, liked=True):
        if liked:
            if self.liked_table.get(user, 0) == 0:
                self.liked_table[user] = 0
            self.liked_table[user] = value
        else:
            if self.nliked_table.get(user, 0) == 0:
                self.nliked_table[user] = 0
            self.nliked_table[user] = value


    def incr_user_count(self, user, liked=True):
        val = self.user_count(user, liked)
        self.set_user_count(user, val+1, liked)

    # TODO: HOMEWORK 4
    # Computes P(L) or P(~L) as the observed ratio of liked/dislike and total rated item count
    # Should include smooting with alpha value
    def user_prob(self, user, liked=True):
          
        return (self.user_count(user, liked) + self.alpha) / ((self.user_count(user, liked) + self.user_count(user, not liked)) + (2 * self.alpha)) if liked else (self.user_count(user, liked) + self.alpha) / ((self.user_count(user, liked) + self.user_count(user, not liked)) + (2 * self.alpha))

    # TODO:HOMEWORK 4
    # Update the table to take into account one new rating
    def process_rating(self, user, rating, features):

        # Determine if liked or disliked
        liked = rating > self.thresh
        # Increment appropriate count for the user
        self.incr_user_count(user, liked)

        # For each feature
            # Increment appropriate feature count for the user
        for feature in features:
            self.incr_user_feature_count(user, feature, liked)
