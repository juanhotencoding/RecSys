# Homework 4 solution
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import UnratedItemCandidateSelector

import numpy as np
import pandas as pd


class NaiveBayesRecommender(Recommender):

    _count_tables = {}
    _item_features = {}
    _nb_table = None
    _min_float = np.power(2.0, -149)

    def __init__(self, item_features=None, thresh=3.0, alpha=0.01, beta=0.01):
        self._item_features = item_features
        self.selector = UnratedItemCandidateSelector()
        self._nb_table = NaiveBayesTable(thresh, alpha, beta)

    def fit(self, ratings, *args, **kwargs):
        # Must fit the selector
        self.selector.fit(ratings)

        self._nb_table.reset()
        # For each rating
        for user, item, rating in ratings.itertuples(index=False):
            # Get associated item features
            features = self.get_features_list(item)
            # Update NBTable
            self._nb_table.process_rating(user, rating, features)

    # Should return ordered data frame with items and score
    def recommend(self, user, n=None, candidates=None, ratings=None):
        if n is None or n == 0:
            return pd.DataFrame({'item': []})

        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        scores = []
        # for each candidate
        for candidate in candidates:
            score = self.score_item(user, candidate)

            # Build list
            scores.append({'item': candidate, 'score': score})

        # Turn result into data frame
        recs_df = pd.DataFrame(scores)

        # Retain n largest scoring rows
        recs_df = recs_df.nlargest(n, 'score', keep='all')

        # Sort by score
        recs_df = recs_df.sort_values(by='score', ascending=False)

        # return data frame
        return recs_df

    def get_features_list(self, item):
        item_info = self._item_features[self._item_features['item'] == item]
        features = list(item_info['feature'])
        return features

    def score_item(self, user, item):
        # get the features
        features = self.get_features_list(item)
        # initialize the scores
        score_liked = self._nb_table.user_prob(user, True)
        score_nliked = self._nb_table.user_prob(user, False)

        conds_liked = {}
        conds_nliked = {}

        # for each feature
        for feature in features:
            # compute product with conditional probability
            score_liked *= self._nb_table.user_feature_prob(user, feature, True)
            conds_liked[feature] = self._nb_table.user_feature_prob(user, feature, True)
            score_nliked *= self._nb_table.user_feature_prob(user, feature, False)
            conds_nliked[feature] = self._nb_table.user_feature_prob(user, feature, False)

        # Handle the case when scores go to zero. Especially an issue for score_nliked
        score_nliked = self.ensure_minimum_score(score_nliked)
        score_liked = self.ensure_minimum_score(score_liked)

        # Compute log-likelihood
        ratio = score_liked / score_nliked
        score = self.ensure_minimum_score(ratio)
        score = np.log(ratio)
        return score

    def get_params(self, deep=True):

        return {'item_features': self._item_features,
                'thresh': self._nb_table.thresh,
                'alpha': self._nb_table.alpha,
                'beta': self._nb_table.beta}

    def ensure_minimum_score(self, val):
        if val == 0.0:
            return self._min_float
        else:
            return val


# Helper class
class NaiveBayesTable:
    liked_cond_table = {}
    nliked_cond_table = {}
    liked_table = {}
    nliked_table = {}
    thresh = 0
    alpha = 0.01
    beta = 0.01

    def __init__(self, thresh=3.0, alpha=0.01, beta=0.01):
        self.thresh = thresh
        self.alpha = alpha
        self.beta = beta

    def reset(self):
        self.liked_cond_table = {}
        self.nliked_cond_table = {}
        self.liked_table = {}
        self.nliked_table = {}

    def user_feature_count(self, user, feature, liked=True):
        if liked:
            cond_table = self.liked_cond_table
        else:
            cond_table = self.nliked_cond_table

        if user not in cond_table:
            return 0
        elif feature not in cond_table[user]:
            return 0
        else:
            return cond_table[user][feature]

    def set_user_feature_count(self, user, feature, count, liked=True):
        if liked:
            if user not in self.liked_cond_table:
                self.liked_cond_table[user] = {}
            self.liked_cond_table[user][feature] = count

        else:
            if user not in self.nliked_cond_table:
                self.nliked_cond_table[user] = {}
            self.nliked_cond_table[user][feature] = count

    def incr_user_feature_count(self, user, feature, liked=True):
        val = self.user_feature_count(user, feature, liked)
        self.set_user_feature_count(user, feature, val+1, liked)

    def user_feature_prob(self, user, feature, liked=True):
        feature_count = self.user_feature_count(user, feature, liked)
        vote_count = self.user_count(user, liked)
        num = feature_count + self.beta
        denom = vote_count + 2 * self.beta
        return num / denom

    def user_prob(self, user, liked=True):
        num = (self.user_count(user, liked) + self.alpha)
        denom = self.user_count(user, True) + self.user_count(user, False) + 2 * self.alpha
        return num / denom

    def user_count(self, user, liked=True):
        if liked:
            if user not in self.liked_table:
                return 0
            else:
                return self.liked_table[user]
        else:
            if user not in self.nliked_table:
                return 0
            else:
                return self.nliked_table[user]

    def set_user_count(self, user, value, liked=True):
        if liked:
            self.liked_table[user] = value
        else:
            self.nliked_table[user] = value

    def incr_user_count(self, user, liked=True):
        val = self.user_count(user, liked)
        self.set_user_count(user, val+1, liked)

    def process_rating(self, user, rating, features):

        liked_rating = (rating > self.thresh)

        self.incr_user_count(user, liked=liked_rating)

        for feature in features:
            self.incr_user_feature_count(user, feature, liked=liked_rating)
