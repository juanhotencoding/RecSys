import numpy as np
from User_KNN import User_KNN

# Subclass of User_KNN, so only unique functionality needs to be implemented.
class User_KNN2(User_KNN):
    _shrinkage = 0
    _user_label = 'userId'
    _item_label = 'itemId'
    _rating_label = 'rating'

    def __init__(self, nhood_size, sim_threshold=0, shrinkage=0,
                 user_label='user', item_label='item', rating_label='score'):
        User_KNN.__init__(self, nhood_size, sim_threshold=sim_threshold)

        self._shrinkage = shrinkage
        self._user_label = user_label
        self._item_label = item_label
        self._rating_label = rating_label

    # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    # Need override because of rating label
    def compute_profile_length(self, u):
        """
        Computes the length of a user's profile vector.
        :param u: user
        :return: length
        """

        return 0

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of rating label.
    def compute_profile_means(self):

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of shrinkage calculation
    def compute_similarity_cache(self):

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of user and item labels for indexing on columns
    def fit(self, ratings):

