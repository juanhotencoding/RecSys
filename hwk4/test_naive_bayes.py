# Homework 4 solution
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

from naive_bayes import NaiveBayesRecommender
import unittest
import pandas as pd


class test_NaiveBayes(unittest.TestCase):

    _algo = None
    _ratings = None
    _features = None

    def setUp(self):
        self._features = pd.read_csv('test_features.csv')
        self._algo = NaiveBayesRecommender(self._features, thresh=2.9, alpha=0.01, beta=0.02)

        self._ratings = pd.read_csv('test.csv')
        self._ratings.columns = ['user', 'item', 'rating']

        self._algo.fit(self._ratings)

    # Test reset
    def test_reset(self):
        nb_table = self._algo._nb_table
        nb_table.reset()
        self.assertEqual(0, nb_table.user_count(1, liked=True), "Reset did not work.")

    # Test handling for zero scores
    def test_zeros(self):
        # Set all of user 1's counts to zero
        nliked_cond_table1 = self._algo._nb_table.nliked_cond_table[1]
        for key in nliked_cond_table1.keys():
            nliked_cond_table1[key] = 0

        self._algo._nb_table.beta = 0

        # Should NOT raise divison by zero
        self._algo.score_item(1, 1)


    # Test values in liked and nliked tables
    # User 1 liked 2 and disliked 2
    # User 3 liked 1 and disliked 2
    def test_liked_tables(self):
        nb_table = self._algo._nb_table
        self.assertEqual(2, nb_table.user_count(1, liked=True), "User 1 liked count incorrect")
        self.assertEqual(2, nb_table.user_count(1, liked=False), "User 1 disliked count incorrect")
        self.assertEqual(1, nb_table.user_count(3, liked=True), "User 3 liked count incorrect")
        self.assertEqual(2, nb_table.user_count(3, liked=False), "User 3 disliked count incorrect")

    # Test values in linked and nliked cond tables
    # User 1 liked items with Feature D 2 times
    # User 1 disliked items with Feature J 1 time
    def test_cond_tables(self):
        nb_table = self._algo._nb_table
        self.assertEqual(2, nb_table.user_feature_count(1, 'D', liked=True),
                         "User 1 liked feature count D incorrect")
        self.assertEqual(1, nb_table.user_feature_count(1, 'J', liked=False),
                         "User 1 disliked feature count J incorrect")

    # Test liked and nliked probability calculation
    def test_liked_prob(self):
        nb_table = self._algo._nb_table
        self.assertAlmostEqual(0.5, nb_table.user_prob(1, liked=True), 5,
                               "User 1 liked probability incorrect")
        # TODO: HOMEWORK 4. Put correct calculated value here
        self.assertAlmostEqual(0, nb_table.user_prob(3, liked=True), 5,
                               "User 3 liked probability incorrect")
        self.assertAlmostEqual(0.5, nb_table.user_prob(1, liked=False), 5,
                               "User 1 disliked probability incorrect")
        # TODO: HOMEWORK 4. Put correct calculated value here
        self.assertAlmostEqual(0, nb_table.user_prob(5, liked=False), 5,
                               "User 5 disliked probability incorrect")

    # Test conditional probability calculations
    def test_cond_prob(self):
        nb_table = self._algo._nb_table
        # TODO: HOMEWORK 4. Put correct calculated value here
        self.assertAlmostEqual(0, nb_table.user_feature_prob(1, 'D', liked=True), 5,
                               "User 1 liked feature prob D incorrect")
        self.assertAlmostEqual(0.009803922, nb_table.user_feature_prob(1, 'G', liked=False), 5,
                               "User 1 disliked feature prob G incorrect")

    # Test score User 3 Item 5
    def test_pred1(self):
        score = self._algo.score_item(3, 5)
        self.assertAlmostEqual(2.67090, score, 5,
                               'User 3 item 5 prediction incorrect')

    # Test score User 1 Item 5
    def test_pred2(self):
        score = self._algo.score_item(1, 5)
        # TODO: HOMEWORK 4. Put correct calculated value here
        self.assertAlmostEqual(0, score, 5,
                               'User 1 item 5 prediction incorrect')

    # Test score User 5 Item 2
    def test_pred3(self):
        score = self._algo.score_item(5, 2)
        # TODO: HOMEWORK 4. Put correct calculated value here
        self.assertAlmostEqual(0, score, 5,
                               'User 5 item 2 prediction incorrect')

    def test_recommend(self):
        recs = self._algo.recommend(3, 2, candidates=[4, 5])
        recs_list = list(recs['item'])
        self.assertEqual(5, recs_list[0], 'Recommendations for user 3 in wrong order.')
