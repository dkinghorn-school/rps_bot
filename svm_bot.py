#!/usr/bin/env python
"""Runs a basic rock-paper-scissors bot"""

from sklearn import svm


class SVMBot:

    def __init__(self):
        self.classifier = svm.SVC(probability=True,
                                  decision_function_shape='ovr')

    def fit(self, X, y):
        """
        Fits the Bot's classifier
        :param X: The input values (what was previously played by us,
                  what was previously played by the other player, as far
                  out as we want to use in predicting)
        :type X: [[string, string, string, ..., string]]
        :param y: The output value (what the other player played next)
        :type y: [string]
        :return: Nothing
        """
        self.classifier.fit(X, y)

    def predict(self, x):
        """
        Predicts the probabilities of the next value the other player will play
        :param x: The previous n throws
        :type x: [string, string, string, ..., string]
        :return: The probabilities of the other player's next throw
        :rtype: [float, float, float]
        """
        return self.classifier.predict_proba(x)
