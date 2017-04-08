#!/usr/bin/env python
"""Runs a basic SVM rock-paper-scissors bot"""

import numpy as np
from sklearn import svm


def convert_throw(throw_str):
    """Return a one-hot representation of a throw"""
    if throw_str == 'rock':
        converted_throw = [1, 0, 0]
    elif throw_str == 'paper':
        converted_throw = [0, 1, 0]
    elif throw_str == 'scissors':
        converted_throw = [0, 0, 1]
    else:
        raise ValueError('Throw was not rock, paper, or scissors')
    return converted_throw


def one_hot(throws):
    """Return a one-hot representation of the data"""
    one_hot_arr = []
    for throw in throws:
        one_hot_arr.extend(convert_throw(throw[0]))
        one_hot_arr.extend(convert_throw(throw[1]))
    return one_hot_arr


class SVMBot:

    def __init__(self):
        self.classifier = svm.SVC(probability=True,
                                  decision_function_shape='ovr')
        self.players = []

    def train(self, instances):
        """
        Train the Bot's classifier
        :param instances: The instances to train on.
        :type instances: [{'output': str,
                           'player_name': str,
                           'previous_throws': {'opponents_throws': [str, ..., str],
                                               'my_throws: [str, ..., str]
                                              }}]
        :return: Nothing
        """
        X = []
        y = []
        for inst in instances:
            if inst['player_name'] not in self.players:
                self.players.append(inst['player_name'])
        for inst in instances:
            name_index = self.players.index(inst['player_name'])
            name_arr = np.zeros((len(self.players), 1))
            name_arr[name_index] = 1
            prev_throws = inst['previous_throws']
            throws = one_hot(list(zip(prev_throws['opponents_throws'],
                                      prev_throws['my_throws'])))
            throws.extend(name_arr)
            X.append(throws)
            y.append(inst['output'])
        self.classifier.fit(X, y)

    def predict(self, inst):
        """
        Predict the probabilities of the next value the other player will play
        :param inst: The instance
        :type inst: {'player_name': str
                         'previous_throws': {'opponents_throws': [str, ..., str],
                                             'my_throws': [str, ..., str]
                                            }}
        :return: The probabilities of the other player's next throw
        :rtype: { 'rock': float, 'paper': float, 'scissors': float }
        """
        x = []
        # Use a name_arr of all zeros if we don't know the player
        name_arr = np.zeros((len(self.players), 1))
        if inst['player_name'] in self.players:
            name_index = self.players.index(inst['player_name'])
            name_arr[name_index] = 1
        prev_throws = inst['previous_throws']
        throws = one_hot(list(zip(prev_throws['opponents_throws'],
                                  prev_throws['my_throws'])))
        throws.extend(name_arr)
        x.append(throws)
        probs = self.classifier.predict_proba(x)
        return {'rock': probs[0][1], 'paper': probs[0][0], 'scissors': probs[0][2]}

    def calculateTestSetAccuracy(self, testset):
        """
        Calculate this model's accuracy on a given test set of data

        :type testset: [{'output': str,
                         'player_name': str,
                          'previous_throws': {'opponents_throws': [str, ..., str],
                                              'my_throws: [str, ..., str]
                                             }}]
        :rtype: float
        """
        correct = 0
        total = 0
        for inst in testset:
            probs = self.predict(inst)
            if probs['rock'] > probs['paper'] and probs['rock'] > probs['scissors']:
                if inst['output'] == 'rock':
                    correct += 1
            elif probs['paper'] > probs['scissors'] and probs['paper'] > probs['rock']:
                if inst['output'] == 'paper':
                    correct += 1
            else:
                if inst['output'] == 'scissors':
                    correct += 1
            total += 1
        return float(correct) / float(total)
