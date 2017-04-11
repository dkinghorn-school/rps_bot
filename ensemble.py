#!/usr/bin/env python
"""This class is an ensemble of Backprop, Naive Bayes, and SVM"""

import random

from Backprop import Backprop
from naive_bayes import naive_bayes
from svm_bot import SVMBot


class Ensemble:
    def __init__(self):
        self.num_models = 1
        self.bps = []
        # self.nbs = []
        self.svms = []
        for i in xrange(0, self.num_models):
            self.bps.append(Backprop())
            # self.nbs.append(naive_bayes())
            self.svms.append(SVMBot())

    def train(self, instances):
        for i in xrange(0, self.num_models):
            self.bps[i].train(instances)
            # self.nbs[i].train(instances)
            self.svms[i].train(instances)

    def calculateTestSetAccuracy(self, testset):
        correct = 0
        total = 0
        for inst in testset:
            # Use this if you want to use probabilities.
            # Naive Bayes seems to overpower the others.
            probs = self.get_probs(inst)
            rock_prob = probs['rock']
            paper_prob = probs['paper']
            scissors_prob = probs['scissors']
            if rock_prob > paper_prob and rock_prob > scissors_prob:
                if inst['output'] == 'rock':
                    correct += 1
            elif paper_prob > rock_prob and paper_prob > scissors_prob:
                if inst['output'] == 'paper':
                    correct += 1
            else:
                if inst['output'] == 'scissors':
                    correct += 1
            # Use this if you want to have the models vote.
            # This seems to do better
            # vote = self.get_vote(inst)
            # if vote == inst['output']:
            #     correct += 1
            total += 1
        return float(correct) / float(total)

    def get_probs(self, inst):
        rock_prob = 0
        paper_prob = 0
        scissors_prob = 0
        for i in xrange(0, self.num_models):
            bp_pred = self.bps[i].predict([inst])
            # nb_pred = self.nbs[i].predict([inst])
            nb_pred = {'rock': 0, 'paper': 0, 'scissors': 0}
            svm_pred = self.svms[i].predict(inst)
            rock_prob += bp_pred['rock'] + nb_pred['rock'] + svm_pred['rock']
            paper_prob += bp_pred['paper'] + nb_pred['paper'] + svm_pred['paper']
            scissors_prob += bp_pred['scissors'] + nb_pred['scissors'] + svm_pred['scissors']
        return {'rock': rock_prob/self.num_models,
                'paper': paper_prob/self.num_models,
                'scissors': scissors_prob/self.num_models}

    def get_vote(self, instance):
        votes = {'rock': 0, 'paper': 0, 'scissors': 0}
        for i in xrange(0, self.num_models):
            bp_probs = self.bps[i].predict([instance])
            # nb_probs = self.nbs[i].predict([instance])
            svm_probs = self.svms[i].predict(instance)
            bp_vote = sorted(list(bp_probs.items()), key=lambda x: x[1], reverse=True)[0][0]
            # nb_vote = sorted(list(nb_probs.items()), key=lambda x: x[1], reverse=True)[0][0]
            svm_vote = sorted(list(svm_probs.items()), key=lambda x: x[1], reverse=True)[0][0]
            votes[bp_vote] += 1
            # votes[nb_vote] += 1
            votes[svm_vote] += 1
        max_vote = sorted(list(votes.items()), key=lambda x: x[1], reverse=True)[0][0]
        return max_vote
