#!/usr/bin/env python
"""Tests the hyperparameters for SVM"""

import pickle

from controller import Controller
from svm_bot import SVMBot


def main():
    printstrings = []
    c = [0.1, 1, 5, 10]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [2, 3, 4, 5, 6]
    epsilon = [1e-4, 1e-3, 1e-2]
    for c_val in c:
        for eps_val in epsilon:
            for kern_val in kernel:
                if kern_val == 'poly':
                    for deg_val in degree:
                        avg_acc = 0
                        total_iters = 0
                        for _ in xrange(0, 100):
                            ctrl = Controller(SVMBot(c_val, kern_val, deg_val, eps_val))
                            avg_acc += ctrl.train(.25)
                            total_iters += 1
                        avg_acc /= total_iters
                        printstr = (avg_acc, 'acc: ' + str(avg_acc) +
                                             ', c: ' + str(c_val) +
                                             ', epsilon: ' + str(eps_val) +
                                             ', kernel: ' + str(kern_val) +
                                             ', degree: ' + str(deg_val))
                        printstrings.append(printstr)
                        print printstr[1]
                else:
                    avg_acc = 0
                    total_iters = 0
                    for _ in xrange(0, 100):
                        ctrl = Controller(SVMBot(c_val, kern_val, 3, eps_val))
                        avg_acc += ctrl.train(.25)
                        total_iters += 1
                    avg_acc /= total_iters
                    printstr = (avg_acc, 'acc: ' + str(avg_acc) +
                                         ', c: ' + str(c_val) +
                                         ', epsilon: ' + str(eps_val) +
                                         ', kernel: ' + str(kern_val))
                    printstrings.append(printstr)
                    print printstr[1]

    filename = 'svm_hyperparam.results'
    printstrings = sorted(printstrings, key=lambda x: x[0], reverse=True)
    with open(filename, 'wb') as ofh:
        pickle.dump(printstrings, ofh)
    print '\nprinting top 10 strings:\n'
    for i in xrange(0, 10):
        print printstrings[i][1]


if __name__ == '__main__':
    main()
