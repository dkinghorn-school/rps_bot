#!/usr/bin/env python

import json
import numpy as np

from svm_bot import SVMBot


def convert_throw(throw_str):
    """Returns a one-hot representation of a throw"""
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
    """Returns a one-hot representation of the data"""
    one_hot_arr = []
    for throw in throws:
        one_hot_arr.extend(convert_throw(throw))
    return one_hot_arr


def main():
    """Get the data and predict what will come next"""
    n = 5
    with open('data/connordata.json') as ifh:
        data = json.load(ifh)
    X = []
    y = []
    for i in range(len(data['throws']['player1'])-n-1):
        my_throws = data['throws']['player1'][i:i+n]
        my_throws_one_hot = one_hot(my_throws)
        next_other_throw = data['throws']['player2'][i+n]
        X.append(my_throws_one_hot)
        y.append(next_other_throw)
    bot = SVMBot()
    bot.fit(X, y)
    last_n_throws = data['throws']['player1'][-n:]
    last_n_throws_one_hot = one_hot(last_n_throws)
    next_val = bot.predict([last_n_throws_one_hot])
    print(bot.classifier.classes_)
    print(next_val)


if __name__ == '__main__':
    main()
