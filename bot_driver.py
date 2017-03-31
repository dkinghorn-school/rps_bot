#!/usr/bin/env python

import json

from svm_bot import SVMBot


def main():
    """Get the data and predict what will come next"""
    # Hyperparameter
    n = 5
    # Get the data
    with open('data/connordata.json') as ifh:
        data = json.load(ifh)
    # Transform the data
    opp_name = data['player2']
    player_throws = data['throws']['player1']
    opp_throws = data['throws']['player2']
    instances = []
    for i in range(len(player_throws)-n-1):
        my_throws = player_throws[i:i+n]
        other_throws = opp_throws[i:i+n]
        next_other_throw = opp_throws[i+n]
        instance = {'player_name': opp_name,
                    'output': next_other_throw,
                    'previous_throws': {
                        'opponents_throws': other_throws,
                        'my_throws': my_throws
                    }}
        instances.append(instance)
    # Train the model
    bot = SVMBot()
    bot.fit(instances)
    i = len(player_throws)-n-1
    my_throws = player_throws[i:i+n]
    other_throws = opp_throws[i:i+n]
    instance = {'player_name': opp_name,
                'previous_throws': {
                    'opponents_throws': other_throws,
                    'my_throws': my_throws
                }}
    probs = bot.predict(instance)
    print(probs)


if __name__ == '__main__':
    main()
