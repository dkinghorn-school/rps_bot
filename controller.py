from os import listdir
from os.path import isfile, join
import json
import random

class Controller():
    def __init__(self, model, history_length=5):
        self.model = model
        self.history_length = 5
        print('starting classifier')

    def train(self, testSetPercentage, file_name=None):
        path = 'data'
        if file_name is not None:
            files = [file_name]
        else:
            files = [f for f in listdir(path) if isfile(join(path, f))]
        models = []
        for f in files:
            with open(join(path,f)) as json_data:
                data = json.load(json_data)
                models = models + data
        instances = []
        for model in models:
            instances = instances + self.get_instances(model)
        random.shuffle(instances)
        testSet = self.getTestSet(instances, testSetPercentage)
        instances = self.getInstances(instances, testSetPercentage)
        self.instances = instances
        self.model.train(self.instances)
        testSetAccuracy = self.model.calculateTestSetAccuracy(testSet)
        print '% of instance set given to set set: ' + str(testSetPercentage)
        print 'testSetAccuracy: ' + str(testSetAccuracy)
    def getTestSet(self, instances, testPercent):
        test = []
        length = len(instances)
        min = (1 - testPercent) * length
        for x in range (int(min), length):
            test.append(instances[x])
        return test
    def getInstances(self, instances, testPercent):
        length = len(instances)
        min = (1 - testPercent) * length
        for x in range(int(min), length):
            instances.pop()
        return instances
    def get_single_instance(self, index, opponents_throws, my_throws, my_name):
        """
        index is the point of the last turn thrown, the output will be at one greater than the index
        """
        return {
            "output": my_throws[index],
            "player_name": my_name,
            "previous_throws":{
                "opponents_throws": opponents_throws[index-self.history_length:index],
                "my_throws": my_throws[index-self.history_length:index]
            }
        }

    def get_instances(self, model):
        instances = []
        player1_throws = model['throws']['player1']
        player2_throws = model['throws']['player2']
        player1_name = model['player1']
        player2_name = model['player2']
        last_index = len(player1_throws) - 1
        while last_index >= self.history_length:
            instances = instances + [self.get_single_instance(last_index, player1_throws, player2_throws, player2_name)]
            instances = instances + [self.get_single_instance(last_index, player2_throws, player1_throws, player1_name)]
            last_index = last_index - 1
        return instances


    def predict(self, instance):
        player1_throws = instance['throws']['player1']
        player2_throws = instance['throws']['player2']
        player1_name = instance['player1']
        player2_name = instance['player2']
        last_index = len(player1_throws) - 1
        return self.model.predict(self.get_single_instance(last_index, player2_throws, player1_throws, player1_name))

