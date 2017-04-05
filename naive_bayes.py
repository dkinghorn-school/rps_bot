import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
class naive_bayes:
    def __init__(self):
        self.model = GaussianNB()
    def train(self, instances):
        labels = self.getLabelsAsArray(instances)
        features = self.getFeaturesAsArray(instances)
        self.model.fit(features, labels);
    def predict(self, newData):

        predicted = self.model.predict(newData)
        # summarize the fit of the model
        print(predicted)

    def getLabelsAsArray(self, instances):
        labels = np.zeros((len(instances), 1))

        for x in xrange(0, len(instances)):
            if instances[x]["output"] == "rock":
                labels[x, 0] = 0
            elif instances[x]["output"] == "paper":
                labels[x, 0] = 1
            elif instances[x]["output"] == "scissors":
                labels[x, 0] = 2
            else:
                print "ERROR! Label other than 'rock', 'paper', or 'scissors' encountered!"
        return labels

    def getFeaturesAsArray(self, instances):
        num_rows = len(instances)

        players = set()
        for instance in instances:
            players.add(instance["player_name"])
        num_players = len(players)

        num_cols = num_players + (4 * len(instances[0]["previous_throws"]["opponents_throws"]))
        players = list(players)

        rows = np.zeros((num_rows, num_cols))

        for rownum in xrange(0, len(instances)):
            instance = instances[rownum]
            rows[rownum, players.index(instance["player_name"])] = 1
            past_throws = instance["previous_throws"]["opponents_throws"]
            my_throws = instance["previous_throws"]["my_throws"]
            for i in xrange(0, len(past_throws)):
                first_index = num_players + 4 * i
                rows[rownum, first_index] = 1 if past_throws[i] == "rock" else 0
                rows[rownum, first_index + 1] = 1 if past_throws[i] == "paper" else 0
                rows[rownum, first_index + 2] = 1 if past_throws[i] == "scissors" else 0
                # rows[rownum, first_index + 3] = self.results(my_throws[i], past_throws[i])
        # print rows
        return rows
