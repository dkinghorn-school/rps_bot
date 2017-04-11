import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets

class naive_bayes:
    def __init__(self):
        self.model = BernoulliNB(alpha=0.0)

    def train(self, instances):
        self.players = set()
        for instance in instances:
            self.players.add(instance["player_name"])
        self.num_players = len(self.players)
        self.players = list(self.players)
        # print(len(instances))

        labels = self.getLabelsAsArray(instances)
        features = self.getFeaturesAsArray(instances)
        self.model.fit(features, labels)

    def calculateTestSetAccuracy(self, testSet):
        # print(len(testSet))
        avg = 0
        labels = self.getLabelsAsArray(testSet)
        features = self.getFeaturesAsArray(testSet)
        for x in range(0, len(features)):
            correctOutput = float(labels[x])
            arr = np.array(features[x]).reshape(1,-1)
            predicted = float(self.model.predict(arr))
            avg = avg + 1 if correctOutput == predicted else avg

        avg = float(avg) / float(len(testSet))

        return avg

    def predict(self, newData):
        features = self.getFeaturesAsArray(newData)
        answer = self.model.predict_proba(features)
        return {"rock":answer[0,0],"paper":answer[0,1],"scissors":answer[0,2]}

    def getLabelsAsArray(self, instances):
        labels = np.zeros((len(instances)))

        for x in xrange(0, len(instances)):
            if instances[x]["output"] == "rock":
                labels[x] = 0
            elif instances[x]["output"] == "paper":
                labels[x] = 1
            elif instances[x]["output"] == "scissors":
                labels[x] = 2
            else:
                print "ERROR! Label other than 'rock', 'paper', or 'scissors' encountered!"
        return labels

    def getFeaturesAsArray(self, instances):
        num_rows = len(instances)

        num_cols = self.num_players + (4 * len(instances[0]["previous_throws"]["opponents_throws"]))

        rows = np.zeros((num_rows, num_cols))

        for rownum in xrange(0, len(instances)):
            instance = instances[rownum]
            rows[rownum, self.players.index(instance["player_name"])] = 1
            past_throws = instance["previous_throws"]["opponents_throws"]
            my_throws = instance["previous_throws"]["my_throws"]
            for i in xrange(0, len(past_throws)):
                first_index = self.num_players + 4 * i
                rows[rownum, first_index] = 1 if past_throws[i] == "rock" else 0
                rows[rownum, first_index + 1] = 1 if past_throws[i] == "paper" else 0
                rows[rownum, first_index + 2] = 1 if past_throws[i] == "scissors" else 0
                # rows[rownum, first_index + 3] = self.results(my_throws[i], past_throws[i])
        # print rows
        return rows
