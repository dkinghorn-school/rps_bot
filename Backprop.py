#!/usr/bin/env python
from __future__ import division
import numpy as np
from sklearn.neural_network import MLPRegressor as MLPR

# Data should be (Who,(R, P, S, Win?)*6)

class Backprop:
  TestInstances = [{
                      "output":"rock",
                      "player_name":"Joey",
                      "previous_throws":{
                           "opponents_throws":["rock","rock","rock"],
                           "my_throws":["paper","scissors","scissors"]
                                        }
                   },{
                      "output":"paper",
                      "player_name":"Joey",
                      "previous_throws":{
                           "opponents_throws":["paper","rock","paper"],
                           "my_throws":["scissors","scissors","scissors"]
                                        }
                   },{
                      "output":"scissors",
                      "player_name":"Mabel",
                      "previous_throws":{
                            "opponents_throws":["rock","rock","scissos"],
                            "my_throws":["paper","scissors","rock"]
                              }
                   }]
  learningRate = .01
  model = None

  def setLearningRate(self, lr):
    learningRate = lr

  def results(self, mine, theirs):
    if mine == theirs:
      return 0
    if mine == "rock":
      if theirs == "paper":
        return -1
      else:
        return 1
      
    elif mine == "paper":
      if theirs == "scissors":
        return -1
      else:
        return 1

    elif mine == "scissors":
      if theirs == "rock":
        return -1
      else:
        return 1


  def getLabelsAsArray(self, instances):
    labels = np.zeros((len(instances),3))
    for x in xrange(0, len(instances)):
      if instances[x]["output"]  == "rock":
        labels[x,0] = 1
      elif instances[x]["output"] == "paper":
        labels[x,1] = 1
      elif instances[x]["output"] == "scissors":
        labels[x,2] = 1
      else:
        print "ERROR! Label other than 'rock', 'paper', or 'scissors' encountered!"
    
    #print "\n\nLabels: \n"
    #print labels
    return labels

  def getFeaturesAsArray(self, instances):
    num_rows = len(instances)

    num_cols = self.num_players + (4 * len(instances[0]["previous_throws"]["opponents_throws"]))
    
    rows = np.zeros((num_rows,num_cols))
    
    for rownum in xrange(0,len(instances)):
      instance = instances[rownum]
      rows[rownum,self.players.index(instance["player_name"])] = 1
      past_throws = instance["previous_throws"]["opponents_throws"]
      my_throws = instance["previous_throws"]["my_throws"]
      for i in xrange(0, len(past_throws)):
        first_index = self.num_players + 4 * i
        rows[rownum,first_index] = 1 if past_throws[i] == "rock" else 0
        rows[rownum,first_index+1] = 1 if past_throws[i] == "paper" else 0
        rows[rownum,first_index+2] = 1 if past_throws[i] == "scissors" else 0
        rows[rownum,first_index+3] = self.results(my_throws[i],past_throws[i])
    #print rows
    return rows

  def train(self, instances):
    self.players = set()
    for instance in instances:
      self.players.add(instance["player_name"])
    self.num_players = len(self.players)
    self.players = list(self.players)

    labels = self.getLabelsAsArray(instances)
    features = self.getFeaturesAsArray(instances)
    #print type(features)
    newModel = MLPR(
      hidden_layer_sizes=(1000,4),
      activation='identity',
      solver='lbfgs', 
      learning_rate='constant', 
      learning_rate_init=self.learningRate,
      max_iter=200, 
      shuffle=True,  
      random_state=None, 
      tol=0.0001, 
      verbose=False,
      warm_start=True)
    newModel.n_outputs_ = np.shape(labels)[1]
    newModel.fit(features, labels)
    self.model = newModel

  def predict(self, instance):
    features = self.getFeaturesAsArray(instance)
    
    answer = self.model.predict(features[1])
    return {"rock":answer[0,0],"paper":answer[0,1],"scissors":answer[0,2]}

  def test(self):
    self.Train(self.TestInstances)
    prediction = self.Predict(self.TestInstances)
    print "\nTestResults:\n"
    print prediction

  def convertToGuess(self, p):
    result = np.zeros(3)
    a = np.max(p)
    for x in xrange(0, len(p)):
      maxIndex = x
      if p[0,x] == a:
        break
    #print maxIndex
    result[maxIndex] = 1
    return result

  def calculateTestSetAccuracy(self, testSet):
    avg = 0
    labels = self.getLabelsAsArray(testSet)
    #print labels
    features = self.getFeaturesAsArray(testSet)
    for x in range(0, len(features)):
       co = labels[x]
       pr = self.model.predict(features[x].reshape(1,-1))
       #print pr
       pr = self.convertToGuess(pr)
       if co[0] == pr[0] and co[1] == pr[1] and co[2] == pr[2]:
         avg = avg + 1

    avg = float(avg) / float(len(testSet))
    
    return avg

'''  
  def calculateTestSetAccuracy(self, testSet):
    labels = self.getLabelsAsArray(testSet)
    features = self.getFeaturesAsArray(testSet)
    reference = list(["rock", "paper", "scissors"])
    numberWin = 0
    numberTie = 0
    numberLose = 0
    totalRows = len(testSet)
    for feature in features:
      answer = self.predict(feature)
      min_val = min(answer.itervalues())
      lowest = answer.keys()[answer.values().index(min_val)]
      toWin = reference[(reference.index(lowest) - 1) % 3]
      toLose = reference[(reference.index(lowest) + 1) % 3]
      if(lowest == toWin):
        numberWin +=1
      elif(lowest == toLose):
        numberLose +=1     
      else:
        numberTie +=1
    print "-----------------------------" 
    print "Backprop Test Set Results:"
    print "Won "  + str(100 * numberWin / totalRows) + "% of the time"
    print "Tied " + str(100 * numberTie / totalRows) + "% of the time"
    print "Lost " + str(100 * numberLose/ totalRows) + "% of the time"
    print "-----------------------------"
'''


#UNCOMMENT TO TEST
#bp = Backprop()
#bp.Test()





