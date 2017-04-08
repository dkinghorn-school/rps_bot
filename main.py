from controller import Controller
from naive_bayes import naive_bayes
from Backprop import Backprop
from svm_bot import SVMBot


controller = Controller(naive_bayes())
print('\n\ntraining Naive Bayes\n\n')
controller.train(.1)

BPcontroller = Controller(Backprop())
print('\n\ntraining Backprop\n\n')
BPcontroller.train(.25)

SVMcontroller = Controller(SVMBot())
print('\n\ntraining SVM\n\n')
SVMcontroller.train(.25)
