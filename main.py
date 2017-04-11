from controller import Controller
from naive_bayes import naive_bayes
from Backprop import Backprop
from svm_bot import SVMBot
from ensemble import Ensemble


nb_acc = 0
bp_acc = 0
svm_acc = 0
ens_acc = 0
total = 0
for _ in xrange(0, 100):
    controller = Controller(naive_bayes())
    # print('\n\ntraining Naive Bayes\n\n')
    nb_acc += controller.train(.25)

    BPcontroller = Controller(Backprop())
    # print('\n\ntraining Backprop\n\n')
    bp_acc += BPcontroller.train(.25)

    SVMcontroller = Controller(SVMBot())
    # print('\n\ntraining SVM\n\n')
    svm_acc += SVMcontroller.train(.25)

    ensController = Controller(Ensemble())
    ens_acc += ensController.train(.25)

    total += 1
    print "Run " + str(total)

nb_acc /= total
bp_acc /= total
svm_acc /= total
ens_acc /= total

print "Naive Bayes Average Accuracy: " + str(nb_acc)
print "Backprop Average Accuracy: " + str(bp_acc)
print "SVM Average Accuracy: " + str(svm_acc)
print "Ensemble Average Accuracy: " + str(ens_acc)
print "total number of runs: " + str(total)
