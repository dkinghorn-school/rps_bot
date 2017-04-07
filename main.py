from controller import Controller
from naive_bayes import naive_bayes
from Backprop import Backprop

controller = Controller(naive_bayes())

controller.train(.1)

BPcontroller = Controller(Backprop())

BPcontroller.train(.25)
