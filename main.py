from controller import Controller
from naive_bayes import naive_bayes

controller = Controller(naive_bayes())

controller.train(.1)