from sklearn import svm

class SequentialModel:
    def __init__(self):
        self.classifier = svm.SVC()