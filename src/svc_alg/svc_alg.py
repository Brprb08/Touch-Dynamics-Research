import numpy as np
from sklearn import svm
from svc_preprocess import preprocess


class svc_model:
    def __init__(self, data: preprocess):
        self.X_train: np.array = data.X_train
        self.X_test: np.array = data.X_test
        self.y_train: np.array = data.y_train
        self.y_test: np.array = data.y_test

    def train(self):
        clf = svm.SVC(kernel='linear', C=1).fit(self.X_train, self.y_train)
        clf.score(self.X_test, self.y_test)

        
if __name__ == "__main__":
    #Directory for csv files
    processor: preprocess = preprocess()
    processor.openData("pubg_firstTwo_touch.csv")
    session: svc_model = svc_model(processor)
    session.train()
    
