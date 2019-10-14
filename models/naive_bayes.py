import pandas as pd
import numpy as np
from collections import defaultdict
from functools import partial


class MultiNomialNB:
    def __init__(self, smoothing_factor=0.01):
        self.thetak = defaultdict(float)
        # self.thetajk = defaultdict(partial(np.ndarray, 0))
        self.thetajk= []
        self.classes = []
        self.smoothing_factor = smoothing_factor
        print("smoothing factor", self.smoothing_factor)
        
    def fit(self, X, y):
        self.classes = y.unique()
        self.thetajk = np.arange(len(self.classes)*X.shape[1], dtype=float).reshape(len(self.classes),X.shape[1])

        for i,k in enumerate(self.classes):
            self.thetak[k] = (y==k).sum()/float(y.shape[0])
            indices = np.where(y==k)[0]
            filteredX = X[indices]
            self.thetajk[i] = (filteredX.sum(axis=0)+(1*self.smoothing_factor))/float(filteredX.shape[0]+(2*self.smoothing_factor))
            self.thetajk[i] = self.thetajk[i]
            
    def predict(self, X):
        left = X.dot(np.log(self.thetajk).T)
        right = (1-X.toarray()).dot(np.log(1-self.thetajk).T)
        temp_sum = np.add(left, right)
        max_num = np.argmax(temp_sum,axis=1)
        return self.classes[max_num]

    def score(self, X, y):
        y_pred = self.predict(X)        
        score = (y_pred==y).sum()/float(y.shape[0])
        return score

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer

    redditDataTrain = pd.read_csv("../data/reddit_train.csv") #, sep="\n", header=None) 
    redditDataTest = pd.read_csv("../data/reddit_test.csv") # sep="\n", header=None)

    redditDataTrain = redditDataTrain

    commentsTrain = redditDataTrain.iloc[:,1]
    subredditsTrain = redditDataTrain.iloc[:,-1]
    commentsTest = redditDataTest.iloc[:,1]

    x_train, x_test, y_train, y_test = train_test_split(commentsTrain, subredditsTrain, test_size=0.2, random_state=4)

    cv = CountVectorizer(binary=True, stop_words='english')
    x_train_v = cv.fit_transform(x_train)
    x_test_v = cv.transform(x_test)

    cmultiNB = MultiNomialNB()
    cmultiNB.fit(x_train_v, y_train)

    print(cmultiNB.score(x_test_v, y_test))


