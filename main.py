import pandas as pd
import numpy as np
import numbers
import decimal
import scipy.stats as ss
import matplotlib.pyplot as plt
from statistics import stdev
from statistics import mean
import time
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import WordNetLemmatizer


redditDataTrain = pd.read_csv("data/reddit_train.csv") #, sep="\n", header=None) 
redditDataTest = pd.read_csv("data/reddit_test.csv") # sep="\n", header=None)
allClasses = ["hockey", "nba", "leagueoflegends", "soccer", "funny", "movies", "anime", "Overwatch", "trees", "GlobalOffensive", "nfl", "AskReddit", "gameofthrones", "conspiracy", "worldnews", "wow", "europe", "canada", "Music", "baseball"]

# redditDataTrain.loc[redditDataTrain["subreddits"] == 'hockey', "subreddits"] = 0
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'nba', "subreddits"] = 1
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'leagueoflegends', "subreddits"] = 2
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'soccer', "subreddits"] = 3
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'funny', "subreddits"] = 4
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'movies', "subreddits"] = 5
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'anime', "subreddits"] = 6
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'Overwatch', "subreddits"] = 7
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'trees', "subreddits"] = 8
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'GlobalOffensive', "subreddits"] = 9
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'nfl', "subreddits"] = 10
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'AskReddit', "subreddits"] = 11
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'gameofthrones', "subreddits"] = 12
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'conspiracy', "subreddits"] = 13
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'worldnews', "subreddits"] = 14
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'wow', "subreddits"] = 15
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'europe', "subreddits"] = 16
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'canada', "subreddits"] = 17
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'Music', "subreddits"] = 18
# redditDataTrain.loc[redditDataTrain["subreddits"] == 'baseball', "subreddits"] = 19

# print(redditDataTrain)


commentsTrain = redditDataTrain.iloc[:,1]
subredditsTrain = redditDataTrain.iloc[:,-1]
commentsTest = redditDataTest.iloc[:,1]

# print(commentsTrain)
# print(subredditsTrain)
# print(commentsTest)

# stopWords = ["the", "and", "is", "upvote", "downvote", "I", "this", "like", "with", "at", "can", "my", "your", "karma", "they", "there", "of", "be", "an", "a", "to", "in", "you", "that", "are", "he", "she", "when", "if", "or", "have", "had", "so", "then", "them", "will", "their", "on", "from", "by"]

lemmatizer = WordNetLemmatizer()
tfidf = TfidfVectorizer(stop_words="english", smooth_idf=True, sublinear_tf=True, norm='l2', use_idf=True)
cv = CountVectorizer()
lr = LogisticRegression()
multiNB = MultinomialNB()
dtc = tree.DecisionTreeClassifier()
kf = StratifiedKFold(n_splits=5)


def getScoretWithModel(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)



# x_train, x_test, y_train, y_test = train_test_split(commentsTrain, subredditsTrain, test_size=0.2, random_state=4)
results = []
for train_index, test_index in kf.split(commentsTrain, subredditsTrain):
    x_train, x_test, y_train, y_test = commentsTrain[train_index], commentsTrain[test_index], subredditsTrain[train_index], subredditsTrain[test_index]
    redditDataTrainTF = tfidf.fit_transform(x_train)
    redditDataTestTF = tfidf.transform(x_test)
    # redditDataTrainTF.toarray()
    results.append(getScoretWithModel(lr, redditDataTrainTF, redditDataTestTF, y_train, y_test))
    print(results)
print(sum(results)/len(results))
# TODO: train with the entire data set then, not just one piece fo data
pred = lr.predict(redditDataTestTF)
# print(pred)
pred = pd.DataFrame(pred, columns=['Category']).to_csv("testResults.csv")

# redditDataTrainTF = tfidf.fit_transform(x_train)
# redditDataTestTF = tfidf.transform(x_test)
# redditDataTrainTF.toarray()
# redditDataTestFinal = tfidf.transform(commentsTest)



# print("dtc: " + getScoretWithModel(dtc, redditDataTrainTF, redditDataTestTF, y_train, y_test))
# print(getScoretWithModel(lr, redditDataTrainTF, redditDataTestTF, y_train, y_test))

# redditDataTrainCV = cv.fit_transform(x_train)
# redditDataTestCV = cv.transform(x_test)
# redditDataTrainCV.toarray()

# redditDataTestTF.toarray()


#########Decision Tree with TIDF###############
# dtc = tree.DecisionTreeClassifier()
# print(dtc.fit(redditDataTrainTF,y_train))
# pred = dtc.predict(redditDataTestTF)
# print(pred)

#########Decision Tree with CV###############
# dtc = tree.DecisionTreeClassifier()
# print(dtc.fit(redditDataTrainCV,y_train))
# pred = dtc.predict(redditDataTestCV)
# print(pred)


#########Logistic Regression##################
# lr = LogisticRegression()
# lr.fit(redditDataTrainTF, y_train)
# print(lr.score(redditDataTestTF, y_test))
# pred = lr.predict(redditDataTestTF)
# print(pred)
# pred = pd.DataFrame(pred, columns=['predictions']).to_csv("testResults.csv")


################multinomical naive bayes################
# multiNB = MultinomialNB()
# print(multiNB.fit(redditDataTrainTF,y_train))
# pred = multiNB.predict(redditDataTestTF)
# # print(pred)



# actual = np.array(y_test)

# counter = 0
# for i in range(len(pred)):
#     if pred[i] == actual[i]:
#         counter = counter+1

# print(counter/len(pred))


# print(redditDataTrainTF.shape)
# print(tfidf.inverse_transform(x[0]))
# print(commentsTrain.iloc[0])
# print(tfidf.get_feature_names())
# print(redditDataTrainTF)

# print(tfidf1.inverse_transform(featureMatrix[0]))

# cleanData = vectorizer.fit_transf


# wordKey = ["player" , "ball" , "ice", "joke", "critic", ]

# def NumberOfClass(x):
#     # allClasses = ["hockey", "nba", "leagueoflegends", "soccer", "funny", "movies", "anime", "Overwatch", "trees", "GlobalOffensive", "nfl", "AskReddit", "gameofthrones", "conspiracy", "worldnews", "wow", "europe", "canada", "Music", "baseball"]
#     classDistribution = []
#     counter = 0
#     for i in allClasses:
#         subreddit = i
#         for j in x.iloc[:,-1]:
#             if(j == subreddit):
#                 counter = counter + 1
#         classDistribution.append(counter)
#         counter = 0
#         # i += 1
#     return classDistribution

# def text_split(x):
#     processedComments = []
#     comments = x.iloc[:,1]
#     for data in comments:
#         data = data.lower()
#         data = data.split()
#         processedComments.append(data)
#     return processedComments


# def DataCleaning(x):
#     cleanData = text_split(x)

#     return cleanData

# print(NumberOfClass(redditDataTrain))
# print(DataCleaning(redditDataTrain))
# print(redditDataTrain.iloc[:,1])


if __name__ == "__main__":
    pass

