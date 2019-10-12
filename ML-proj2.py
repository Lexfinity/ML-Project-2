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

redditDataTrain = pd.read_csv("reddit_train.csv") #, sep="\n", header=None) 
redditDataTest = pd.read_csv("reddit_test.csv") # sep="\n", header=None)
allClasses = ["hockey", "nba", "leagueoflegends", "soccer", "funny", "movies", "anime", "Overwatch", "trees", "GlobalOffensive", "nfl", "AskReddit", "gameofthrones", "conspiracy", "worldnews", "wow", "europe", "canada", "Music", "baseball"]

redditDataTrain.loc[redditDataTrain["subreddits"] == 'hockey', "subreddits"] = 0
redditDataTrain.loc[redditDataTrain["subreddits"] == 'nba', "subreddits"] = 1
redditDataTrain.loc[redditDataTrain["subreddits"] == 'leagueoflegends', "subreddits"] = 2
redditDataTrain.loc[redditDataTrain["subreddits"] == 'soccer', "subreddits"] = 3
redditDataTrain.loc[redditDataTrain["subreddits"] == 'funny', "subreddits"] = 4
redditDataTrain.loc[redditDataTrain["subreddits"] == 'movies', "subreddits"] = 5
redditDataTrain.loc[redditDataTrain["subreddits"] == 'anime', "subreddits"] = 6
redditDataTrain.loc[redditDataTrain["subreddits"] == 'Overwatch', "subreddits"] = 7
redditDataTrain.loc[redditDataTrain["subreddits"] == 'trees', "subreddits"] = 8
redditDataTrain.loc[redditDataTrain["subreddits"] == 'GlobalOffensive', "subreddits"] = 9
redditDataTrain.loc[redditDataTrain["subreddits"] == 'nfl', "subreddits"] = 10
redditDataTrain.loc[redditDataTrain["subreddits"] == 'AskReddit', "subreddits"] = 11
redditDataTrain.loc[redditDataTrain["subreddits"] == 'gameofthrones', "subreddits"] = 12
redditDataTrain.loc[redditDataTrain["subreddits"] == 'conspiracy', "subreddits"] = 13
redditDataTrain.loc[redditDataTrain["subreddits"] == 'worldnews', "subreddits"] = 14
redditDataTrain.loc[redditDataTrain["subreddits"] == 'wow', "subreddits"] = 15
redditDataTrain.loc[redditDataTrain["subreddits"] == 'europe', "subreddits"] = 16
redditDataTrain.loc[redditDataTrain["subreddits"] == 'canada', "subreddits"] = 17
redditDataTrain.loc[redditDataTrain["subreddits"] == 'Music', "subreddits"] = 18
redditDataTrain.loc[redditDataTrain["subreddits"] == 'baseball', "subreddits"] = 19

# print(redditDataTrain)


commentsTrain = redditDataTrain.iloc[:,1]
subredditsTrain = redditDataTrain.iloc[:,-1]
# commentsTest = redditDataTest.iloc[:,1]

# print(commentsTrain)
# print(subredditsTrain)
# print(commentsTest)


tfidf = TfidfVectorizer()
# tfidf1 = TfidfVectorizer()

x_train, x_test, y_train, y_test = train_test_split(commentsTrain, subredditsTrain, test_size=0.2, random_state=4)

redditDataTrainCV = tfidf.fit_transform(x_train)
redditDataTestCV = tfidf.transform(x_test)
redditDataTrainCV.toarray()
# redditDataTestCV.toarray()

multiNB = MultinomialNB()
print(multiNB.fit(redditDataTrainCV,y_train))
pred = multiNB.predict(redditDataTestCV)
# print(pred)
actual = np.array(y_test)

counter = 0
for i in range(len(pred)):
    if pred[i] == actual[i]:
        counter = counter+1

print(counter/len(pred))


# print(redditDataTrainCV.shape)
# print(tfidf.inverse_transform(x[0]))
# print(commentsTrain.iloc[0])
# print(tfidf.get_feature_names())
# print(redditDataTrainCV)

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