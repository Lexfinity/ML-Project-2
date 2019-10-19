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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import mutual_info_classif
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

####Importing the datasets
redditDataTrain = pd.read_csv("data/reddit_train.csv")
redditDataTest = pd.read_csv("data/reddit_test.csv") 


#Separate the comments and subreddits into 
commentsTrain = redditDataTrain.iloc[:,1]
subredditsTrain = redditDataTrain.iloc[:,-1]
commentsTest = redditDataTest.iloc[:,1]


#Custom Stopwords list
stopWords= [ 'never', 'yours', 'meanwhile', 'side', 'herself', 'sixty', 'some', 'first', 'someone', 'had', 'via', 'if', 'front', 'latterly', 'fifteen', 'either', 'toward', 'next', 'during', 'take', 'since', 'inc', 'on', 'was', 'eg', 'something', 'there', 'none', 'cant', 'still', 'well', 'and', 'has', 'its', 'anything', 'anywhere', 'whereafter', 'towards', 're', 'herein', 'where', 'again', 'only', 'nobody', 'how', 'sincere', 'among', 'etc', 'whereas', 'five', 'while', 'up', 'so', 'whither', 'down', 'back', 'cry', 'his', 'ever', 'against', 'former', 'he', 'two', 'own', 'this', 'sometimes', 'such', 'beyond', 'found', 'himself', 'others', 'by', 'whoever', 'that', 'in', 'whom', 'whereby', 'neither', 'go', 'full', 'whole', 'might', 'ltd', 'should', 'bill', 'anyhow', 'most', 'a', 'we', 'all', 'thru', 'for', 'ten', 'show', 'could', 'put', 'move', 'us', 'part', 'besides', 'about', 'thereby', 'because', 'between', 'elsewhere', 'she',
 'indeed', 'twelve', 'ourselves', 'why', 'above', 'give', 'now', 'perhaps', 'yourselves', 'become', 'somehow', 'at', 'each', 'otherwise', 'as', 'your', 'our', 'fill', 'becoming', 'they', 'am', 'eleven', 'several', 'get', 'thence', 'per', 'here', 'nowhere', 'who', 'are', 'anyway', 'not', 'much', 'mill', 'empty', 'when', 'thereupon', 'off', 'once', 'seemed', 'be', 'may', 'hereafter', 'also', 'which', 'third', 'whereupon', 'along',
 'hasnt', 'out', 'do', 'further', 'however', 'therein', 'amount', 'somewhere', 'together', 'anyone', 'nor', 'always', 'both', 'will', 'least', 'nine', 'nevertheless', 'thereafter', 'find', 'last', 'around', 'sometime', 'you', 'moreover', 'amongst', 'myself', 'within', 'interest', 'with',
 'describe', 'co', 'wherever', 'already', 'cannot', 'mine', 'me', 'call', 'whose', 'became', 'seems', 'those', 'beforehand', 'although', 'ie', 'seeming', 'hers', 'fire', 'nothing', 'due', 'throughout', 'wherein', 'him', 'thus', 'after', 'twenty', 'bottom', 'it', 'few', 'into', 'over', 'except', 'top', 'before', 'amoungst', 'made', 'namely', 'eight', 'upon', 'less', 'noone', 'would', 'hereby', 'then', 'any', 'behind', 'an', 'every', 'many', 'de', 'same', 'con', 'hundred', 'one', 'across', 'other', 'beside', 'through', 'to', 'her', 'must', 'been', 'the', 'though', 'what',
 'ours', 'but', 'whether', 'done', 'alone', 'i', 'hereupon', 'whence', 'afterwards', 'whenever', 'whatever', 'four', 'from', 'being', 'system', 'therefore', 'onto', 'enough', 'itself', 'very', 'almost', 'serious', 'couldnt', 'have', 'latter', 'of', 'my', 'else', 'rather', 'is', 'than', 'themselves', 'often', 'thick', 'can', 'name', 'yourself', 'everywhere', 'until', 'keep', 'without', 'even', 'another', 'were', 'more', 'detail',
 'everything', 'or', 'forty', 'please', 'under', 'their', 'fifty', 'seem', 'everyone', 'yet', 'hence', 'three', 'them', 'these', 'un', 'thin', 'becomes', 'see', 'no', 'below', 'too', 'six', 'mostly', 'formerly' 'abov', 'afterward', 'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 'cri', 'describ', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'formerli', 'forti', 'ha', 'henc', 'hereaft', 'herebi',
 'hi', 'howev', 'hundr', 'inde', 'latterli', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'seriou', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv', 'twenti', 'veri', 'wa', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv', 'ha', 'le', 'u', 'wa', 
'upvote', 'downvote', 'karma', 'lmao', 'tldr', '*', 'iirc', 'lol', 'nice', 'edit', 'ha', 'gold', 'silver', 'platinum' ]



#####Text Preprocessing techniques, using tfidf and countvectorizer
tfidf = TfidfVectorizer(stop_words=stopWords, smooth_idf=False, sublinear_tf=True, use_idf=True, strip_accents='unicode', max_features=65500) #max_df=0.1) #max_features=65500)  
cv = CountVectorizer(stop_words="english", strip_accents='unicode', analyzer='word',)

##### SKlearn Models created and used for training 
lr = LogisticRegression(solver='sag')
multiNB = MultinomialNB(alpha=0.3)
dtc = tree.DecisionTreeClassifier()
kf = StratifiedKFold(n_splits=5)



###Helper method created to specify model being used, and to fit using said model and return a test score
def getScoretWithModel(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


###################Code block used to train using all of the train comments and to predict on all the test comments
# x_train = tfidf.fit_transform(commentsTrain)
# x_test = tfidf.transform(commentsTest)
# multiNB.fit(x_train, subredditsTrain)
# pred = multiNB.predict(x_test)
# # # print(pred)
# pred = pd.DataFrame(pred, columns=['Category']).to_csv("testResults_new3.csv")


###################Train Test split 80/20 Using multinomial NB, fits model and runs prediction returning score##################
#############(or any model can be inserted as the first parameter in the getscorewithmodel method)###################
x_train, x_test, y_train, y_test = train_test_split(commentsTrain, subredditsTrain, test_size=0.2, random_state=4)
redditDataTrainTF = tfidf.fit_transform(x_train)
redditDataTestTF = tfidf.transform(x_test)
print(getScoretWithModel(multiNB, redditDataTrainTF, redditDataTestTF, y_train, y_test))


###################Train using k-fold cross validation, Using multinomial NB, fits model and runs prediction returning score##################
#############(or any model can be inserted as the first parameter in the getscorewithmodel method)###################
# results = []
# for train_index, test_index in kf.split(commentsTrain, subredditsTrain):
#     x_train1, x_validate, y_train1, y_validate = commentsTrain[train_index], commentsTrain[test_index], subredditsTrain[train_index], subredditsTrain[test_index]
#     redditDataTrainTF = tfidf.fit_transform(x_train1)
#     redditDataTestTF = tfidf.transform(x_validate)
    # redditDataTrainTF.toarray()
    # print(redditDataTrainTF.shape)
    # feature_scores = mutual_info_classif(redditDataTrainTF, y_train1)
    # print(feature_scores)
    
#     results.append(getScoretWithModel(multiNB, redditDataTrainTF, redditDataTestTF, y_train1, y_validate))
#     print(results)
# print(sum(results)/len(results))


if __name__ == "__main__":
    pass

