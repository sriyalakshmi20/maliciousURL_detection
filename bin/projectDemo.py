import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Read in the Demo CSV file
rawDataset = pd.read_csv('final4.csv')

# Import custom tokenizer function
def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    if 'org' in allTokens:
        allTokens.remove('org')
    if 'www' in allTokens:
        allTokens.remove('www')
    if 'net' in allTokens:
        allTokens.remove('net')
    return allTokens

# Get x and y points
x = rawDataset["url"]
y = rawDataset["isMalicious"]

# Transform X values
vectorizer = TfidfVectorizer(tokenizer=getTokens)
x = vectorizer.fit_transform(x)

# Test/training split function
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)

# Create Prediction dataset with 10 urls
# True set : [0 1 1 0 1 0 0 1 1 0]
x_true = [0,1,1,0,1,0,0,1,1,0]
x_predict = ['onlinecollege.org', 'jaybirdsport.com',
             'olcpr.org', 'scrippscollege.edu', 'indiaforuminc.info',
             'costumecraze.com', 'courageonthemountain', 'https://isorina.com/att.com.support/login2.php,1,46'
             'http://glowshow.net/bankofamerica/security.php', 'ayurvedapsychologie.at']
x_predict = vectorizer.transform(x_predict)


''' Logistic Regression '''
LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(x_train, y_train)
LogisticRegressionScores = cross_val_score(LogisticRegressionModel,x,y,cv=10)
print('10 Fold Cross Validation results are: ',LogisticRegressionScores)
print('Prediction Matrix is: ',LogisticRegressionModel.predict(x_predict))
print('Truth Matrix is: ',x_true)
''' Logistic Regression '''






