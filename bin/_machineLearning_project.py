# Import Dependencies

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


#custom tokenizer for URLs.
#first split - "/"
#second split - "-"
#third split - "."
#remove ".com" (also "http://", but we dont have "http://" in our dataset)
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
    #allTokens = (list(nltk.bigrams(allTokens)))
    return allTokens

#read from a CSV file
data1 = pd.read_csv("final4.csv",',')	# reading training file

#convert it into numpy array and shuffle the dataset
data1 = np.array(data1)
random.shuffle(data1)

#convert text data into numerical data for machine learning models
y1 = [d[1] for d in data1]
url = [d[0] for d in data1]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
x1 = vectorizer.fit_transform(url)

#split the data set in to train and test
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, train_size=0.7)

# Train Machine Learning Models

#1 - Logistic Regression
print("Logistic Regression results\n")

LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(X_train, y_train)
LogisticRegressionScores = cross_val_score(LogisticRegressionModel,x1,y1,cv=10)
print(LogisticRegressionScores)
logisticRegressionData_y_pred = LogisticRegressionModel.predict(X_test)
print(classification_report(y_test,logisticRegressionData_y_pred))
print("Accuracy: %0.2f (+/- %0.2f)\n" % (LogisticRegressionScores.mean(), LogisticRegressionScores.std() * 2))


# Display the Confusion Matrix
plt.title('Confusion Matrix for Logistic Regression')
sns.heatmap(confusion_matrix(y_test,logisticRegressionData_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


#2 - SVM
print("Support Vector Model results\n")
svmModel = svm.SVC(kernel='linear', probability=True)
svmModel.fit(X_train, y_train)
SVMScores = cross_val_score(svmModel,x1,y1,cv=10)
print(SVMScores)

SVMdata_y_pred = svmModel.predict(X_test)
print(classification_report(y_test,SVMdata_y_pred))
print("Accuracy: %0.2f (+/- %0.2f)\n" % (SVMScores.mean(), SVMScores.std() * 2))

# Display the Confusion Matrix
plt.title('Confusion Matrix for SVM')
sns.heatmap(confusion_matrix(y_test,SVMdata_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


#3 - Random Forest
print("Random Forest results\n")

RandomForestModel = RandomForestClassifier(n_estimators=50)
RandomForestModel.fit(X_train, y_train)
RandomForestScores = cross_val_score(RandomForestModel,x1,y1,cv=10)
print(RandomForestScores)
randomForestData_y_pred = RandomForestModel.predict(X_test)
print(classification_report(y_test,randomForestData_y_pred))
print("Accuracy: %0.2f (+/- %0.2f)\n" % (RandomForestScores.mean(), RandomForestScores.std() * 2))

# Display the Confusion Matrix
plt.title('Confusion Matrix for Random Forest')
sns.heatmap(confusion_matrix(y_test,randomForestData_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

#4 - Neural Networks
print("Neural Network results\n")
NeuralNetworkModel = MLPClassifier(activation='relu',hidden_layer_sizes=(15,10),max_iter=500)
NeuralNetworkModel.fit(X_train, y_train)
NeuralNetworkScores = cross_val_score(NeuralNetworkModel,x1,y1,cv=10)
print(NeuralNetworkScores)
NeuralNetwork_y_pred = NeuralNetworkModel.predict(X_test)
print(classification_report(y_test,NeuralNetwork_y_pred))
print("Accuracy: %0.2f (+/- %0.2f)\n" % (NeuralNetworkScores.mean(), NeuralNetworkScores.std() * 2))

# Display the Confusion Matrix
plt.title('Confusion Matrix for Neural Network')
sns.heatmap(confusion_matrix(y_test,NeuralNetwork_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

