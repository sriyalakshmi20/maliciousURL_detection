# Import Dependencies

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


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
    return allTokens


#read from a CSV file
data1 = pd.read_csv("final.csv",',',error_bad_lines=False)	# reading training file
data1 = pd.read_csv("feed.csv",',')                          # reading test file

#convert it into numpy array and shuffle the dataset
data1 = np.array(data1)
random.shuffle(data1)

#convert text data into numerical data for machine learning models
y1 = [d[1] for d in data1]
corpus = [d[0] for d in data1]
vectorizer = TfidfVectorizer(tokenizer=getTokens)
x1 = vectorizer.fit_transform(corpus)


#split the data set in to train and test
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)

# Train Machine Learning Models

#1 - Logistic Regression
print("Logistic Regression results\n")

LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(X_train, y_train)

logisticRegressionData_y_pred = LogisticRegressionModel.predict(X_test)
print(classification_report(y_test,logisticRegressionData_y_pred))
print(accuracy_score(y_test,logisticRegressionData_y_pred, normalize=True),"\n")

# Display the Confusion Matrix
plt.title('Confusion Matrix for Logistic Regression')
sns.heatmap(confusion_matrix(y_test,logisticRegressionData_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

#2 - SVM
print("Support Vector Model results\n")
svmModel = svm.SVC(kernel='linear')
svmModel.fit(X_train, y_train)

SVMdata_y_pred = svmModel.predict(X_test)
print(classification_report(y_test,SVMdata_y_pred))
print(accuracy_score(y_test,SVMdata_y_pred, normalize=True),"\n")

# Display the Confusion Matrix
plt.title('Confusion Matrix for SVM')
sns.heatmap(confusion_matrix(y_test,SVMdata_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

#3 - Random Forest
print("Random Forest results\n")

RandomForestModel = RandomForestClassifier(n_estimators=100)
RandomForestModel.fit(X_train, y_train)

randomForestData_y_pred = RandomForestModel.predict(X_test)
print(classification_report(y_test,randomForestData_y_pred))
print(accuracy_score(y_test,randomForestData_y_pred, normalize=True),"\n")

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

NeuralNetwork_y_pred = NeuralNetworkModel.predict(X_test)
print(classification_report(y_test,NeuralNetwork_y_pred))
print(accuracy_score(y_test,NeuralNetwork_y_pred, normalize=True),"\n")

# Display the Confusion Matrix
plt.title('Confusion Matrix for Neural Network')
sns.heatmap(confusion_matrix(y_test,NeuralNetwork_y_pred),annot=True,fmt="d")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# ROC Curves for the Four classifiers

# Logistic Regression
fpr_logistic,tpr_logistic,thresholds = roc_curve(y_test,logisticRegressionData_y_pred)

# SVM
fpr_svm,tpr_svm,thresholds = roc_curve(y_test,SVMdata_y_pred)

# RandomForest
fpr_random,tpr_random,thresholds = roc_curve(y_test, randomForestData_y_pred)

# Neural Network
fpr_neural,tpr_neural,thresholds = roc_curve(y_test,NeuralNetwork_y_pred)

# Plot the ROC Curve
plt.plot(fpr_logistic,tpr_logistic, label='Logistic Regression',lw=1)
plt.plot(fpr_svm, tpr_svm, label='SVM', lw=1, alpha=0.3)
plt.plot(fpr_random, tpr_random, 'g^',label='Random Forest', lw=1)
plt.plot(fpr_neural, tpr_neural, 'g-.',label='Neural Network',lw=1, alpha=0.3)
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.show()



