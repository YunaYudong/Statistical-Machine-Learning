# -*- coding: utf-8 -*-
# @Time    : 2019-09-10 11:44
# @Author  : Man Fu
# @File    : Token_SVM.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def submission_file(data):
    import csv
    with open('predictionSubTF.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['id', 'Predicted'])
        for count, predicted in enumerate(data):
            writer.writerow([count + 1, predicted])

trainSample = pd.read_csv('cleanTraining2.csv')
trainSample['tweet'].str.strip('\n')
for (index, row) in trainSample.iterrows():
    temp = row.loc['tweet']
    temp = temp[1:len(temp) - 1]
    temp = temp.replace(',', '')
    trainSample.at[index, 'tweet'] = row.loc['tweet'] = temp

vectorizer = TfidfVectorizer(sublinear_tf=True)
vectors = vectorizer.fit_transform(trainSample['tweet'])
#vectorizer = CountVectorizer()
#vectors = vectorizer.fit_transform(trainSample['tweet'])
# print(vectors.shape)
X_train, X_test, y_train, y_test = train_test_split(vectors,trainSample['user_id'], test_size=0.3)
# print "3-7分"

#svm = LinearSVC(penalty='l1',loss='squared_hinge',dual=False)
# print "Training....."
svm = LinearSVC()
svm.fit(X_train, y_train)

# print "Test"
predictions = svm.predict(X_test)
# print(accuracy_score(y_test, predictions))

testData = pd.read_csv('cleanTestData2.csv')
# print testData['tweet']
testData['tweet'].str.strip('\n')
for (index, row) in testData.iterrows():
    temp = row.loc['tweet']
    temp = temp[1:len(temp) - 1]
    temp = temp.replace(',', '')
    testData.at[index, 'tweet'] = row.loc['tweet'] = temp
#
# print testData['tweet']
unlabel_dtm = vectorizer.transform(testData['tweet'])
# print unlabel_dtm.shape
unlabel_pred = svm.predict(unlabel_dtm)
submission_file(unlabel_pred)
