# -*- coding: utf-8 -*-
# @Time    : 2019-09-09 00:54
# @Author  : Man Fu
# @File    : svm_w2v.py
# @Software: PyCharm
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC
from sklearn import svm
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

word2vec_path = "GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)



def submission_file(data):
    import csv
    with open('prediction.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['id', 'Predicted'])
        for count, predicted in enumerate(data):
            writer.writerow([count + 1, predicted])



trainSample = pd.read_csv('cleanTraining.csv')
trainSample['tweet'].str.strip('\n')
for (index, row) in trainSample.iterrows():
    temp = row.loc['tweet']
    temp = temp[1:len(temp) - 1]
    temp = temp.replace(',', '')
    trainSample.at[index, 'tweet'] = row.loc['tweet'] = temp
tokenizer = RegexpTokenizer(r'\w+')
trainSample["tokens"] = trainSample["tweet"].apply(tokenizer.tokenize)
embeddings = get_word2vec_embeddings(word2vec, trainSample)
embeddings = np.array(embeddings)
y = trainSample['user_id'].tolist()
# print trainSample['tweet']
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, y, test_size=0.3, random_state=40)

print("length of X_train: %d, X_text: %d, y_train: %d, y_test: %d" % (len(X_train_word2vec), len(X_test_word2vec),
                                                                      len(y_train_word2vec), len(y_test_word2vec)))
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(X_train_word2vec)
xtrain_svd = svd.transform(X_train_word2vec)
xvalid_svd = svd.transform(X_test_word2vec)

#对从SVD获得的数据进行缩放
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)


clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(xtrain_svd_scl,y_train_word2vec)
score_linear = clf_linear.score(xvalid_svd_scl,y_test_word2vec)

testData = pd.read_csv('cleanTestData.csv')
testData['tweet'].str.strip('\n')
for (index, row) in trainSample.iterrows():
    temp = row.loc['tweet']
    temp = temp[1:len(temp) - 1]
    temp = temp.replace(',', '')
    testData.at[index, 'tweet'] = row.loc['tweet'] = temp
tokenizer = RegexpTokenizer(r'\w+')
testData["tokens"] = testData["tweet"].apply(tokenizer.tokenize)
embeddingsTest = get_word2vec_embeddings(word2vec, testData)
embeddingsTest = np.array(embeddingsTest)
test_svd = scl.transform(embeddingsTest)
unlabel_pred = svm.predict(test_svd)
submission_file(unlabel_pred)

# clf=svm.SVC()
# clf = SVC(C=1.0, probability=True) # since we need probabilities
# clf.fit(xtrain_svd_scl, y_train_word2vec)
# predictions = clf.predict_proba(xvalid_svd_scl)

print (score_linear)




# clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
#                          multi_class='multinomial', random_state=40)
# clf_w2v.fit(X_train_word2vec, y_train_word2vec)
# y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)
