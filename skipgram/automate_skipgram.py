import pickle
from gensim.models import word2vec, Word2Vec
import multiprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd
import pickle
import requests
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from scipy.spatial.distance import cosine
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from chembl_webresource_client.new_client import new_client
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, GRU, SpatialDropout1D, LSTM, Dropout
from keras.layers import BatchNormalization, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from imblearn.under_sampling import ClusterCentroids
import sys

## Train Large Dataset

dimensions = int(sys.argv[3].split("_")[-1])

w2v = word2vec.Word2Vec.load(sys.argv[3])
# w2v = word2vec.Word2Vec.load('../SPVec/model_300dim.pkl')

## Load ATC data

with open("./data/mol_sentences.pkl", "rb") as file:
    sentences = pickle.load(file)

atc = [sentence[1][0] for sentence in sentences]
sentences = [sentence[3] for sentence in sentences]

# dimensions = 50
# window_size = 5
# min_count = 5
# negative = 15
# iterations = 10
# 
# w2v = Word2Vec(sentences, size=dimensions, window=window_size, min_count=min_count, negative=negative, iter=20)

## Vectorisation

vectors = []

for sentence in sentences:
    vector = []
    for word in sentence:
        try:
            vector+=list(w2v.wv.word_vec(word))
        except:
            vector+=([0 for a in range(0, dimensions)])
    vectors.append(vector)
    
vectors = np.asarray(vectors)

sum_vectors = []

for vector in vectors:
    arr = np.asarray(vector)
    arr = arr.reshape((int(arr.shape[0]/dimensions), dimensions) )
    sum_vectors.append(arr.sum(axis=0)/arr.shape[0])

sum_vectors = np.asarray(sum_vectors)

le = LabelEncoder()
le.fit(atc)

atc = le.transform(atc)

def balance(X, y):    
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    return X_resampled, y_resampled

## Data Preparation

X = sum_vectors
y = atc

atc_counter = Counter(y)
num_classes = int(sys.argv[2])
top_atc = [a for a, _ in atc_counter.most_common(int(sys.argv[2]))]
if int(sys.argv[2])==3:
    top_atc = [2, 6, 9]

temp_x = []
temp_y = []

for i in range(y.shape[0]):
    if y[i] in top_atc:
        temp_x.append(X[i])
        temp_y.append(y[i])
        
X = np.asarray(temp_x)
y = np.asarray(temp_y)

balance_path = ["unbalanced", "balanced"]
if sys.argv[1]=="1":
    X, y = balance(X, y)
balance_path = balance_path[int(sys.argv[1])]

atc_labels = le.inverse_transform(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(random.random()*100))

## Helper Functions

def accuracy(y_true, y_pred, previous_accuracy, figname, filename, atc_labels=atc_labels):
    curr_accuracy = np.sum(np.equal(y_true, y_pred))/y_true.shape[0]
    if curr_accuracy>previous_accuracy:
        matrix = confusion_matrix(y_true, y_pred)
        with open(filename, "w") as file:
            file.truncate()
            file.write(f"Accuracy: {curr_accuracy}\n")
            file.write(classification_report(y_true, y_pred))
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, annot=True,cbar=False, xticklabels=atc_labels, yticklabels=atc_labels, cmap="YlGnBu")
        plt.savefig(figname)

    return max(curr_accuracy, previous_accuracy)

path = f"../results/auto/{sys.argv[3]}/{balance_path}/"
## KNN
def KNN(X_train, X_test, y_train, y_test, previous_accuracy):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    return accuracy(y_pred, y_test, previous_accuracy, f"{path}cm_{num_classes}_knn", f"{path}report_{num_classes}_knn", atc_labels)

## Random Forest

def RF(X_train, X_test, y_train, y_test, previous_accuracy):
    rf_model = RandomForestClassifier(n_estimators=100, 
                                   bootstrap = True,
                                   max_features = 'sqrt', criterion='entropy', random_state=56)
    # Fit on training data
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    return accuracy(y_pred, y_test, previous_accuracy, f"{path}cm_{num_classes}_rf", f"{path}report_{num_classes}_rf", atc_labels)

## Extra Trees

def XT(X_train, X_test, y_train, y_test, previous_accuracy):
    xt_model = ExtraTreesClassifier(n_estimators=100, warm_start=True,
                                max_features = 'sqrt', criterion='entropy', random_state=98)
    # Fit on training data
    xt_model.fit(X_train, y_train)

    y_pred = xt_model.predict(X_test)
    return accuracy(y_pred, y_test, previous_accuracy, f"{path}cm_{num_classes}_xt", f"{path}report_{num_classes}_xt", atc_labels)

prev_accuracy = 0
for a in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(random.random()*100))
    prev_accuracy = KNN(X_train, X_test, y_train, y_test, prev_accuracy)
    sys.stdout.write(f"\rKNN {a}/100: {prev_accuracy}")
sys.stdout.write(f"\rKNN {a}/100: {prev_accuracy}\n")


prev_accuracy = 0
for a in range(0, 20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(random.random()*100))
    prev_accuracy = RF(X_train, X_test, y_train, y_test, prev_accuracy)
    sys.stdout.write(f"\rRF {a}/20: {prev_accuracy}")
sys.stdout.write(f"\rRF {a}/20: {prev_accuracy}\n")

prev_accuracy = 0
for a in range(0, 50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(random.random()*100))
    prev_accuracy = XT(X_train, X_test, y_train, y_test, prev_accuracy)
    sys.stdout.write(f"\rRF {a}/50: {prev_accuracy}")
sys.stdout.write(f"\rRF {a}/50: {prev_accuracy}\n")
