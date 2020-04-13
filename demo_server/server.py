import flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify
from gensim.models import word2vec
# from rdkit import Chem
# from rdkit.Chem import AllChem
import pandas as pd
import pickle
import sys
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, GRU, SpatialDropout1D, LSTM, Dropout
from keras.layers import BatchNormalization, MaxPool1D
from sklearn.ensemble import ExtraTreesClassifier

app = flask.Flask(__name__)
CORS(app)

class Network:
    def __init__(self, num_classes, oe):
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=dimensions, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.oe = oe
        self.num_classes = num_classes
        print(self.model.summary())
        
    def train(self, X, y, X_test=None, y_test=None, epochs=20):
        self.model.fit(X, y, epochs=epochs)
    
    def predict(self, X):
        y_pred = self.model.predict(X).argmax()
        y_pred_act = [0 for _ in range(self.num_classes)]
        y_pred_act[y_pred] = 1
        self.oe.inverse_transform(np.asarray([y_pred_act]))
        return self.oe.inverse_transform(np.asarray([y_pred_act]))


def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    # _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


@app.route('/getClassification', methods=['GET'])
@cross_origin()
def home():
	try:
		smile = request.args["smile"]
		try:
			# sentence = mol2alt_sentence(Chem.MolFromSmiles(smile), 1)
			vector = []
			for word in sentence:
				try:
					vector+=list(w2v.wv.word_vec(word))
				except:
					vector+=([0 for a in range(0, dimensions)])
			arr = np.asarray(vector)
			arr = arr.reshape((int(arr.shape[0]/dimensions), dimensions) )
			vector = arr.sum(axis=0)
			X = np.asarray([vector])

			result = {
			"KNN": le.inverse_transform(neigh.predict(X))[0],
			"RF": le.inverse_transform(rf_model.predict(X))[0],
            "ANN": le.inverse_transform(network.predict(X))[0],
			"XT": le.inverse_transform(xt_model.predict(X))[0]
			}

			return jsonify(result), 200

		except Exception as e:
			print(e)
			return jsonify([]), 400

	except Exception as e:
		print(e)
		return jsonify([-1, "Smile not provided"]), 400

def getFeatures(one, two):
    if one.lower()=='salmeterol' and two.lower()=='sulindac':
        index = 0
    elif one.lower()=='bupivacaine' and two.lower()=='verpamil':
        index = 1
    return pca.iloc[index, : 1000]


@app.route('/getAdrClassification', methods=['GET'])
def home2():
    try:
        features = request.args["features"]
        features = features.strip('][').split(',')
        features = getFeatures(str(features[0]), str(features[1]))
        try:
            result = {}
            result['AFIB'] = int(afib.predict([features])[0])
            result['AG'] = int(ag.predict([features])[0])
            result['AL'] = int(al.predict([features])[0])
            result['AJ'] = int(aj.predict([features])[0])
            result['AC'] = int(ac.predict([features])[0])

            result['AFIBtrue'] = int(label.loc[index, 'AFIB'])
            result['AGtrue'] = int(label.loc[index, 'Abnormal.Gait'])
            result['ALtrue'] = int(label.loc[index, 'Abnormal.LFTs'])
            result['AJtrue'] = int(label.loc[index, 'Aching.joints'])
            result['ACtrue'] = int(label.loc[index, 'Acidosis'])
            print(result)
            return jsonify(result), 200

        except Exception as e:
            print(e)
            return jsonify([]), 400

    except Exception as e:
        print(e)
        return jsonify([-1, "Input not provided"]), 400


print("Loading Models")
dimensions = 100
w2v = word2vec.Word2Vec.load('../skipgram/mine_100')
afib = pickle.load(open('../Models/RFforAFIB.sav', 'rb'))
ag = pickle.load(open('../Models/DTforAG.sav', 'rb'))
al = pickle.load(open('../Models/DTforAL.sav', 'rb'))
aj = pickle.load(open('../Models/DTforAJ.sav', 'rb'))
ac = pickle.load(open('../Models/DTforAC.sav', 'rb'))

print("Loading Data")
with open("../skipgram/data/mol_sentences.pkl", "rb") as file:
    sentences = pickle.load(file)
atc = [sentence[1][0] for sentence in sentences]
sentences = [sentence[3] for sentence in sentences]
label = pd.read_csv("../data/ADR_combined.csv")
pca = pd.read_csv("../data/selectedPCA.csv")
index=0

print("Vectorising")
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
    sum_vectors.append(arr.sum(axis=0))

sum_vectors = np.asarray(sum_vectors)
le = LabelEncoder()
le.fit(atc)

atc = le.transform(atc)
X = sum_vectors
y = atc

atc_labels = le.inverse_transform(np.unique(y))

print("Setting Up KNN")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)

print("Setting Up Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, 
                           bootstrap = True,
                           max_features = 'sqrt', criterion='entropy')
rf_model.fit(X, y)

xt_model = ExtraTreesClassifier(n_estimators=100, warm_start=True,
                                max_features = 'sqrt', criterion='entropy', random_state=98)
# Fit on training data
xt_model.fit(X, y)

print("Setting Up ANN")
oeAtc = OneHotEncoder(sparse=False)
oeAtc.fit(y.reshape(-1, 1))
y_onehot = oeAtc.transform(y.reshape(-1, 1))
network = Network(len(atc_labels), oeAtc)
network.train(X, y_onehot, epochs=10)


if __name__=="__main__":
		app.run()
