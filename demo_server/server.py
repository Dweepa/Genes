import flask
from flask import request, jsonify
from gensim.models import word2vec
from rdkit import Chem
from rdkit.Chem import AllChem
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

app = flask.Flask(__name__)

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
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

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
def home():
	try:
		smile = request.args["smile"]
		try:
			sentence = mol2alt_sentence(Chem.MolFromSmiles(smile), 1)
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
			"ANN": le.inverse_transform(network.predict(X))[0]
			}

			return jsonify(result), 200

		except Exception as e:
			print(e)
			return jsonify([]), 400

	except Exception as e:
		print(e)
		return jsonify([-1, "Smile not provided"]), 400


@app.route('/getAdrClassification', methods=['GET'])
def home2():
    try:
        features = request.args["features"]
        features = features.strip('][').split(', ')
        features = [float(e) for e in features]
        print(commonADRs)
        try:
            result = {}
            for adr in commonADRs:
                clf = adrModels[adr]
                clf.fit(pca, label[adr])
                features = pd.Series(features)
                result[adr] = int(clf.predict([features])[0])
                print(adr, result[adr])
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
adrModelMap = {'AFIB': 'LR',
 'Abnormal.Gait': 'LR',
 'Abnormal.LFTs': 'LR',
 'Aching.joints': 'SGD',
 'Acidosis': 'LR',
 'Acute.Respiratory.Distress.Syndrome': 'LR',
 'Adenopathy': 'LR',
 'Amnesia': 'LR',
 'Anorexia': 'SGD',
 'Anxiety': 'LR',
 'Arrhythmia': 'LR',
 'Aspartate.Aminotransferase.Increase': 'SGD',
 'Back.Ache': 'LR',
 'Bacterial.infection': 'LR',
 'Bilirubinaemia': 'LR',
 'Bladder.inflammation': 'LR',
 'Bleeding': 'SGD',
 'Blood.calcium.decreased': 'LR',
 'Candida.Infection': 'LR',
 'Cardiac.decompensation': 'LR',
 'Cardiac.ischemia': 'LR',
 'Cardiomyopathy': 'LR',
 'Chronic.Kidney.Disease': 'LR',
 'Cough': 'SGD',
 'Diabetes': 'LR',
 'Difficulty.breathing': 'LR',
 'Diplopia': 'LR',
 'Disorder.Lung': 'LR',
 'Drug.addiction': 'LR',
 'Drug.hypersensitivity': 'LR',
 'Dyspnea.exertional': 'LR',
 'Embolism.pulmonary': 'LR',
 'Excess.potassium': 'LR',
 'Extrasystoles.Ventricular': 'LR',
 'Extremity.pain': 'LR',
 'Fainting': 'LR',
 'Fatigue': 'SGD',
 'Feeling.unwell': 'SGD',
 'Hallucination': 'LR',
 'Head.ache': 'SGD',
 'Hepatic.failure': 'LR',
 'High.blood.pressure': 'SGD',
 'Hypotension.Orthostatic': 'LR',
 'Hypoventilation': 'DT',
 'Incontinence': 'LR',
 'Infection.Upper.Respiratory': 'LR',
 'Infection.Urinary.Tract': 'SGD',
 'Intervertebral.Disc.Herniation': 'LR',
 'Mod': 'LR',
 'Nephrolithiasis': 'LR',
 'Neutropenia': 'LR',
 'Osteoarthritis': 'LR',
 'Pain': 'LR',
 'Paraesthesia': 'LR',
 'Pleural.Effusion': 'SGD',
 'Scar': 'LR',
 'Sinusitis': 'DT',
 'abdominal.distension': 'LR',
 'abdominal.pain.upper': 'LR',
 'abdominal.pain': 'SGD',
 'abnormal.movements': 'LR',
 'abscess': 'LR',
 'aching.muscles': 'LR',
 'acid.reflux': 'LR',
 'acute.brain.syndrome': 'LR',
 'acute.kidney.failure': 'SGD',
 'adenocarcinoma': 'LR',
 'adynamic.ileus': 'LR',
 'agitated': 'LR',
 'allergic.dermatitis': 'LR',
 'allergies': 'LR',
 'alopecia': 'LR',
 'anaemia': 'LR',
 'angina': 'LR',
 'apoplexy': 'LR',
 'aptyalism': 'LR',
 'arterial.pressure.NOS.decreased': 'SGD',
 'arterial.pressure.NOS.increased': 'LR',
 'arteriosclerosis': 'LR',
 'arteriosclerotic.heart.disease': 'LR',
 'arthritis': 'LR',
 'arthropathy': 'LR',
 'ascites': 'LR',
 'aseptic.necrosis.bone': 'DT',
 'aspiration.pneumonia': 'LR',
 'asthenia': 'SGD',
 'asthma': 'LR',
 'asystole': 'LR',
 'atelectasis': 'LR',
 'balance.disorder': 'LR',
 'bladder.retention': 'LR',
 'bleb': 'LR',
 'blood.in.urine': 'LR',
 'blood.sodium.decreased': 'SGD',
 'blurred.vision': 'LR',
 'body.temperature.increased': 'LR',
 'bone.marrow.failure': 'LR',
 'bone.pain': 'LR',
 'bradycardia': 'LR',
 'bronchitis': 'LR',
 'bruise': 'LR',
 'bruxism': 'LR',
 'bulging': 'LR',
 'burning.sensation': 'LR',
 'bursitis': 'LR',
 'candidiasis.of.mouth': 'LR',
 'cardiac.disease': 'LR',
 'cardiac.enlargement': 'LR',
 'cardiac.failure': 'LR',
 'cardiac.murmur': 'LR',
 'cardiovascular.collapse': 'LR',
 'carpal.tunnel': 'LR',
 'cataract': 'LR',
 'cellulitis': 'LR',
 'cerebral.infarct': 'LR',
 'cervicalgia': 'LR',
 'chest.pain': 'LR',
 'chill': 'LR',
 'cholecystitis': 'LR',
 'cholelithiasis': 'LR',
 'chronic.obstructive.airway.disease': 'LR',
 'cognitive.disorder': 'LR',
 'coma': 'LR',
 'confusion': 'SGD',
 'conjunctivitis': 'LR',
 'constipated': 'SGD',
 'convulsion': 'SGD',
 'coughing.blood': 'LR',
 'deep.vein.thromboses': 'LR',
 'deglutition.disorder': 'LR',
 'dehydration': 'SGD',
 'diarrhea': 'SGD',
 'difficulty.in.walking': 'LR',
 'disease.of.liver': 'LR',
 'disorder.Renal': 'LR',
 'diverticulitis': 'LR',
 'dizziness': 'SGD',
 'drowsiness': 'LR',
 'drug.toxicity.NOS': 'LR',
 'drug.withdrawal': 'LR',
 'dysarthria': 'LR',
 'dyspepsia': 'LR',
 'dysuria': 'LR',
 'edema.extremities': 'LR',
 'edema': 'SGD',
 'elevated.cholesterol': 'LR',
 'emesis': 'SGD',
 'encephalopathy': 'LR',
 'enlarged.liver': 'SGD',
 'epistaxis': 'LR',
 'eruption': 'SGD',
 'erythema': 'LR',
 'esophagitis': 'LR',
 'facial.flushing': 'LR',
 'fatty.liver': 'LR',
 'fibromyalgia': 'LR',
 'flatulence': 'LR',
 'flu': 'LR',
 'fungal.disease': 'LR',
 'gall.bladder': 'LR',
 'gastric.inflammation': 'LR',
 'gastroenteritis': 'LR',
 'gastrointestinal.bleed': 'LR',
 'haematochezia': 'LR',
 'haemorrhage.rectum': 'LR',
 'haemorrhoids': 'LR',
 'head.injury': 'DT',
 'heart.attack': 'LR',
 'heart.rate.increased': 'SGD',
 'hemiparesis': 'LR',
 'hepatitis': 'LR',
 'hernia.hiatal': 'LR',
 'herpes.zoster': 'LR',
 'hive': 'LR',
 'hot.flash': 'DT',
 'hyperglycaemia': 'SGD',
 'hyperhidrosis': 'LR',
 'hyperlipaemia': 'LR',
 'hypoglycaemia': 'LR',
 'hypokalaemia': 'LR',
 'hypothyroid': 'LR',
 'hypoxia': 'LR',
 'icterus': 'LR',
 'increased.platelet.count': 'LR',
 'increased.white.blood.cell.count': 'LR',
 'infection': 'LR',
 'insomnia': 'SGD',
 'itch': 'SGD',
 'joint.swelling': 'LR',
 'kidney.failure': 'SGD',
 'leucocytosis': 'DT',
 'leucopenia': 'LR',
 'lipoma': 'LR',
 'lobar.pneumonia': 'LR',
 'loss.of.consciousness': 'LR',
 'loss.of.weight': 'SGD',
 'lung.edema': 'LR',
 'lung.infiltration': 'LR',
 'malnourished': 'DT',
 'metabolic.acidosis': 'LR',
 'migraine': 'LR',
 'mucositis.oral': 'LR',
 'muscle.spasm': 'LR',
 'muscle.weakness': 'LR',
 'musculoskeletal.pain': 'LR',
 'narcolepsy': 'LR',
 'nasal.congestion': 'LR',
 'nausea': 'SGD',
 'nervous.tension': 'LR',
 'neumonia': 'LR',
 'neuropathy.peripheral': 'LR',
 'neuropathy': 'LR',
 'night.sweat': 'LR',
 'obesity': 'LR',
 'osteomyelitis': 'LR',
 'osteoporosis': 'LR',
 'palpitation': 'LR',
 'pancreatitis': 'LR',
 'panic.attack': 'LR',
 'pericardial.effusion': 'LR',
 'phlebothrombosis': 'LR',
 'pleural.pain': 'LR',
 'pulmonary.hypertension': 'LR',
 'pyoderma': 'LR',
 'respiratory.failure': 'LR',
 'rhabdomyolysis': 'LR',
 'rhinorrhea': 'LR',
 'road.traffic.accident': 'LR',
 'sepsis': 'LR',
 'septic.shock': 'LR',
 'sinus.tachycardia': 'LR',
 'skin.lesion': 'LR',
 'sleep.apnea': 'LR',
 'sleep.disorder': 'LR',
 'spinning.sensation': 'LR',
 'still.birth': 'LR',
 'tachycardia.ventricular': 'LR',
 'thrombocytopenia': 'LR',
 'tinnitus': 'LR',
 'transient.ischaemic.attack': 'LR',
 'tremor': 'LR',
 'weight.gain': 'LR',
 'wheeze': 'LR'}

adrs = list(adrModelMap.keys())
adrModels = {}
for adr in adrs:
    if(adrModelMap[adr] == 'DT'):
        adrModels[adr] = DecisionTreeClassifier(max_depth=10, random_state=101, min_samples_leaf=15)
    elif(adrModelMap[adr] == 'LR'):
        adrModels[adr] = LogisticRegression(random_state=0)
    else:
        adrModels[adr] = SGDClassifier(max_iter=1500)

print("Loading Data")
with open("../skipgram/data/mol_sentences.pkl", "rb") as file:
    sentences = pickle.load(file)
atc = [sentence[1][0] for sentence in sentences]
sentences = [sentence[3] for sentence in sentences]
label = pd.read_csv("../data/ADR_combined.csv")
pca = pd.read_csv("../data/pca300.csv")
pca.drop("Unnamed: 0", axis=1, inplace=True)
label.drop("Unnamed: 0", axis=1, inplace=True)
commonADRs = label.sum()
commonADRs.sort_values(ascending=False, inplace=True)
commonADRs = commonADRs[:10]
commonADRs = list(commonADRs.index)

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

print("Setting Up ANN")
oeAtc = OneHotEncoder(sparse=False)
oeAtc.fit(y.reshape(-1, 1))
y_onehot = oeAtc.transform(y.reshape(-1, 1))
network = Network(len(atc_labels), oeAtc)
network.train(X, y_onehot, epochs=10)


if __name__=="__main__":
		app.run()
