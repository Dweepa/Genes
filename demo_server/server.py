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
        try:
            result = {}
            for adr in adrs:
                clf = adrModels[adr]
                features = pd.Series(features)
                result[adr] = int(clf.predict([features])[0])
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
adrModelMap = {'AFIB': '../Models/AFIB-LR-1.0.sav',
 'Abnormal.Gait': '../Models/Abnormal.Gait-LR-1.0.sav',
 'Abnormal.LFTs': '../Models/Abnormal.LFTs-NB-1.0.sav',
 'Aching.joints': '../Models/Aching.joints-LR-1.0.sav',
 'Acidosis': '../Models/Acidosis-NB-1.0.sav',
 'Acute.Respiratory.Distress.Syndrome': '../Models/Acute.Respiratory.Distress.Syndrome-NB-2.0.sav',
 'Adenopathy': '../Models/Adenopathy-NB-1.0.sav',
 'Amnesia': '../Models/Amnesia-LR-1.0.sav',
 'Anorexia': '../Models/Anorexia-NB-1.0.sav',
 'Anxiety': '../Models/Anxiety-NB-1.0.sav',
 'Arrhythmia': '../Models/Arrhythmia-NB-5.0.sav',
 'Aspartate.Aminotransferase.Increase': '../Models/Aspartate.Aminotransferase.Increase-NB-5.0.sav',
 'Back.Ache': '../Models/Back.Ache-LR-1.0.sav',
 'Bacterial.infection': '../Models/Bacterial.infection-LR-2.0.sav',
 'Bilirubinaemia': '../Models/Bilirubinaemia-NB-2.0.sav',
 'Bladder.inflammation': '../Models/Bladder.inflammation-LR-1.0.sav',
 'Bleeding': '../Models/Bleeding-NB-1.0.sav',
 'Blood.calcium.decreased': '../Models/Blood.calcium.decreased-LR-1.0.sav',
 'Candida.Infection': '../Models/Candida.Infection-NB-1.0.sav',
 'Cardiac.decompensation': '../Models/Cardiac.decompensation-NB-1.0.sav',
 'Cardiac.ischemia': '../Models/Cardiac.ischemia-LR-2.0.sav',
 'Cardiomyopathy': '../Models/Cardiomyopathy-NB-2.0.sav',
 'Chronic.Kidney.Disease': '../Models/Chronic.Kidney.Disease-LR-5.0.sav',
 'Cough': '../Models/Cough-LR-1.0.sav',
 'Diabetes': '../Models/Diabetes-NB-1.0.sav',
 'Difficulty.breathing': '../Models/Difficulty.breathing-LR-1.0.sav',
 'Diplopia': '../Models/Diplopia-NB-1.0.sav',
 'Disorder.Lung': '../Models/Disorder.Lung-LR-3.0.sav',
 'Drug.addiction': '../Models/Drug.addiction-LR-5.0.sav',
 'Drug.hypersensitivity': '../Models/Drug.hypersensitivity-LR-5.0.sav',
 'Dyspnea.exertional': '../Models/Dyspnea.exertional-NB-5.0.sav',
 'Embolism.pulmonary': '../Models/Embolism.pulmonary-NB-1.0.sav',
 'Excess.potassium': '../Models/Excess.potassium-NB-1.0.sav',
 'Extrasystoles.Ventricular': '../Models/Extrasystoles.Ventricular-LR-5.0.sav',
 'Extremity.pain': '../Models/Extremity.pain-NB-5.0.sav',
 'Fainting': '../Models/Fainting-NB-1.0.sav',
 'Fatigue': '../Models/Fatigue-LR-1.0.sav',
 'Feeling.unwell': '../Models/Feeling.unwell-SGD-1.0.sav',
 'Hallucination': '../Models/Hallucination-LR-4.0.sav',
 'Head.ache': '../Models/Head.ache-NB-4.0.sav',
 'Hepatic.failure': '../Models/Hepatic.failure-LR-2.0.sav',
 'High.blood.pressure': '../Models/High.blood.pressure-NB-2.0.sav',
 'Hypotension.Orthostatic': '../Models/Hypotension.Orthostatic-LR-1.0.sav',
 'Hypoventilation': '../Models/Hypoventilation-NB-1.0.sav',
 'Incontinence': '../Models/Incontinence-LR-1.0.sav',
 'Infection.Upper.Respiratory': '../Models/Infection.Upper.Respiratory-NB-1.0.sav',
 'Infection.Urinary.Tract': '../Models/Infection.Urinary.Tract-NB-1.0.sav',
 'Intervertebral.Disc.Herniation': '../Models/Intervertebral.Disc.Herniation-LR-1.0.sav',
 'Mod': '../Models/Mod-LR-2.0.sav',
 'Nephrolithiasis': '../Models/Nephrolithiasis-LR-5.0.sav',
 'Neutropenia': '../Models/Neutropenia-LR-2.0.sav',
 'Osteoarthritis': '../Models/Osteoarthritis-LR-1.0.sav',
 'Pain': '../Models/Pain-NB-1.0.sav',
 'Paraesthesia': '../Models/Paraesthesia-LR-1.0.sav',
 'Pleural.Effusion': '../Models/Pleural.Effusion-SGD-1.0.sav',
 'Scar': '../Models/Scar-LR-5.0.sav',
 'Sinusitis': '../Models/Sinusitis-NB-5.0.sav',
 'abdominal.distension': '../Models/abdominal.distension-NB-1.0.sav',
 'abdominal.pain.upper': '../Models/abdominal.pain.upper-NB-3.0.sav',
 'abdominal.pain': '../Models/abdominal.pain-LR-5.0.sav',
 'abnormal.movements': '../Models/abnormal.movements-LR-5.0.sav',
 'abscess': '../Models/abscess-LR-2.0.sav',
 'aching.muscles': '../Models/aching.muscles-LR-1.0.sav',
 'acid.reflux': '../Models/acid.reflux-NB-1.0.sav',
 'acute.brain.syndrome': '../Models/acute.brain.syndrome-NB-1.0.sav',
 'acute.kidney.failure': '../Models/acute.kidney.failure-LR-1.0.sav',
 'adenocarcinoma': '../Models/adenocarcinoma-NB-1.0.sav',
 'adynamic.ileus': '../Models/adynamic.ileus-SGD-5.0.sav',
 'agitated': '../Models/agitated-NB-5.0.sav',
 'allergic.dermatitis': '../Models/allergic.dermatitis-LR-1.0.sav',
 'allergies': '../Models/allergies-NB-1.0.sav',
 'alopecia': '../Models/alopecia-LR-1.0.sav',
 'anaemia': '../Models/anaemia-NB-1.0.sav',
 'angina': '../Models/angina-NB-1.0.sav',
 'apoplexy': '../Models/apoplexy-LR-3.0.sav',
 'aptyalism': '../Models/aptyalism-LR-3.0.sav',
 'arterial.pressure.NOS.decreased': '../Models/arterial.pressure.NOS.decreased-LR-1.0.sav',
 'arterial.pressure.NOS.increased': '../Models/arterial.pressure.NOS.increased-LR-4.0.sav',
 'arteriosclerosis': '../Models/arteriosclerosis-LR-2.0.sav',
 'arteriosclerotic.heart.disease': '../Models/arteriosclerotic.heart.disease-LR-1.0.sav',
 'arthritis': '../Models/arthritis-NB-1.0.sav',
 'arthropathy': '../Models/arthropathy-LR-5.0.sav',
 'ascites': '../Models/ascites-LR-2.0.sav',
 'aseptic.necrosis.bone': '../Models/aseptic.necrosis.bone-NB-2.0.sav',
 'aspiration.pneumonia': '../Models/aspiration.pneumonia-NB-2.0.sav',
 'asthenia': '../Models/asthenia-LR-1.0.sav',
 'asthma': '../Models/asthma-LR-3.0.sav',
 'asystole': '../Models/asystole-NB-3.0.sav',
 'atelectasis': '../Models/atelectasis-NB-1.0.sav',
 'balance.disorder': '../Models/balance.disorder-NB-1.0.sav',
 'bladder.retention': '../Models/bladder.retention-LR-1.0.sav',
 'bleb': '../Models/bleb-LR-4.0.sav',
 'blood.in.urine': '../Models/blood.in.urine-NB-4.0.sav',
 'blood.sodium.decreased': '../Models/blood.sodium.decreased-SGD-1.0.sav',
 'blurred.vision': '../Models/blurred.vision-LR-2.0.sav',
 'body.temperature.increased': '../Models/body.temperature.increased-LR-1.0.sav',
 'bone.marrow.failure': '../Models/bone.marrow.failure-NB-1.0.sav',
 'bone.pain': '../Models/bone.pain-LR-1.0.sav',
 'bradycardia': '../Models/bradycardia-NB-1.0.sav',
 'bronchitis': '../Models/bronchitis-NB-1.0.sav',
 'bruise': '../Models/bruise-NB-1.0.sav',
 'bruxism': '../Models/bruxism-LR-2.0.sav',
 'bulging': '../Models/bulging-LR-1.0.sav',
 'burning.sensation': '../Models/burning.sensation-NB-1.0.sav',
 'bursitis': '../Models/bursitis-LR-1.0.sav',
 'candidiasis.of.mouth': '../Models/candidiasis.of.mouth-NB-1.0.sav',
 'cardiac.disease': '../Models/cardiac.disease-NB-1.0.sav',
 'cardiac.enlargement': '../Models/cardiac.enlargement-NB-5.0.sav',
 'cardiac.failure': '../Models/cardiac.failure-LR-3.0.sav',
 'cardiac.murmur': '../Models/cardiac.murmur-LR-1.0.sav',
 'cardiovascular.collapse': '../Models/cardiovascular.collapse-LR-5.0.sav',
 'carpal.tunnel': '../Models/carpal.tunnel-LR-3.0.sav',
 'cataract': '../Models/cataract-LR-5.0.sav',
 'cellulitis': '../Models/cellulitis-NB-5.0.sav',
 'cerebral.infarct': '../Models/cerebral.infarct-LR-1.0.sav',
 'cervicalgia': '../Models/cervicalgia-NB-1.0.sav',
 'chest.pain': '../Models/chest.pain-NB-1.0.sav',
 'chill': '../Models/chill-NB-1.0.sav',
 'cholecystitis': '../Models/cholecystitis-NB-5.0.sav',
 'cholelithiasis': '../Models/cholelithiasis-LR-1.0.sav',
 'chronic.obstructive.airway.disease': '../Models/chronic.obstructive.airway.disease-LR-4.0.sav',
 'cognitive.disorder': '../Models/cognitive.disorder-LR-2.0.sav',
 'coma': '../Models/coma-NB-2.0.sav',
 'confusion': '../Models/confusion-NB-4.0.sav',
 'conjunctivitis': '../Models/conjunctivitis-NB-1.0.sav',
 'constipated': '../Models/constipated-NB-1.0.sav',
 'convulsion': '../Models/convulsion-NB-1.0.sav',
 'coughing.blood': '../Models/coughing.blood-SGD-5.0.sav',
 'deep.vein.thromboses': '../Models/deep.vein.thromboses-LR-1.0.sav',
 'deglutition.disorder': '../Models/deglutition.disorder-LR-1.0.sav',
 'dehydration': '../Models/dehydration-NB-1.0.sav',
 'diarrhea': '../Models/diarrhea-LR-1.0.sav',
 'difficulty.in.walking': '../Models/difficulty.in.walking-NB-1.0.sav',
 'disease.of.liver': '../Models/disease.of.liver-NB-5.0.sav',
 'disorder.Renal': '../Models/disorder.Renal-NB-1.0.sav',
 'diverticulitis': '../Models/diverticulitis-LR-1.0.sav',
 'dizziness': '../Models/dizziness-NB-1.0.sav',
 'drowsiness': '../Models/drowsiness-NB-1.0.sav',
 'drug.toxicity.NOS': '../Models/drug.toxicity.NOS-NB-2.0.sav',
 'drug.withdrawal': '../Models/drug.withdrawal-NB-1.0.sav',
 'dysarthria': '../Models/dysarthria-NB-5.0.sav',
 'dyspepsia': '../Models/dyspepsia-LR-1.0.sav',
 'dysuria': '../Models/dysuria-NB-1.0.sav',
 'edema.extremities': '../Models/edema.extremities-LR-1.0.sav',
 'edema': '../Models/edema-SGD-2.0.sav',
 'elevated.cholesterol': '../Models/elevated.cholesterol-NB-1.0.sav',
 'emesis': '../Models/emesis-SGD-5.0.sav',
 'encephalopathy': '../Models/encephalopathy-NB-5.0.sav',
 'enlarged.liver': '../Models/enlarged.liver-NB-5.0.sav',
 'epistaxis': '../Models/epistaxis-NB-2.0.sav',
 'eruption': '../Models/eruption-LR-1.0.sav',
 'erythema': '../Models/erythema-NB-1.0.sav',
 'esophagitis': '../Models/esophagitis-NB-1.0.sav',
 'facial.flushing': '../Models/facial.flushing-LR-1.0.sav',
 'fatty.liver': '../Models/fatty.liver-LR-5.0.sav',
 'fibromyalgia': '../Models/fibromyalgia-LR-1.0.sav',
 'flatulence': '../Models/flatulence-NB-1.0.sav',
 'flu': '../Models/flu-LR-1.0.sav',
 'fungal.disease': '../Models/fungal.disease-LR-1.0.sav',
 'gall.bladder': '../Models/gall.bladder-NB-1.0.sav',
 'gastric.inflammation': '../Models/gastric.inflammation-LR-1.0.sav',
 'gastroenteritis': '../Models/gastroenteritis-LR-5.0.sav',
 'gastrointestinal.bleed': '../Models/gastrointestinal.bleed-LR-1.0.sav',
 'haematochezia': '../Models/haematochezia-NB-1.0.sav',
 'haemorrhage.rectum': '../Models/haemorrhage.rectum-LR-1.0.sav',
 'haemorrhoids': '../Models/haemorrhoids-NB-1.0.sav',
 'head.injury': '../Models/head.injury-NB-1.0.sav',
 'heart.attack': '../Models/heart.attack-LR-3.0.sav',
 'heart.rate.increased': '../Models/heart.rate.increased-NB-3.0.sav',
 'hemiparesis': '../Models/hemiparesis-LR-5.0.sav',
 'hepatitis': '../Models/hepatitis-NB-5.0.sav',
 'hernia.hiatal': '../Models/hernia.hiatal-NB-4.0.sav',
 'herpes.zoster': '../Models/herpes.zoster-NB-1.0.sav',
 'hive': '../Models/hive-LR-1.0.sav',
 'hot.flash': '../Models/hot.flash-NB-1.0.sav',
 'hyperglycaemia': '../Models/hyperglycaemia-NB-2.0.sav',
 'hyperhidrosis': '../Models/hyperhidrosis-NB-1.0.sav',
 'hyperlipaemia': '../Models/hyperlipaemia-LR-1.0.sav',
 'hypoglycaemia': '../Models/hypoglycaemia-NB-1.0.sav',
 'hypokalaemia': '../Models/hypokalaemia-NB-1.0.sav',
 'hypothyroid': '../Models/hypothyroid-LR-5.0.sav',
 'hypoxia': '../Models/hypoxia-NB-5.0.sav',
 'icterus': '../Models/icterus-LR-2.0.sav',
 'increased.platelet.count': '../Models/increased.platelet.count-LR-1.0.sav',
 'increased.white.blood.cell.count': '../Models/increased.white.blood.cell.count-LR-1.0.sav',
 'infection': '../Models/infection-LR-3.0.sav',
 'insomnia': '../Models/insomnia-NB-3.0.sav',
 'itch': '../Models/itch-NB-1.0.sav',
 'joint.swelling': '../Models/joint.swelling-LR-1.0.sav',
 'kidney.failure': '../Models/kidney.failure-NB-1.0.sav',
 'leucocytosis': '../Models/leucocytosis-LR-1.0.sav',
 'leucopenia': '../Models/leucopenia-LR-5.0.sav',
 'lipoma': '../Models/lipoma-NB-5.0.sav',
 'lobar.pneumonia': '../Models/lobar.pneumonia-LR-1.0.sav',
 'loss.of.consciousness': '../Models/loss.of.consciousness-NB-1.0.sav',
 'loss.of.weight': '../Models/loss.of.weight-SGD-1.0.sav',
 'lung.edema': '../Models/lung.edema-NB-2.0.sav',
 'lung.infiltration': '../Models/lung.infiltration-LR-2.0.sav',
 'malnourished': '../Models/malnourished-LR-1.0.sav',
 'metabolic.acidosis': '../Models/metabolic.acidosis-NB-1.0.sav',
 'migraine': '../Models/migraine-NB-2.0.sav',
 'mucositis.oral': '../Models/mucositis.oral-LR-2.0.sav',
 'muscle.spasm': '../Models/muscle.spasm-LR-1.0.sav',
 'muscle.weakness': '../Models/muscle.weakness-NB-1.0.sav',
 'musculoskeletal.pain': '../Models/musculoskeletal.pain-NB-1.0.sav',
 'narcolepsy': '../Models/narcolepsy-NB-2.0.sav',
 'nasal.congestion': '../Models/nasal.congestion-NB-4.0.sav',
 'nausea': '../Models/nausea-SGD-5.0.sav',
 'nervous.tension': '../Models/nervous.tension-LR-3.0.sav',
 'neumonia': '../Models/neumonia-NB-3.0.sav',
 'neuropathy.peripheral': '../Models/neuropathy.peripheral-LR-1.0.sav',
 'neuropathy': '../Models/neuropathy-NB-1.0.sav',
 'night.sweat': '../Models/night.sweat-NB-4.0.sav',
 'obesity': '../Models/obesity-LR-5.0.sav',
 'osteomyelitis': '../Models/osteomyelitis-LR-2.0.sav',
 'osteoporosis': '../Models/osteoporosis-LR-1.0.sav',
 'palpitation': '../Models/palpitation-NB-1.0.sav',
 'pancreatitis': '../Models/pancreatitis-NB-1.0.sav',
 'panic.attack': '../Models/panic.attack-LR-1.0.sav',
 'pericardial.effusion': '../Models/pericardial.effusion-NB-1.0.sav',
 'phlebothrombosis': '../Models/phlebothrombosis-LR-2.0.sav',
 'pleural.pain': '../Models/pleural.pain-NB-2.0.sav',
 'pulmonary.hypertension': '../Models/pulmonary.hypertension-NB-1.0.sav',
 'pyoderma': '../Models/pyoderma-LR-1.0.sav',
 'respiratory.failure': '../Models/respiratory.failure-NB-1.0.sav',
 'rhabdomyolysis': '../Models/rhabdomyolysis-LR-2.0.sav',
 'rhinorrhea': '../Models/rhinorrhea-LR-1.0.sav',
 'road.traffic.accident': '../Models/road.traffic.accident-LR-5.0.sav',
 'sepsis': '../Models/sepsis-LR-1.0.sav',
 'septic.shock': '../Models/septic.shock-LR-4.0.sav',
 'sinus.tachycardia': '../Models/sinus.tachycardia-NB-4.0.sav',
 'skin.lesion': '../Models/skin.lesion-LR-1.0.sav',
 'sleep.apnea': '../Models/sleep.apnea-LR-5.0.sav',
 'sleep.disorder': '../Models/sleep.disorder-LR-5.0.sav',
 'spinning.sensation': '../Models/spinning.sensation-NB-5.0.sav',
 'still.birth': '../Models/still.birth-NB-5.0.sav',
 'tachycardia.ventricular': '../Models/tachycardia.ventricular-NB-2.0.sav',
 'thrombocytopenia': '../Models/thrombocytopenia-NB-5.0.sav',
 'tinnitus': '../Models/tinnitus-LR-1.0.sav',
 'transient.ischaemic.attack': '../Models/transient.ischaemic.attack-NB-1.0.sav',
 'tremor': '../Models/tremor-NB-1.0.sav',
 'weight.gain': '../Models/weight.gain-NB-1.0.sav',
 'wheeze': '../Models/wheeze-LR-5.0.sav'}

adrs = list(adrModelMap.keys())
adrModels = {}
for adr in adrs:
    adrModels[adr] = pickle.load(open(adrModelMap[adr], 'rb'))

print("Loading Data")
with open("../skipgram/data/mol_sentences.pkl", "rb") as file:
    sentences = pickle.load(file)
atc = [sentence[1][0] for sentence in sentences]
sentences = [sentence[3] for sentence in sentences]

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
