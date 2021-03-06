import time
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/SMILES/<smiles>', methods=['GET'])
def get_smiles(smiles):
    return {'smiles': smiles, 'KNN': 'J', 'ANN': 'N', 'SVM': 'M', 'Actual': 'J'}

@app.route('/LINCS/<genes>', methods=['GET'])
def get_lincs(genes):
    return {
    'genes': genes,
    'AFIB': 'LR',
    'Abnormal.Gait': 'LR',
    'Abnormal.LFTs': 'NB',
    'Aching.joints': 'LR',
    'Acidosis': 'NB',
    'Acute.Respiratory.Distress.Syndrome': 'NB',
    'Adenopathy': 'NB',
    'Amnesia': 'LR',
    'Anorexia': 'NB',
    'Anxiety': 'NB',
    'Arrhythmia': 'NB',
    'Aspartate.Aminotransferase.Increase': 'NB',
    'Back.Ache': 'LR',
    'Bacterial.infection': 'LR',
    'Bilirubinaemia': 'NB',
    'Bladder.inflammation': 'LR',
    'Bleeding': 'NB',
    'Blood.calcium.decreased': 'LR',
    'Candida.Infection': 'NB',
    'Cardiac.decompensation': 'NB',
    'Cardiac.ischemia': 'LR',
    'Cardiomyopathy': 'NB',
    'Chronic.Kidney.Disease': 'LR',
    'Cough': 'LR',
    'Diabetes': 'NB',
    'Difficulty.breathing': 'LR',
    'Diplopia': 'NB',
    'Disorder.Lung': 'LR',
    'Drug.addiction': 'LR',
    'Drug.hypersensitivity': 'LR',
    'Dyspnea.exertional': 'NB',
    'Embolism.pulmonary': 'NB',
    'Excess.potassium': 'NB',
    'Extrasystoles.Ventricular': 'LR',
    'Extremity.pain': 'NB',
    'Fainting': 'NB',
    'Fatigue': 'LR',
    'Feeling.unwell': 'SGD',
    'Hallucination': 'LR',
    'Head.ache': 'NB',
    'Hepatic.failure': 'LR',
    'High.blood.pressure': 'NB',
    'Hypotension.Orthostatic': 'LR',
    'Hypoventilation': 'NB',
    'Incontinence': 'LR',
    'Infection.Upper.Respiratory': 'NB',
    'Infection.Urinary.Tract': 'NB',
    'Intervertebral.Disc.Herniation': 'LR',
    'Mod': 'LR',
    'Nephrolithiasis': 'LR',
    'Neutropenia': 'LR',
    'Osteoarthritis': 'LR',
    'Pain': 'NB',
    'Paraesthesia': 'LR',
    'Pleural.Effusion': 'SGD',
    'Scar': 'LR',
    'Sinusitis': 'NB',
    'abdominal.distension': 'NB',
    'abdominal.pain.upper': 'NB',
    'abdominal.pain': 'LR',
    'abnormal.movements': 'LR',
    'abscess': 'LR',
    'aching.muscles': 'LR',
    'acid.reflux': 'NB',
    'acute.brain.syndrome': 'NB',
    'acute.kidney.failure': 'LR',
    'adenocarcinoma': 'NB',
    'adynamic.ileus': 'SGD',
    'agitated': 'NB',
    'allergic.dermatitis': 'LR',
    'allergies': 'NB',
    'alopecia': 'LR',
    'anaemia': 'NB',
    'angina': 'NB',
    'apoplexy': 'LR',
    'aptyalism': 'LR',
    'arterial.pressure.NOS.decreased': 'LR',
    'arterial.pressure.NOS.increased': 'LR',
    'arteriosclerosis': 'LR',
    'arteriosclerotic.heart.disease': 'LR',
    'arthritis': 'NB',
    'arthropathy': 'LR',
    'ascites': 'LR',
    'aseptic.necrosis.bone': 'NB',
    'aspiration.pneumonia': 'NB',
    'asthenia': 'LR',
    'asthma': 'LR',
    'asystole': 'NB',
    'atelectasis': 'NB',
    'balance.disorder': 'NB',
    'bladder.retention': 'LR',
    'bleb': 'LR',
    'blood.in.urine': 'NB',
    'blood.sodium.decreased': 'SGD',
    'blurred.vision': 'LR',
    'body.temperature.increased': 'LR',
    'bone.marrow.failure': 'NB',
    'bone.pain': 'LR',
    'bradycardia': 'NB',
    'bronchitis': 'NB',
    'bruise': 'NB',
    'bruxism': 'LR',
    'bulging': 'LR',
    'burning.sensation': 'NB',
    'bursitis': 'LR',
    'candidiasis.of.mouth': 'NB',
    'cardiac.disease': 'NB',
    'cardiac.enlargement': 'NB',
    'cardiac.failure': 'LR',
    'cardiac.murmur': 'LR',
    'cardiovascular.collapse': 'LR',
    'carpal.tunnel': 'LR',
    'cataract': 'LR',
    'cellulitis': 'NB',
    'cerebral.infarct': 'LR',
    'cervicalgia': 'NB',
    'chest.pain': 'NB',
    'chill': 'NB',
    'cholecystitis': 'NB',
    'cholelithiasis': 'LR',
    'chronic.obstructive.airway.disease': 'LR',
    'cognitive.disorder': 'LR',
    'coma': 'NB',
    'confusion': 'NB',
    'conjunctivitis': 'NB',
    'constipated': 'NB',
    'convulsion': 'NB',
    'coughing.blood': 'SGD',
    'deep.vein.thromboses': 'LR',
    'deglutition.disorder': 'LR',
    'dehydration': 'NB',
    'diarrhea': 'LR',
    'difficulty.in.walking': 'NB',
    'disease.of.liver': 'NB',
    'disorder.Renal': 'NB',
    'diverticulitis': 'LR',
    'dizziness': 'NB',
    'drowsiness': 'NB',
    'drug.toxicity.NOS': 'NB',
    'drug.withdrawal': 'NB',
    'dysarthria': 'NB',
    'dyspepsia': 'LR',
    'dysuria': 'NB',
    'edema.extremities': 'LR',
    'edema': 'SGD',
    'elevated.cholesterol': 'NB',
    'emesis': 'SGD',
    'encephalopathy': 'NB',
    'enlarged.liver': 'NB',
    'epistaxis': 'NB',
    'eruption': 'LR',
    'erythema': 'NB',
    'esophagitis': 'NB',
    'facial.flushing': 'LR',
    'fatty.liver': 'LR',
    'fibromyalgia': 'LR',
    'flatulence': 'NB',
    'flu': 'LR',
    'fungal.disease': 'LR',
    'gall.bladder': 'NB',
    'gastric.inflammation': 'LR',
    'gastroenteritis': 'LR',
    'gastrointestinal.bleed': 'LR',
    'haematochezia': 'NB',
    'haemorrhage.rectum': 'LR',
    'haemorrhoids': 'NB',
    'head.injury': 'NB',
    'heart.attack': 'LR',
    'heart.rate.increased': 'NB',
    'hemiparesis': 'LR',
    'hepatitis': 'NB',
    'hernia.hiatal': 'NB',
    'herpes.zoster': 'NB',
    'hive': 'LR',
    'hot.flash': 'NB',
    'hyperglycaemia': 'NB',
    'hyperhidrosis': 'NB',
    'hyperlipaemia': 'LR',
    'hypoglycaemia': 'NB',
    'hypokalaemia': 'NB',
    'hypothyroid': 'LR',
    'hypoxia': 'NB',
    'icterus': 'LR',
    'increased.platelet.count': 'LR',
    'increased.white.blood.cell.count': 'LR',
    'infection': 'LR',
    'insomnia': 'NB',
    'itch': 'NB',
    'joint.swelling': 'LR',
    'kidney.failure': 'NB',
    'leucocytosis': 'LR',
    'leucopenia': 'LR',
    'lipoma': 'NB',
    'lobar.pneumonia': 'LR',
    'loss.of.consciousness': 'NB',
    'loss.of.weight': 'SGD',
    'lung.edema': 'NB',
    'lung.infiltration': 'LR',
    'malnourished': 'LR',
    'metabolic.acidosis': 'NB',
    'migraine': 'NB',
    'mucositis.oral': 'LR',
    'muscle.spasm': 'LR',
    'muscle.weakness': 'NB',
    'musculoskeletal.pain': 'NB',
    'narcolepsy': 'NB',
    'nasal.congestion': 'NB',
    'nausea': 'SGD',
    'nervous.tension': 'LR',
    'neumonia': 'NB',
    'neuropathy.peripheral': 'LR',
    'neuropathy': 'NB',
    'night.sweat': 'NB',
    'obesity': 'LR',
    'osteomyelitis': 'LR',
    'osteoporosis': 'LR',
    'palpitation': 'NB',
    'pancreatitis': 'NB',
    'panic.attack': 'LR',
    'pericardial.effusion': 'NB',
    'phlebothrombosis': 'LR',
    'pleural.pain': 'NB',
    'pulmonary.hypertension': 'NB',
    'pyoderma': 'LR',
    'respiratory.failure': 'NB',
    'rhabdomyolysis': 'LR',
    'rhinorrhea': 'LR',
    'road.traffic.accident': 'LR',
    'sepsis': 'LR',
    'septic.shock': 'LR',
    'sinus.tachycardia': 'NB',
    'skin.lesion': 'LR',
    'sleep.apnea': 'LR',
    'sleep.disorder': 'LR',
    'spinning.sensation': 'NB',
    'still.birth': 'NB',
    'tachycardia.ventricular': 'NB',
    'thrombocytopenia': 'NB',
    'tinnitus': 'LR',
    'transient.ischaemic.attack': 'NB',
    'tremor': 'NB',
    'weight.gain': 'NB',
    'wheeze': 'LR'}

if __name__ == '__main__':
    app.run()
