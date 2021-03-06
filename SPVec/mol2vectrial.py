from gensim.models import word2vec
from rdkit import Chem, AllChem
import pandas as pd
import pickle

from helper_functions import mol2alt_sentence

model = word2vec.Word2Vec.load('./model_300dim.pkl')

df = pd.read_csv("../data/drug_class_identification/phase1/drugbank.csv", nrows=100)
df.drop(df[df.smiles.isna()].index, inplace = True)
# df.drop(df[df.atc.isna()].index, inplace = True)
# df.drop(df[[False if len(smile)<250 else True for smile in df.smiles]].index, inplace = True)


# sentences = []

# for x in df.index:
# 	try:
# 		sentence = []
# 		sentence.append(df.name[x])
# 		sentence.append(df.atc[x])
# 		sentence.append(df.smiles[x])
# 		sentence.append(mol2alt_sentence(Chem.MolFromSmiles(df.smiles[x]), 1))
# 		sentences.append(sentence)
# 	except:
# 		pass

# with open("mol_sentences.pkl", "wb") as file:
# 	pickle.dump(sentences, file)
# for aa in aas:
# 	sentence = mol2alt_sentence(aa, 1)
# 	print(len(sentence))

# # print(model.wv.word_vec('2246728737'))


print(mol2alt_sentence(Chem.MolFromSmiles(list(df.smiles)[0]), 1))
