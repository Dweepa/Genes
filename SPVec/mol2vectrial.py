from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
import pandas as pd
import pickle

model = word2vec.Word2Vec.load('./model_300dim.pkl')

df = pd.read_csv("../data/drug_class_identification/phase1/drugbank.csv")
df.drop(df[df.smiles.isna()].index, inplace = True)
df.drop(df[df.atc.isna()].index, inplace = True)
df.drop(df[[False if len(smile)<250 else True for smile in df.smiles]].index, inplace = True)

print(df.shape)

sentences = []

for x in df.index:
	try:
		sentence = []
		sentence.append(df.name[x])
		sentence.append(df.atc[x])
		sentence.append(df.smiles[x])
		sentence.append(mol2alt_sentence(Chem.MolFromSmiles(df.smiles[x]), 1))
		sentences.append(sentence)
	except:
		pass

with open("mol_sentences.pkl", "wb") as file:
	pickle.dump(sentences, file)
# for aa in aas:
# 	sentence = mol2alt_sentence(aa, 1)
# 	print(len(sentence))

# # print(model.wv.word_vec('2246728737'))