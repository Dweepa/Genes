from gensim.models import word2vec
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import pickle
import sys

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

# df = pd.read_csv("../data/drug_class_identification/phase1/drugbank.csv")
# df.drop(df[df.smiles.isna()].index, inplace = True)
# df.drop(df[df.atc.isna()].index, inplace = True)
# df.drop(df[[False if len(smile)<250 else True for smile in df.smiles]].index, inplace = True)


# sentences = []

# for x in df.index:
# 	try:
# 		sentence = []
# 		sentence.append(df.name[x])
# 		sentence.append(df.atc[x])
# 		sentence.append(df.smiles[x])
# 		sentence.append(mol2alt_sentence(Chem.MolFromSmiles(df.smiles[x]), 2))
# 		sentences.append(sentence)
# 	except Exception as e:
# 		print(e)

# with open("./data/mol_sentences_2.pkl", "wb") as file:
# 	pickle.dump(sentences, file)

df = pd.read_csv("../data/pcba.csv", nrows=100000)
smiles = df.smiles
df = 0

smiles.dropna(inplace=True)

sentences = []

total = smiles.shape[0]
i = 0 
for smile in smiles:
	i+=1
	try:
		sentences.append(mol2alt_sentence(Chem.MolFromSmiles(smile), 2))
		sys.stdout.write(f"\r{i}/{total}")
	except Exception as e:
		# print(e)
		pass

print("\n", len(sentences), sep="")

with open("./data/pcba_mol_sentences_2.pkl", "wb") as file:
	pickle.dump(sentences, file)
