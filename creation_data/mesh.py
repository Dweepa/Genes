from rdflib import Graph
import pickle

mesh = pickle.load(open("../data/mesh_nt_file","rb"))
print(len(mesh))

print("mesh")

len(mesh)