import numpy as np
import pickle

n, beta = 120, 55

filename = f"./lwe_instances/reduced_lattices/tail_bdd_n{n}_b{beta}.pkl"
with open(filename,"rb") as file:
    output = pickle.load(file)

l = []
for DD in output:
    for key in DD.keys():
        l += DD[key]

lsucc, lfail = [], []
for ll in l:
    if ll[1]:
        lsucc.append( ll[0] )
    else:
        lfail.append( ll[0] )

Hs = histogram( lsucc, bins=20, color="green" )
Hf = histogram( lfail, bins=20, color="red" )
# print(f"mean gamma: {np.mean([ll[0] for ll in l])}")

filename = f"tbdd_histo_n{n}_b{beta}.png"
(Hs+Hf).save_image( filename, title=f"Tail-BDD n={n}, beta={beta}, {len(l)} instances" )
print(f"Saved figure to {filename}")