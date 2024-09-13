import pickle
import numpy as np
from math import log
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from sample import *

# def guess(kappa, nsampl, distribution):
#     ans = distribution.sample(kappa)
#     succ = False
#     for tries in range(nsampl):
#         if tries%100000 ==0:
#             print(tries,end=", ")
#         cand =  distribution.sample(kappa)
#         if cand == ans:
#             # print(f"succsess!")
#             tries+=1
#             succ = True
#             break

# def experiment(kappa, nsampl, distribution, nexp=1):
#     print(f"kappa, nsampl, log(nsampl,2): {kappa, nsampl}, {log(nsampl,2): .5f}")
#
#     pool = Pool(processes = nthreads )
#     tasks = []
#     for _ in range(nexp):
#         nsampl = 1024*ceil( 2 ** ( dist.entropy * kappa ) )
#         tasks.append( pool.apply_async(
#             experiment, (kappa, nsampl, dist)
#         ) )
#
#     out = []
#     for t in tasks:
#         kappa,tries,succ =  t.get()
#         out.append([kappa,tries,succ])
#     # print()
#
#     print()
#     return out

def experiment(kappa, nsampl, distribution):
    print(f"kappa, nsampl, log(nsampl,2): {kappa, nsampl}, {log(nsampl,2): .5f}")
    ans = distribution.sample(kappa)

    succ = False
    for tries in range(nsampl):
        # if tries%100000 ==0:
            # print(tries,end=", ")
        cand =  distribution.sample(kappa)
        if cand == ans:
            # print(f"succsess!")
            tries+=1
            succ = True
            break
    print()
    return [kappa,tries,succ]

nthreads = 10
nexp = 200
eta, kappa_min, kappa_max = 3, 2, 8
dist = centeredBinomial(eta)

kappas = range(kappa_min, kappa_max+1)
pool = Pool(processes = nthreads )
tasks = []
for kappa in kappas:
    for _ in range(nexp):
        nsampl = 2*ceil( 2 ** ( dist.entropy * kappa ) )
        tasks.append( pool.apply_async(
            experiment, (kappa, nsampl, dist)
        ) )

output=[]
for t in tasks:
        output.append( t.get() )

pool.close()

"""
output=[]
for kappa in kappas:
    nsampl = 1024*ceil( 2 ** ( dist.entropy * kappa ) )
    output.append( experiment(kappa, nsampl, dist, nexp) )
"""

print( output )

dict = {}
for o in output:
    if not (o[0] in dict.keys()):
        dict[o[0]] = [ [o[1]], [o[2]] ]
    else:
        dict[o[0]][0].append(o[1])
        dict[o[0]][1].append(o[2])

print(dict)
with open(f"exp{kappa_min}_{kappa_max}.pkl","wb") as f:
    pickle.dump(dict,f)

for key in dict.keys():
    print(f"- - - kappa:{kappa} - - -")
    print(f"tries: avg:{np.mean(dict[key][0])}, med:{np.median(dict[key][0])}, std:{np.std(dict[key][0])}")
    print(f"Success ratio: {dict[key][1].count(True) / len(dict[key][0])}")
