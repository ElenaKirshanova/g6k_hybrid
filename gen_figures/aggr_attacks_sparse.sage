import os, re
import time

import pickle 
import numpy as np
from hybrid_estimator.batchCVP import batchCVPP_cost
from sample import Distribution, ternaryDist, centeredBinomial

from hybrid_estimator.batchCVP import batchCVPP_cost
out_path = "lwe_instances/reduced_lattices/"

sec_type = "ternary"
sec_param = float(1/12)
q = 3329
dist=sec_type
dist_param=sec_param

    
lwe_inst = [ #do we need lats_per_dim and inst_per_lat?
    {"n": 170, "q": 3329, "dist": 'ternary', "dist_param": sec_param},
    {"n": 180, "q": 3329, "dist": 'ternary', "dist_param": sec_param},
    {"n": 190, "q": 3329, "dist": 'ternary', "dist_param": sec_param},
    {"n": 200, "q": 3329, "dist": 'ternary', "dist_param": sec_param},
    {"n": 210, "q": 3329, "dist": 'ternary', "dist_param": sec_param},
]

hparams = { #n_guess_coord's, n_slicer_coord from preprocessing.py
    160: (6,45),
    170: (8,49),
    180: (8,53),
    190: (8,60),
    200: (8,67),
    210: (8,77)
}


if sec_type=="binomial":
    distrib = centeredBinomial(sec_param)
elif sec_type=="ternary":
     distrib = ternaryDist(sec_param)

data = []
path = "./lwe_instances/reduced_lattices/"
# regex = re.compile(r'^report_prehyb_(\d+)_')
pattern = re.compile(r'''
    ^report_prehyb_             
    (?P<n>\d+)_                 
    (?P<q>\d+)_                 
    (?P<dist>[^_]+)_            
    (?P<dist_param>[+-]?\d+\.\d{4})_  
    (?P<seed>\d+)_              
    (?P<kappa>\d+)_             
    (?P<sieve_dim_min>\d+)_     
    (?P<sieve_dim_max>\d+)_     
    (?P<beta_bkz>\d+)           
    \.pkl$
''', re.VERBOSE)
regex = re.compile(pattern)

for path, directories, files in os.walk(path):
    for candidate in files:
        match = regex.match(candidate)
        if match and sec_type in candidate and f"{sec_param:0.4f}" in candidate:
            gd = match.groupdict()
            n             = int(gd['n'])
            q             = int(gd['q'])
            dist          = gd['dist']
            dist_param    = float(gd['dist_param'])
            seed          = int(gd['seed'])
            kappa         = int(gd['kappa'])
            sieve_dim_min = int(gd['sieve_dim_min'])
            sieve_dim_max = int(gd['sieve_dim_max'])
            beta_bkz      = int(gd['beta_bkz'])
            try:
                try:
                    if ( #ncur in [tmp["n"] for tmp in lwe_inst] and
                        n in [tmp["n"] for tmp in lwe_inst] and
                        kappa == hparams[n][0] and
                        beta_bkz == hparams[n][1] and
                        sec_type in candidate and
                        f"{sec_param:0.4f}" in candidate and
                        f"_{hparams[n][0]}_" in candidate
                    ):
                        with open( path+candidate, "rb" ) as file:
                            data.append( pickle.load(file) )
                except KeyError as err:
                    print( err )
                    pass
            except ValueError as expt:
                print(expt)
                pass

processed_data = {}
for D in data: #loading from the text output
    n, q, dist, sec_param, seed = D["params"]
    if dist != sec_type:
        continue
    sieve_dim_max = D["sieve_dim_max"]
    sieve_dim_min = D["sieve_dim_min"]
    kappa = D["kappa"]
    if kappa != hparams[n][0]: #or sieve_dim_min != hparams[n][1]:
        continue
    
    bkz_runtime = D["bkz_runtime"]
    bdgl_runtime = D["bdgl_runtime"]

    nsampl = ceil( 2 ** ( distrib.entropy * kappa ) )

    for i in range(len(bdgl_runtime)):
        sievedim = sieve_dim_max-i
        latdim = n - kappa
        if not (n,kappa,sievedim,latdim) in processed_data.keys():
            processed_data[(n,kappa,sievedim,latdim)] = {"bkz_runtime": [], "bdgl_runtime": [], "succrate": 0.}
        processed_data[(n,kappa,sievedim,latdim)]["bkz_runtime"].append( bkz_runtime )
        processed_data[(n,kappa,sievedim,latdim)]["bdgl_runtime"].append( bdgl_runtime )

aggrigated_data = {}
for (n,k,sievedim,latdim) in processed_data.keys():
    D = processed_data[(n,k,sievedim,latdim)]
    bkz_runtime = np.mean( D["bkz_runtime"] )
    bdgl_runtime = np.mean( D["bdgl_runtime"], axis=int(0) )
    
    aggrigated_data[ (n,k,sievedim) ] =  bkz_runtime + np.zeros(len(bdgl_runtime))



l0, l1 = {}, {}

for (n,k,sievedim) in aggrigated_data.keys():
    l0[n] = aggrigated_data[(n,k,sievedim)][-1]
    l1[n] = aggrigated_data[(n,k,sievedim)][0]


preprocess_hyb_time = deepcopy(l0)

P = list_plot_semilogy(l0, plotjoined=True, base=10, axes_labels=["$n$", "$log(T)$"], color="green", legend_label="Hybrid preprocessing")
# P.show( title=f'Preprocessing Time for Hybrid, Kyber-$n$.', figsize=12 )

# - - - processing the hybrid attack
L = {}
available_ns = []
"""
The tha files now have a new naming convention.
"""

data = []
path = "./lwe_instances/reduced_lattices/"
pattern = re.compile(r'''
    ^tph_                          
    (?P<n>\d+)_                    
    (?P<dist>[^_]+)_               
    (?P<dist_param>[+-]?\d+\.\d{4})_  
    (?P<kappa>\d+)_                
    (?P<beta_bkz>\d+)_             
    (?P<sieve_dim_max>\d+)         
    \.pkl$                         
''', re.VERBOSE)
regex = re.compile(pattern)

lol=0
max_n = 0
for path, directories, files in os.walk(path):
    lol+=1
    for candidate in files:
        match = regex.match(candidate)
        if match and sec_type in candidate and f"{sec_param:0.4f}" in candidate:
            gd = match.groupdict()
            n             = int(gd['n'])
            dist          = gd['dist']
            dist_param    = float(gd['dist_param'])
            kappa         = int(gd['kappa'])
            sieve_dim_max = int(gd['sieve_dim_max'])
            beta_bkz      = int(gd['beta_bkz'])
            
            available_ns.append(n)
            with open(path+candidate,"rb") as file:
                L.update( pickle.load(file) )
            

wtimes = {}
succs = {}
for n in available_ns:
    wtimes[n] = []
    if not n in succs:
        succs[n] = [0,0]
    
NRAND_FACTOR = 10.
for key in L:
    n, _, n_slicer_coord, n_guess_coord, _ = key
    if n_guess_coord == hparams[n][0]:
        nrand_, _ = batchCVPP_cost(n_slicer_coord,100,L[key]["g6k_len"] **(1./n_slicer_coord),1)
        nrand = ceil(NRAND_FACTOR*(1./nrand_)**n_slicer_coord)
        utar_per_batch = ceil( L[key]["g6k_len"] / nrand ) #how many unique targets in batch
    
        curtime = abs( L[key]["wrong_guess_time_alg2"] ) + abs( L[key]["wrong_guess_time_alg3"] )
        overhead_tsieve = L[key]["overhead_tsieve"]
        batnum = ceil( L[key]["key_num"]/utar_per_batch )
        curtime *= batnum  #time * how many batches needed
        wtimes[n].append( [curtime , overhead_tsieve] )
        succs[n][0]+=1
        succs[n][1]+=L[key]['succ']

""" #this estimation was a mistake. We do not perform sieving for each batch -- only once
for n in available_ns:
    wtimes[n] = np.mean(wtimes[n])
    try:
        cur_succ_rate = float(succs[n][1] / succs[n][0])
    except TypeError:
        cur_succ_rate = succs[n]
    succs[n] = cur_succ_rate if cur_succ_rate>0 else 1/100.

ltot_hyb_att = {}
for key in wtimes.keys():
    walltime = wtimes[key]
    ltot_hyb_att[key] = l1[key] + ( 2*walltime ) / succs[key]  #success rate is 1/2 * slicer's proba
"""

for n in available_ns:
    wtimes[n] = np.mean(wtimes[n], axis=0) #we don`t need to sieve for each new batch
    try:
        cur_succ_rate = float(succs[n][1] / succs[n][0])
    except TypeError:
        cur_succ_rate = succs[n]
    succs[n] = cur_succ_rate if cur_succ_rate>0 else 1/100.

ltot_hyb_att = {}
for key in wtimes.keys():
    walltime = wtimes[key]
    ltot_hyb_att[key] = float( l1[key] + walltime[1] + ( 2*walltime[0] ) / succs[key] )  #success rate is 1/2 * slicer's proba

# - - - now we process the two-step attack

data = []
path = "./lwe_instances/reduced_lattices/"
pattern = re.compile(r'''
    ^tsa_                          
    (?P<n>\d+)_                    
    (?P<dist>[^_]+)_               
    (?P<dist_param>[+-]?\d+\.\d{4})     
    \.pkl$                         
''', re.VERBOSE)
regex = re.compile(pattern)

L_two_step = {}
for path, directories, files in os.walk(path):
    for candidate in files:
        match = regex.match(candidate)
        if match and sec_type in candidate and f"{sec_param:0.4f}" in candidate:
            gd = match.groupdict()
            n             = int(gd['n'])
            dist          = gd['dist']
            dist_param    = float(gd['dist_param'])
            
            available_ns.append(n)
            with open(path+candidate,"rb") as file:
                L_two_step[n] = pickle.load(file)

L_two_step_ = {}
#print( L_two_step )
for n in L_two_step.keys():
    Ts = []

    succs = [0,0]
    cntr=0
    data = L_two_step[n]
    #print(data)
    for D in data:
        tmp = 0
        for bkz in D["bkz_invoked"].values():
            tmp +=  sum( bkz["times"] )
        lol = sum( [ll[1] for ll in D["svp_calls"]] )
        tmp += sum( [ll[1] for ll in D["svp_calls"]] )
        Ts.append( tmp )
        succs[1]+=1
        cntr+=1
        if D["succ"]:
            succs[0]+=1

    avgtime = np.mean( Ts )
    print( avgtime, avgtime*succs[1]/succs[0] )
    print( succs )
    L_two_step_[n] = avgtime*succs[1]/succs[0]

P += list_plot_semilogy(L_two_step_, plotjoined=True, base=10, axes_labels=["$n$", "$log(T)$"], color="orange", legend_label="Two-step total")

print(f"succs: {succs}")
print(f"ltot_hyb_att: {ltot_hyb_att}")
print(f"two_step_timings: {L_two_step_}")
# - - -

P += list_plot_semilogy(ltot_hyb_att, plotjoined=True, base=10, axes_labels=["$n$", "$log(T)$"], color="red", legend_label="Hybrid total")
plotfilename = f"time_{dist}_{dist_param:0.4f}_{available_ns}.png"

filename = f"hybVSprima_d_{dist}_{dist_param:0.4f}.png"
P.save_image( filename, title=f'Preprocessing + attack Time for Hybrid, Kyber-$n$. Ternary {dist_param}', figsize=12 )
print(f"Saved figure to {filename}")