import numpy as np
import pickle


L = {}

path = "./lwe_instances/reduced_lattices/"
filename = "tph_170_binomial_3.0000_1_91_97.pkl" #fig. 2 (left)
# filename = "tph_170_binomial_3.0000_1_91_93.pkl" #fig. 2 (right)
n_guess_coords = [ int( filename.split("_")[4] ) ]
with open(path+filename,"rb") as file:
    L.update( pickle.load(file) )
    
print(len(L))
D = L

AGGR = {}
AGGRT = {}
just_dists = []
walltimes = []
for key in D.keys():
    n, lat_index, n_slicer_coord, n_guess_coord, ex_cntr = key
    walltime =  abs(D[key]['wrong_guess_time_alg3']) + abs(D[key]['wrong_guess_time_alg2']) #D[key]["walltime"]
    dist_bnd = D[key]["dist_bnd"]
    succ = D[key]["succ"]
    fail_reason = D[key]["fail_reason"]
    just_dists.append( dist_bnd )
    if not n_guess_coord in AGGR.keys():
        AGGR[n_guess_coord] = {}
        AGGRT[n_guess_coord] = {}
    if not dist_bnd in AGGR[n_guess_coord].keys():
        AGGR[n_guess_coord][dist_bnd] = [0, 0, 0, 0] #totnum, succ, failparasites, failothers 
        AGGRT[n_guess_coord][dist_bnd] = walltime
    walltimes.append(walltime)
    AGGR[n_guess_coord][dist_bnd][0] += 1
    AGGR[n_guess_coord][dist_bnd][1] += succ
    if not succ:
        if "par" in fail_reason:
            AGGR[n_guess_coord][dist_bnd][2] += 1
        else:
            AGGR[n_guess_coord][dist_bnd][3] += 1

for key in AGGR.keys():
    for kkey in AGGR[key]:
        AGGRT[key][kkey] = AGGRT[key][kkey]/AGGR[key][kkey][0]
        AGGR[key][kkey] = {
            "succ": AGGR[key][kkey][1]/AGGR[key][kkey][0],
            "para": AGGR[key][kkey][2]/AGGR[key][kkey][0],
            "oth": AGGR[key][kkey][3]/AGGR[key][kkey][0],
        }

for kappa in n_guess_coords: 
    l, r = [], []
    for key in AGGR[kappa]:
        l.append( (key,AGGR[kappa][key]) )
    
    l = sorted( l, key= lambda x: -x[1]["succ"] )
    ls, lf_para, lf_oth = [], [], []
    for ll in l:
        if ll[1]["succ"]:
            ls.append(ll[0])
        else:
            if ll[1]["oth"]:
                lf_oth.append(ll[0])
            else:
                lf_para.append(ll[0])
            
    

    print(f"lf_para, lf_oth, ls: {len(lf_para),len(lf_oth),len(ls)}")
    tot = len(lf_para)+len(lf_oth)+len(ls)
    print( f"sum: {tot}" )
    succpr = (len(ls) / tot)*100.
    Hs = histogram(ls,bins=20,color="green", label=["Succ"], alpha=0.4, title=f"Hybrid ternary $n$={n}, sli_dim={n_slicer_coord}, kappa={kappa}, x10 nrand , succ={succpr:.01f}%" , figsize=10.5)
    Hpara = histogram(lf_para,bins=20,color="blue", label=["Too short"], alpha=0.4)
    Hpoth = histogram(lf_oth,bins=20,color="red", label=["Fail"], alpha=0.4)
    Histo = list_plot([(1,1)], color="white")
    if len(ls):
        Histo += Hs
    if len(lf_para):
        Histo += Hpara
    if len(lf_oth):
        Histo += Hpoth
    filename = f"apprf_histograms{n}_{kappa}.png"
    Histo.save_image( filename )
    print(f"Figure saved to {filename}")
