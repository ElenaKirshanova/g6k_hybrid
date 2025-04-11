import pickle
import numpy as np

n = 60
sieve_dim = n
filename = f"slicsucc_{n}.pkl"
with open(filename,"rb") as file:
    aggregated_data = pickle.load(file)


L = {}
nlats = len(aggregated_data)
for aggregated_data_ in aggregated_data:
    print(aggregated_data_)
    for approx, Ds in aggregated_data_:
        if not approx in L.keys():
            L[approx] = []
        L[approx] +=  Ds 

Laggr = {}
for nrand in L.keys():
    if not nrand in Laggr.keys():
        Laggr[nrand] = {}
        
    Dlist = L[nrand]
    D = {}
    latnum = 0
    for Dcur in Dlist:
        latnum += 1
        for n, appr in Dcur:
            if not (n,appr) in D.keys():
                D[(n,appr)] = [0,0,0]
            D[(n,appr)][1] += Dcur[(n,appr)][1] #
            D[(n,appr)][2] += Dcur[(n,appr)][2]
    for n, appr in D:
        D[(n,appr)][1] /= latnum
        D[(n,appr)][2] /= latnum
    Laggr[nrand] = D

ls = {}
lbab = {}
lacvp = {}

for key in L.keys():
    nrand = key
    ls[nrand] = {}
    lacvp[nrand] = {}
    for D in L[key]:
        for kkey in D.keys():
            dim, appfact = kkey
            _, sli, bab, acvp = D[kkey]
    
            if not appfact in ls[nrand].keys():
                ls[nrand][appfact] = []
            if not appfact in lacvp[nrand].keys():
                lacvp[nrand][appfact] = []
            
            if not appfact in lbab.keys():
                lbab[appfact] = []
                
            ls[nrand][appfact].append( sli )
            lacvp[nrand][appfact].append( acvp )
            lbab[appfact].append( bab )
    for kkey in D.keys():
        _, appfact = kkey
        ls[nrand][appfact] = np.mean(ls[nrand][appfact])
        lacvp[nrand][appfact] = np.mean(lacvp[nrand][appfact])
        
for appfact in lbab.keys():
    lbab[appfact] = np.mean(lbab[appfact])
    

print( ls )
print( lbab )
print( lacvp )

print( f"nlats: {nlats}" )

P = list_plot( lbab, color = "red", plotjoined=True, legend_label="Babai" )
for nrand in L.keys():
    P += list_plot( ls[nrand], color = Color([uniform(0,1) for i in range(3)]), legend_label=f"Slicer x{nrand:0.2f} rerand", plotjoined = True )
P += line( [(0.75,0.5),(1.1,0.5)], color = "black" ) + line( [(0.95,0.),(0.95,1.0)], color = "black" )
P.set_legend_options(loc='lower left')

filename = f"sli_appr_fact_{dim}.png"
P.save( filename,
    title=f"dim={dim} | nlats={nlats} | s_dim = {sieve_dim}", 
    axes_labels=['approx. fact', 'proba'], figsize=7.5 , xmin=0.9, xmax=1.0
)
print(f"saved at {filename}")