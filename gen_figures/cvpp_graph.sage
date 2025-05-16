import pickle
import numpy as np

n = 80
sieve_dim = n
filename = f"./lwe_instances/reduced_lattices/slicsucc_{n}.pkl"
with open(filename,"rb") as file:
    aggregated_data = pickle.load(file)

L = {}
nlats = len(aggregated_data)
for aggregated_data_ in aggregated_data:
    for approx, Ds in aggregated_data_:
        if not approx in L.keys():
            L[approx] = []
        L[approx] +=  Ds 

# L = aggregated_data
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

P = list_plot( lbab, color = "red", plotjoined=True, legend_label="Babai" ) +\
list_plot( ls[1.], color = "blue", legend_label="Slicer", plotjoined = True ) +\
list_plot( ls[5.], color = "darkcyan", legend_label="Slicer x5 rerand", plotjoined = True ) +\
list_plot( ls[10.], color = "purple", legend_label="Slicer x10 rerand", plotjoined = True ) +\
line( [(0.75,0.5),(1.1,0.5)], color = "black" ) + line( [(0.95,0.),(0.95,1.0)], color = "black" )
# point( [ [0.9,0.886], [0.95,641/1000.], [1.0,291/1000.] ] , color="grey", size=60, legend_label="batchcvpp x10 rerand true" )  +\
# point( [ [0.9,0.724], [0.95,0.44], [1.0,0.248] ] , color="red", size=60, legend_label="batchcvpp x10 rerand false" )

P.set_legend_options(loc='lower left')

# P.show( title=f"dim={dim} $\\beta$ = 53 | nlats={nlats}", axes_labels=['approx. fact', 'proba'], figsize=7.5 )
P.save_image( f"cvp_graph_{n}.png", 
    title=f"dim={dim} $\\beta$ = 53 | nlats={nlats} | s_dim = {sieve_dim}", 
    axes_labels=['approx. fact', 'proba'], figsize=7.5 , xmin=0.9, xmax=1.0)
print(f"Saved figure to cvp_graph_{n}.png")
