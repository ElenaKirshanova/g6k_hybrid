import math
from math import log, sqrt
from hybrid_estimator.batchCVP import batchCVPP_cost
import matplotlib.pyplot as plt
import pickle

nbs = [(50+i*10) for i in range(1)]
beta = 48
Ddata = {}
for n in nbs:
    filename = f"dbsize_{n}_{beta}_exp.pkl"
    with open(filename,"rb") as file:
        D = pickle.load(file)
    Ddata[n] = D

cntr = 0
fig, ax = plt.subplots( figsize=(7.5, 6), layout='constrained' )
for key in Ddata:
    dim = key
    Nexperiments = Ddata[key]["Nexperiments"]
    data = Ddata[key]["density_plots"]

    N = len(data)
    print(N)
    print(data[0])

    average = [0]*len(data[0])
    factors = [0]*len(data[0])
    for i in range(len(data)):
        for j, el in enumerate(data[i]):
            average[j]+=el[1]

    for i, el in enumerate(data[0]):
        factors[i] = el[0]
        average[i] = (average[i]/N)/Nexperiments

    print(factors)
    assert(len(factors)==len(average))


    # db_sizes = {60: 17919, 65: 36784, 70: 75511, 75: 155008, 80: 318200, 90: 1340890, 100: 5650500}
    # db_size = db_sizes[dim]

    db_size = Ddata[key]["dbsize_start"]
    alpha = db_size**(1/dim)
    nrand_, _ = batchCVPP_cost(dim,100,alpha,1) #100 can be any constant >1
    # print((1./nrand_)**dim)
    print(f"alpha: {alpha} vs sqrt(4/3): {math.sqrt(4/3.)}")

    theory = [0]*len(data[0])
    dbs = []
    for i, el in enumerate(factors):
        # prob, _ = batchCVPP_cost(dim,math.log((1./nrand_)**dim,2),math.sqrt(4./3)*(el)**(1./dim),1)
        prob, _ = batchCVPP_cost(dim,math.log((1./nrand_)**dim,2),(alpha)*(el**(1./dim)),1.)
        theory[i] = prob
        dbs.append( el )
    print("Theory:")
    print(theory)
    print(f"Practise:")
    print(average)


    with open('dbsize_{dim}.csv', "w") as f:
        f.write('factor,succ \n')
        for i in range(len(factors)):
            f.write(str(factors[i])+", "+str(average[i])+'\n')

    xaxis = [ log(tmp,2) for tmp in dbs ]
    yaxis_th = theory
    yaxis_pr = average
    if cntr==0:
        ax.plot( xaxis, yaxis_th, label="Theory" )
    ax.plot( xaxis, yaxis_pr, label= f"Practice n={dim}" )
    cntr+=1

ax.set_xlabel('log_2 shrinking factor')
ax.set_ylabel('Success rate')
plt.title("Database size vs. succ. proba")
ax.legend()
filename = f'succrate_{tuple(c for c in Ddata.keys())}_aggr.png'
plt.savefig(filename)
print(f'saved at {filename}')