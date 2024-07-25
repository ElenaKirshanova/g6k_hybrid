import sys
import numpy as np
from lwe_gen import generateLWEInstance
from LatticeReduction import LatticeReduction
from fpylll import IntegerMatrix, GSO
from math import sqrt
import time

import warnings
warnings.filterwarnings("ignore", message="Dimension of lattice is larger than maximum supported")

entropy = 2.3
eta = 3

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

from multiprocessing import Pool

def experiment(n):

    A,b,q,s,e = generateLWEInstance(n)

    w = 1/q * (b - s.dot(A) - e)
    w = w.astype(int)
    
    n,m = A.shape

    d = n+m

    B = np.identity(d).astype(int)
    for i in range(m):
        B[i,i] *= int(q)

    for i in range(n):
        for j in range(m):
            B[m+i,j] =  int(A[i,j])

    x = np.concatenate( [w,s] )
    v = x.dot(B)
    e = np.concatenate( [e,-s] )
    t = x.dot(B) + e
    
    assert ( t == np.concatenate( [b,[0]*n] ) ).all()

    bkz = LatticeReduction(B)

    beta = 1

    print("beta: ", end="")

    while beta < 66:

        beta += 1

        print(beta, end=", ")

        bkz.BKZ(beta)

        
        M = bkz.gso
        _ = M.update_gso()

        t_ = M.from_canonical(t)
        t_1 = t_[:d-beta]
        t_2 = t_[d-beta:]

        #Simulate CVP oracle.

        numericalProblems = False

        e_ =  M.from_canonical(e)
        e_scaled = list(e_)

        #At this point, e_scaled contains the coefficients of e
        #with respect to the GS basis b_0^*, ..., b_{n-1}^*.
        #We compute the coefficients
        #with respect to the normalized GS basis.

        for i in range(d):
            try:
                e_scaled[i] *= sqrt( M.get_r(i,i) )
            except ValueError:
                numericalProblems = True

        if numericalProblems or 1/4 * M.get_r(d-beta,d-beta) <= sum( e_i**2 for e_i in e_scaled[d-beta:] ):
            continue
        
        v2_ = M.from_canonical(v)[d-beta:]
        print("")

        #End of simulation

        #Recover x2 inductively via the identity x2_{beta-i} + \sum_{j=1}^i  R_{d-j,d-i} = v2_{beta-i}.
        x2 = [0]*beta
        for i in range(1,beta+1):
            x2[beta-i] = v2_[beta-i] - 1/ M.get_r(d-i, d-i) * sum( [ x2[beta-j] * M.get_r(d-j, d-i) for j in range(1,i) ] ) 
            x2[beta-i] = int( round( x2[beta-i] ) )
        x2 = tuple(x2)

        Uinv = M.UinvT.transpose()
        U = M.U
        
        print(n)
        print(x2)
        print( Uinv.multiply_left(x)[d-beta:] )
        print("")


        return (n,beta)
        
    #     i = 0
    #     smallEnough = True
    #     while smallEnough and i < d-beta:
    #         if abs(e_project[i]) >= 1/2 * sqrt( M.get_r(i,i) ):
    #             smallEnough = False
    #         i += 1
        
    #     if smallEnough:
    #         print(n,beta)
    #         return (n,beta)

for n in range(10,120,10):
    experiment(n)  

# inputs = []

# trials = 10

# for _ in range(trials):
#     for n in range(80,140,10):
#         inputs.append(n)

# with Pool() as p:
#     results = p.map(experiment, inputs)
    
#     table = {}

#     for n,beta in results:
#         if n not in table:
#             table[n] = {}
        
#         if beta not in table[n]:
#             table[n][beta] = 1
#         else:
#             table[n][beta] += 1

#     for n in table:
#         print(n)

#         avg = 0

#         for beta in table[n]:
#             ctr = table[n][beta]
#             print("\t%d: %d" % (beta,ctr))

#             avg += beta*ctr
        
#         print( "\tavg: %f" % (avg/trials) )