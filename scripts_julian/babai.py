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
    firstPoint = 0

    while beta < 66:

        bkz.BKZ(beta)

        beta += 1

        M = bkz.gso
        M.update_gso()

        lift_fix = M.babai(t, 0, d)
        lift_can = (M.B).multiply_left(lift_fix)
        v = np.array(t) - np.array(lift_can)

        if (v==e).all():
            print(n, "\t", beta)
            return
    
    print( "n = %d\tFail" % n )

trials = 10

inputs = []
for _ in range(trials):
    for n in range(80,140,10):
        inputs.append(n)

with Pool() as p:
    p.map(experiment, inputs)