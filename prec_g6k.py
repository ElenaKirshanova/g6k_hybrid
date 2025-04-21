from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL, config
import pickle, time, sys
import numpy as np
from LatticeReduction import LatticeReduction

from fpylll.algorithms.bkz2 import BKZReduction

filename = "g6kdump_170_3329_binomial_3.0000_9_3_92_91.pkl"  #allegedly, a bad lattice
with open(filename,"rb") as file:
    D = pickle.load( file )

B = D['B']
B = IntegerMatrix.from_matrix( B )
G = GSO.Mat(B, float_type="dd")
G.update_gso()

lpre = [ G.r() ]

""""""
then = time.perf_counter()
lll = LLL.Reduction( G )
lll()
print(f"LLL done in: {time.perf_counter()-then}")

bkzobg = BKZReduction(G)
flags = BKZ.AUTO_ABORT|BKZ.MAX_LOOPS|BKZ.GH_BND

tours = 5
for beta in range(30,53,1):
    par = BKZ_FPYLLL.Param(
            beta,
            strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
            max_loops=tours,
            flags=flags
          )
    then = time.perf_counter()
    bkzobg(par)
    print(f"BKZ-{beta} done in: {time.perf_counter()-then}")
B = G.B

""""""

LR = LatticeReduction( B, threads_bkz=10 )
print(f"Starting bkz...", flush=True)

bkz_start = time.perf_counter()
for beta in range(70,72):
    then_round=time.perf_counter()
    LR.BKZ(beta, tours=2)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")
    sys.stdout.flush()

G = GSO.Mat( LR.basis, float_type="dd" )
G.update_gso()
lpost = [ G.r() ]

with open("out170.pkl","wb") as file:
    pickle.dump([lpre,lpost,G.B], file)