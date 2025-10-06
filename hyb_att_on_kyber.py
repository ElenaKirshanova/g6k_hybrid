from experiments.lwe_gen import *

import sys,os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *

from global_consts import *

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from sample import *

from preprocessing import load_lwe
from hybrid_estimator.batchCVP import batchCVPP_cost

approx_fact = 1.0001

max_nsampl = 2**31-1
inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"



def alg_3(g6k,B,H11,t,n_guess_coord, eta, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):

    # - - - prepare targets - - -
    then_start = perf_counter()
    dim = B.nrows
    print(f"dim: {dim}")

    t1, t2 = t[:-n_guess_coord], t[-n_guess_coord:]
    slicer = RandomizedSlicer(g6k)
    distrib = centeredBinomial(eta)
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = [t1] #first target is always the original one
    vtilde2s = [np.array(t2) ]

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    for times in range(nsampl): #Alg 3 steps 4-7
        if times!=0 and times%64 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        etilde2 = np.array( distrib.sample( n_guess_coord ), dtype=DTYPE ) #= (0 | e2)
        vtilde2 = np.array(t2, dtype=DTYPE)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = H12.multiply_left(vtilde2)

        t1_ = np.array( list(t1), dtype=DTYPE ) - tmp
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

    v1 = np.array( g6k.M.B[:len(ctilde1)].multiply_left( ctilde1 ) )
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:

        tmp = H12.multiply_left(vtilde2)
        v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
        v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
        v_t = v - np.array(t)
        vv = v_t@v_t
        if vv < minv:
            minv = vv
            argminv = v
        cntr += 1
    return argminv




