import fpylll
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL
from time import perf_counter
import numpy as np

import sys, os
import glob #for automated search in subfolders

from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever, SaturationError
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from math import sqrt, ceil, floor, log, exp
from copy import deepcopy
from random import shuffle, randrange

from global_consts import *

import pickle
# try:
#     from multiprocess import Pool  # you might need pip install multiprocess
# except ModuleNotFoundError:
#     from multiprocessing import Pool
from multiprocessing import Pool 

from LatticeReduction import LatticeReduction
from utils import * #random_on_sphere, reduce_to_fund_par_proj
from hybrid_estimator.batchCVP import batchCVPP_cost

def gen_cvpp_g6k(n,betamax=None,k=None,bits=11.705,seed=0):
    #TODO: consider if we may load an already reduced basis and extend the context
    betamax=n if betamax is None else betamax
    k = n//2 if k is None else k
    B = IntegerMatrix(n,n)
    B.randomize("qary", bits=bits, k = k)

    LR = LatticeReduction( B )
    for beta in range(5,betamax+1):
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}", flush=True)

    int_type = LR.gso.B.int_type
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    G = GSO.Mat( LR.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
    param_sieve = SieverParams()
    param_sieve['threads'] = 1
    param_sieve['db_size_base'] = (4/3.)**0.5 #(4/3.)**0.5 ~ 1.1547
    param_sieve['db_size_factor'] = 3.2 #3.2
    param_sieve['saturation_ratio'] = 0.5
    param_sieve['saturation_radius'] = 1.32

    g6k = Siever(G,param_sieve)
    g6k.initialize_local(0,0,n)
    print("Running bdgl2...")
    then=perf_counter()
    try:
        g6k(alg="bdgl2")
    except SaturationError:
        pass
    print(f"bdgl2-{n} done in {perf_counter()-then}")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")
    g6k.dump_on_disk(f"cvppg6k_n{n}_{seed}_test.pkl")

def run_exp(n,cntr,ntargets,max_slicer_interations=N_MAX_SLICER_ITERATIONS, nrand_param=5., nthreads=1):
    g6k = Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl")
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve
    
    G = g6k.M
    B = G.B

    sieve_dim = n
    gh = gaussian_heuristic(G.r())
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
    sieve_dim = n

    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
    dists = []
    radi = 32.
    batch_uq_size = len(g6k)//nrand+1
    print(f"g6k: {len(g6k)} | nrand: {nrand}")
    print( f"batch_uq_size: {batch_uq_size}" )
    expnum = ntargets//batch_uq_size+1
    print( f"expnum: {expnum}" )
    for ccntr in range( expnum ):
        print(f"{ccntr} out of {expnum} done at {cntr}")
        ts = uniform_in_ball( batch_uq_size, n, radi*G.get_r(0, 0)**0.5 )
        slicer = RandomizedSlicer(g6k)
        slicer.set_nthreads(nthreads)
        for t in ts:
            t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )
            t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim)
            slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)
        blocks = 2 # should be the same as in siever
        blocks = min(3, max(1, blocks))
        blocks = min(int(sieve_dim / 28), blocks)
        sp = SieverParams()
        N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
        buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
        buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
        buckets = max(buckets, 2**(blocks-1))

        slicer.set_proj_error_bound(0.5) #do we really want this number to be close to 0?
        slicer.set_max_slicer_interations(max_slicer_interations)

        print(f"Slicing...", flush=True)
        then = perf_counter()
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)
        print( f"Slicer done in {perf_counter()-then}" )

        iterator = slicer.itervalues_cdb_t()
        for tmp in iterator:
            out_gs_reduced = np.array( tmp )  #cdb[0]
            curnrm = (out_gs_reduced@out_gs_reduced)**0.5
            if curnrm > EPS2:
                break
            dists.append( curnrm )
    return dists

if __name__=="__main__":
    nthreads = 3
    nworkers = 2
    max_slicer_interations = N_MAX_SLICER_ITERATIONS
    nrand_param = NRAND_FACTOR
    n = 65
    ntargets = 512
    nlats = 5

    betamax, bits = 53, 11.82 #11.82

    # - - - loading
    to_be_computed = []
    g6ks = []
    load_succ = True
    for cntr in range(nlats):
        try:
            Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl")
            print(f"g6k={cntr} loaded")
        except FileNotFoundError:
            load_succ = False
            to_be_computed.append( (cntr,n,betamax,None,bits) )
            print(f"g6k={cntr} is yet to be processed")

    tasks = []
    pool = Pool( processes = nworkers )
    for cntr,n,betamax,k,bits in to_be_computed:
        tasks.append( pool.apply_async(
            gen_cvpp_g6k, (n, betamax, k, bits, cntr)
            ) )

    start_writing_index = len(g6ks)
    print(f"start_writing_index: {start_writing_index}")
    for t in tasks:
         t.get()

    pool.close()
    # - - - end loading
    
    pool = Pool( processes = nworkers )
    for cntr in range(nlats):
        tasks.append( pool.apply_async(
            run_exp, (n,cntr,ntargets,max_slicer_interations, nrand_param, nthreads)
            ) )
    dists = []
    for t in tasks:
        dists+=t.get()
    pool.close()
    
    with open( f"parasites_{n}.pkl", "wb" ) as file:
        pickle.dump(dists,file)
    print(dists)
    # run_exp(n,cntr,ntargets,max_slicer_interations=N_MAX_SLICER_ITERATIONS, nrand_param=5., nthreads=1)