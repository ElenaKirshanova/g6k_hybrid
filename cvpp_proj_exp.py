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
    g6k.dump_on_disk(f"cvppg6k_n{n}_d{n_slicer_coord}_{seed}_test.pkl")

def run_exp(g6k,ntests,approx_facts,max_slicer_interations=100, nthreads=1, nrand_params=[1.]):
    G = g6k.M
    B = G.B
    n = G.d
    n_slicer_coord = g6k.r-g6k.l

    sieve_dim = n
    gh = gaussian_heuristic(G.r())
    lambda1 = min( [G.get_r(0, 0)**0.5, gh**0.5] )
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k = Siever(G,param_sieve) #temporary solution
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()

    gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
    rii_sqrt = [ sqrt(rii/gh_sub) for rii in G.r()[:n-n_slicer_coord] ] 

    aggregated_data = []
    for nrand_param in nrand_params:
        D = {}
        Ds = []
        for approx_fact in approx_facts:
            nsucc_slic, nsucc_bab = 0, 0
            for tstnum in range(ntests):
                print(f" - - - {approx_fact} #{tstnum} out of {ntests} - - - nrand: {nrand_param}", flush=True)
                c = [ randrange(-2,3) for j in range(n) ]

                # e = np.array( random_on_sphere(n,approx_fact*lambda1) )
                # e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh) )
                e_ = np.array( random_on_sphere(n_slicer_coord,approx_fact) ) #the projected part of the error 
                e0_ = np.array( [np.random.uniform(0, -rii/2.01, rii/2.01) for rii in rii_sqrt] ) #the projected part of the error
                e_proj = np.concatenate( [e0_,e_] )
                e = np.array( to_canonical_scaled(G,e_proj,offset=sieve_dim,scale_fact=gh_sub) )
                print(f"e_proj: {e_proj}")
                print(f"e: {e}")

                b = np.array( B.multiply_left( c ) )
                t = b+e

                """
                Testing Babai.
                """
                then = perf_counter()
                ctmp = G.babai( t )
                tmp = B.multiply_left( ctmp )
                print(f"Babai-{n} done in {perf_counter()-then}")
                err = tmp-b
                succ_bab = (err@err)<10**-6
                if not ( succ_bab ):
                    print(f"FAIL after babai: {(err@err)}")
                else:
                    print(f"SUCCSESS after babai!")
                    nsucc_bab += 1
                    nsucc_slic += 1

                """
                Testing Slicer.
                """
                if not succ_bab:
                    sieve_dim = n
                    t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )

                    #retrieve the projective sublattice
                    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]

                    try:
                        # e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh) )
                        gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
                        print("projected target squared length:", (e_@e_))

                        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )
                        #retrieve the projective sublattice
                        B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
                        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
                        t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

                        slicer = RandomizedSlicer(g6k)
                        slicer.set_nthreads(nthreads);

                        nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
                        nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
                        slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

                        blocks = 2 # should be the same as in siever
                        blocks = min(3, max(1, blocks))
                        blocks = min(int(sieve_dim / 28), blocks)
                        sp = SieverParams()
                        N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
                        buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
                        buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
                        buckets = max(buckets, 2**(blocks-1))

                        slicer.set_max_slicer_interations(max_slicer_interations)
                        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)

                        iterator = slicer.itervalues_cdb_t()
                        for tmp in iterator:
                            out_gs_reduced = tmp  #cdb[0]
                            break
                        out_gs = out_gs_reduced + t_gs_shift

                        # - - - Check - - - -
                        out = to_canonical_scaled( G,out_gs,offset=sieve_dim,scale_fact=gh_sub )

                        projerr = G.to_canonical( G.from_canonical(e,start=n-sieve_dim), start=n-sieve_dim)
                        out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
                        bab_01 = np.array( G.babai( np.array(t)-out ) )

                        succ = all(c==bab_01)
                        print(f"Slic Succsess: {succ}")
                        if not ( succ ):
                            print(f"FAIL after slicer: {(err@err)}")
                        else:
                            nsucc_slic += 1
                        # del slicer
                    except Exception as excpt: #if slicer fails for some reason,
                        #then prey, this is not a devastating segfault
                        print(excpt)
                        raise excpt

            D[(n,approx_fact)] = (0, 1.0*nsucc_slic / ntests, 1.0*nsucc_bab / ntests)
            Ds.append(D)
            print( f"Experiments for nrand_param={nrand_param} done..." )
        aggregated_data.append([nrand_param, Ds]) 
    return aggregated_data

if __name__=="__main__":
    nthreads = 1
    nworkers = 2
    max_slicer_interations = 300
    ntests = 10
    nlats = 2

    n = 360
    n_slicer_coord = 60
    betamax = 60

    bits = 11.705
    approx_facts = [ 0.4 + 0.05*i for i in range(15) ] #
    print(approx_facts)

    to_be_computed = []
    g6ks = []
    load_succ = True
    for cntr in range(nlats):
        try:
            g6ks.append( Siever.restore_from_file(f"cvppg6k_n{n}_d{n_slicer_coord}_{cntr}_test.pkl") )
            print(f"g6k={cntr} loaded")
        except FileNotFoundError:
            load_succ = False
            to_be_computed.append( (cntr,n,betamax,None,bits) )
            print(f"g6k={cntr} is yet to be processed")

    tasks = []
    output = []
    pool = Pool( processes = nworkers )
    for cntr,n,betamax,k,bits in to_be_computed:
        tasks.append( pool.apply_async(
            gen_cvpp_g6k, (n, betamax, k, bits, cntr)
            ) )

    start_writing_index = len(g6ks)
    print(f"start_writing_index: {start_writing_index}")
    for t in tasks:
         t.get()

    for cntr in range(start_writing_index,nlats):
        g6ks.append( Siever.restore_from_file(f"cvppg6k_n{n}_d{n_slicer_coord}_{cntr}_test.pkl") )

    pool.close()
    aggregated_data = []
    nrand_params = [1., 3., 5.]

    for g6k in g6ks:
        aggregated_data += [ run_exp(g6k,ntests,approx_facts,max_slicer_interations=max_slicer_interations, nthreads=nthreads, nrand_params=nrand_params) ]

    for tmp in aggregated_data:
        print(f"nrand_parameter: {aggregated_data[0]}")
        print(aggregated_data[1])

    with open(f"slicsucc_{n}_{n_slicer_coord}.pkl","wb") as file:
        pickle.dump(aggregated_data, file)