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
from multiprocessing import Pool 

from lattice_reduction import LatticeReduction
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

def run_exp(cntr,params):
    n = params["n"]
    ntests=params["ntests"]
    approx_facts=params["approx_facts"]
    max_slicer_interations=params["max_slicer_interations"]
    nthreads=params["nthreads"]
    nrand_params=params["nrand_params"]

    saturation_scalar = SATURATION_SCALAR
    g6k = Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl")
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve

    G = g6k.M
    B = G.B

    sieve_dim = n
    gh = gaussian_heuristic(G.r())
    lambda1 = min( [G.get_r(0, 0)**0.5, gh**0.5] )

    aggregated_data = []
    #retrieve the projective sublattice
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
    for nrand_param in nrand_params:
        D = {}
        Ds = []
        for approx_fact in approx_facts:
            nsucc_slic, nsucc_bab = 0, 0
            nsucc_slic_apprcvp, n_tot_tar = 0, 0
            nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
            nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
            batch_size = ceil(len(g6k)/nrand)
            for tstnum in range(ntests):
                print(f" - - - {approx_fact} #{tstnum} out of {ntests} - - - nrand: {nrand_param} - - - bs: {batch_size}", flush=True)
                cebt_list = []
                for _ in range(batch_size):
                    c = [ randrange(-2,3) for j in range(n) ]
                    e = np.array( random_on_sphere(n,approx_fact*lambda1) )
                    b = np.array( B.multiply_left( c ) )
                    t = b+e
                    n_tot_tar += 1
                    cebt_list.append( [c,e,b,t] )

                """
                Testing Babai.
                """
                cebt_list_filtered = []
                for c,e,b,t in cebt_list:
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
                        nsucc_bab += 1/len(cebt_list) #division since we have proba rather than True/False
                        nsucc_slic += 1/len(cebt_list)
                        nsucc_slic_apprcvp += 1
                    if not succ_bab:
                        cebt_list_filtered.append( (c,e,b,t) )

                """
                Testing Slicer.
                """
                blocks = 2 # should be the same as in siever
                blocks = min(3, max(1, blocks))
                blocks = min(int(sieve_dim / 28), blocks)
                sp = SieverParams()
                N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
                buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
                buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
                buckets = max(buckets, 2**(blocks-1))
                if not len(cebt_list_filtered)==0:
                    sieve_dim = n
                    #retrieve the projective sublattice
                    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
                    slicer = RandomizedSlicer(g6k)
                    slicer.set_nthreads(nthreads)
                    t_gs_reduced_list = []
                    dist_sq_bnd_max = 0
                    for c,e,b,t in cebt_list_filtered:
                        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )
                        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh) )
                        gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
                        # print("projected target squared length:", (e_@e_))
                        dist_sq_bnd_max = max( dist_sq_bnd_max, (e_@e_) )

                        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )
                        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
                        t_gs_reduced_list.append(t_gs_reduced)
                        # t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer
                        slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

                    slicer.set_proj_error_bound(EPS2*dist_sq_bnd_max)
                    slicer.set_max_slicer_interations(max_slicer_interations)
                    slicer.set_Nt(len(cebt_list_filtered))
                    slicer.set_saturation_scalar(saturation_scalar)
                    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], True)

                    iterator = slicer.itervalues_cdb_t()
                    succ = False
                    attemptcntr = 0
                    succcntr = 0
                    print(f"indexes: ",end="")
                    for tmp, index in iterator:
                        out_gs_reduced = np.array( tmp )  #cdb[0]
                        if (out_gs_reduced@out_gs_reduced)>EPS2*dist_sq_bnd_max:
                            break
                        attemptcntr += 1
                        
                        c, e, b, t = cebt_list_filtered[index]
                        out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
                        bab_01 = np.array( G.babai( np.array(t)-out ) )

                        # - - - Check - - - -

                        succ = all(c==bab_01)
                        if succ:
                            succcntr+=1
                            if succcntr >= len(cebt_list_filtered):
                                print(f"All successful!")
                                break
                    print(f"Slic Succsess: {succcntr}/{len(cebt_list_filtered)} after {attemptcntr} attempts")
                    if not ( succ ):
                        print(f"FAIL after slicer: {(err@err)}")
                    else:
                        nsucc_slic += float( succcntr / len(cebt_list) ) #division since we have proba rather than True/False


            D[(n,approx_fact)] = (0, 1.0*nsucc_slic / ntests, 1.0*nsucc_bab / ntests, 1.0*nsucc_slic_apprcvp / ntests)
            Ds.append(D)
            print( f"Experiments for nrand_param={nrand_param} done..." )
        aggregated_data.append([nrand_param, Ds]) 
    return aggregated_data

if __name__=="__main__":
    nthreads = 5
    nworkers = 2
    max_slicer_interations = 300
    ntests = 30 #200
    nlats = 2 #10
    n = 45
    bits = 11.705
    betamax = 43
    approx_facts = [ 0.9 + 0.02*i for i in range(6) ]
    nrand_params = [ 1.0,5.0,10.0 ]
    print(approx_facts)
    
    params ={
        "nthreads": nthreads,
        "max_slicer_interations": max_slicer_interations,
        "ntests": ntests,
        "nlats": nlats,
        "n": n,
        "bits": bits,
        "approx_facts": approx_facts,
        "nrand_params": nrand_params,
    }

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

    # for cntr in range(start_writing_index,nlats):
    #     g6ks.append( Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl") )

    pool.close()
    aggregated_data = []

    tasks = []
    output = []
    pool = Pool( processes = nworkers )
    for cntr in range(nlats):
        tasks.append( pool.apply_async(
            run_exp, (cntr,params)
            ) )
        print(cntr)

    for t in tasks:
        aggregated_data += [ t.get() ]
    pool.close()

    for tmp in aggregated_data:
        print(f"nrand_parameter: {aggregated_data[0]}")
        print(aggregated_data[1])
        # print(f"poisoned: {poison_dbt}")

    filename = f"slicsucc_{n}" + ".pkl"
    with open(filename,"wb") as file:
        pickle.dump(aggregated_data, file)
    print( f"saved in {filename}" )