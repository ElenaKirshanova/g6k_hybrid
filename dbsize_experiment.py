from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic

from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import save_folder, random_on_sphere, uniform_in_ball, from_canonical_scaled, to_canonical_scaled, reduce_to_fund_par_proj #*
import numpy as np
import argparse
import sys, os
from hybrid_estimator.batchCVP import batchCVPP_cost
from random import shuffle, randrange
import numpy as np
from math import sqrt, ceil, floor, log, exp
import time
import pickle

from LatticeReduction import LatticeReduction

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import sys, os

def run_exp(lat_id, n, betamax, sieve_dim, shrink_factor, n_shrinkings, Nexperiments, nthreads, succ_criterion_factor, nrand_param=1.):

    slack = 1.03
    ft = "ld" if n<50 else ( "dd" if config.have_qd else "mpfr")
    print(f"launching n, betamax, sieve_dim = {n, betamax, sieve_dim}")

    slicer_suc = [0]*n_shrinkings
    slicer_fail = [0]*n_shrinkings
    babai_suc = [0]*n_shrinkings
    # - - - try load a lattice - - -
    filename = f"saved_lattices/bdgl2_n{n}_b{sieve_dim}_{lat_id}.pkl"
    nothing_to_load = True
    try:
        g6k = Siever.restore_from_file(filename)
        G = g6k.M
        nothing_to_load = False
        print(f"Load succeeded...")
    except Exception as excpt:
        print(excpt)
        pass
    # - - - end try load a lattice - - -

    # - - - Make all fpylll objects - - -
    if nothing_to_load:
        print(f"Nothing to load. Computing")
        B = IntegerMatrix(n,n)
        B.randomize("qary", k=n//2, bits=11.705)
        G = GSO.Mat(B, float_type=ft)
        G.update_gso()

        if sieve_dim<30: print("Slicer is not implemented on dim < 30")
        if sieve_dim<40: print("LSH won't work on dim < 40")

        lll = LLL.Reduction(G)
        lll()

        bkz = LatticeReduction(B,threads_bkz=nthreads)
        for beta in range(5,betamax+1):
            then_round=time.perf_counter()
            bkz.BKZ(beta,tours=5)
            round_time = time.perf_counter()-then_round
            print(f"BKZ-{beta} done in {round_time}", flush=True)
            sys.stdout.flush()

        int_type = bkz.gso.B.int_type
        G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
        G.update_gso()
        lll = LLL.Reduction( G )
        lll()
    # - - - end Make all fpylll objects - - -
    # make Siver object
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()
    gh = min( gaussian_heuristic(G.r())**0.5, G.r()[0]**0.5 )
    if nothing_to_load:
        g6k.dump_on_disk(filename)

    print("db_dize:", g6k.db_size())

    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    dbsize_start = g6k.db_size()
    nrand_, _ = batchCVPP_cost(sieve_dim,100,dbsize_start**(1./sieve_dim),1) #100 can be any constant >1
    print("nrand:", (1./nrand_)**sieve_dim)

    cs = []
    es = []
    bs = []
    for i in range(Nexperiments):
        c = [ randrange(-10,10) for k in range(n) ]
        e = np.array( random_on_sphere(n, 0.95 * gh) ) #error vector
        # e = uniform_in_ball( 1, n, 0.5 * gh )[0]
        b = G.B.multiply_left( c )
        cs.append( c )
        es.append( e )
        bs.append( b )

    for j in range(n_shrinkings):
        print("Running experiment ", j, "out of ", n_shrinkings)

        for i in range(Nexperiments):
            if i%10 == 0:
                print(f"{i} out of {Nexperiments} done...", flush=True)
            c = cs[i] #[ randrange(-10,10) for k in range(n) ]
            e = es[i] #np.array( random_on_sphere(n, 0.95 * gh) ) #error vector
            print(f"gauss: {gh} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")
            e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )

            b = bs[i] #G.B.multiply_left( c )
            b_ = np.array(b,dtype=np.int64)
            t_ = e+b_
            t = [ int(tt) for tt in t_ ]

            #project onto the last projective lattice and babai reduce
            t_gs = from_canonical_scaled( G,t,offset=sieve_dim )
            t_gs_non_scaled = G.from_canonical(t)[-sieve_dim:]
            shift_babai_c = G.babai((n-sieve_dim)*[0] + list(t_gs_non_scaled), start=n-sieve_dim,gso=True)
            shift_babai = G.B.multiply_left( (n-sieve_dim)*[0] + list( shift_babai_c ) )
            t_gs_reduced = from_canonical_scaled( G,np.array(t)-shift_babai,offset=sieve_dim ) #this is the actual reduced target
            t_gs_shift = from_canonical_scaled( G,shift_babai,offset=sieve_dim )

            print("projected reduced target squared length:", (t_gs_reduced@t_gs_reduced))
            print("projected error squared length:", (e_@e_))


            # - - - Babai check - - -
            out = to_canonical_scaled( G,t_gs_reduced,offset=sieve_dim )
            N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
            N.update_gso()
            bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
            succ = all( np.array( c[G.d-sieve_dim:] )==bab_1 )
            print(f"Babai Success: {succ}", flush=True)

            if succ:
                babai_suc[j]+=1

            if not succ:
                #need to define it here since old targets and their rerandomizations
                #would remain to be in db_t
                slicer = RandomizedSlicer(g6k)
                slicer.set_nthreads(nthreads);
                n_per_target = ceil( nrand_param*(1./nrand_)**sieve_dim ) #10.8 for dim=55?
                print(f"Forcing nrerand = {n_per_target}")
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=n_per_target)
                try:
                    slicer.set_proj_error_bound(1.01*(e_@e_))
                    # slicer.set_lifted_error_bound(8.01*(e_@e_))
                    slicer.set_max_slicer_interations(100)
                    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"])

                    # iterator = slicer.itervalues_cdb_t()
                    # for tmp in iterator:
                    #     out_gs_reduced = np.array( tmp )  #cdb[0]
                    #     break
                    # out_gs = out_gs_reduced + t_gs_shift

                    # # - - - Check - - - -
                    # out = to_canonical_scaled( G,out_gs,offset=sieve_dim )
                    # bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s

                    # bab_01 =  np.array( bab_1 ) #shifted answer. Good since it is smaller, thus less rounding error
                    # bab_01 += np.array(shift_babai_c)

                    # TODO: fix empty iterator bug (done?)
                    iterator2 = slicer.itervalues_db_lifted()
                    res_lifted = np.array(sieve_dim*[0])
                    for tmp in iterator2:
                        res_lifted = np.array(tmp)
                        print(res_lifted)
                        break

                    bab_01 = np.round(to_canonical_scaled( G, res_lifted ))
                    bab_01 = np.array( G.babai( t-bab_01 ) )

                    if (all(c==bab_01)):
                        print(f"SUCCESS")
                        succeeded = True
                    else:
                        #slicer_fail[j] += 1
                        succeeded = False
                        print(f"FAIL")
                    if succeeded:
                        slicer_suc[j] += 1
                    else:
                        slicer_fail[j] += 1

                except Exception as e:
                    print(f" - - - {e} - - -")
                    raise e

        g6k.shrink_db(shrink_factor*g6k.db_size())

    print(f"Lattice-{lat_id} processed...")
    print(babai_suc)
    print(slicer_suc)
    print(slicer_fail)

    density_plot = []
    cntr = 0
    s = 1
    for j in range(n_shrinkings):
        density_plot.append( (s,slicer_suc[cntr]+babai_suc[cntr]) )
        cntr+=1
        s *= shrink_factor
    return density_plot


if __name__ == '__main__':

    Nexperiments = 20
    Nlats = 12
    path = "saved_lattices/"
    isExist = os.path.exists(path)
    if not isExist:
        try:
            os.makedirs(path)
        except:
            pass


    FPLLL.set_precision(200)

    n, betamax, sieve_dim = 60, 45, 60 #also 70, 25, 70 and 80, 25, 80

    nthreads = 3 # number of workers
    slicer_threads = 1 # threads the slicer will use
    nrand_param = 5.5
    shrink_factor = 0.7071 # ~ 1/sqrt(2)
    n_shrinkings = 9
    succ_criterion_factor = 1.0 #0 for uSVP check and >0 for approx_fact check
    pool = Pool(processes = nthreads )
    tasks = []

    density_plots = []
    for lat_id in range(Nlats):
        tasks.append( pool.apply_async(
            run_exp, (lat_id, n, betamax, sieve_dim, shrink_factor, n_shrinkings, Nexperiments, slicer_threads, succ_criterion_factor, nrand_param)
        ) )

    for t in tasks:
        density_plots.append( t.get() )


    with open(f"dbsize_{n}_exp.pkl", "wb") as file:
        pickle.dump( density_plots, file )

    print(density_plots)
    sys.stdout.flush()
