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
        # print(f"len etilde2: {len(etilde2)}")
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

def alg_2_batched( g6k,target_candidates, dist_sq_bnd=1.0, nthreads=N_SIEVE_THREADS, tracer_alg2=None ): #this works
    if not tracer_alg2 is None:
        startt = time.perf_counter()
        tracer_alg2["walltime"] = 0
    sieve_dim = g6k.r-g6k.l #n_slicer_coord
    print(f"in alg2 sieve_dim={sieve_dim}", flush=True)

    G = g6k.M
    gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
    B = G.B
    dim = G.d
    Gsub = GSO.Mat( G.B[:dim-sieve_dim], float_type=G.float_type )
    Gsub.update_gso()

    # - - - prepare Slicer for batch cvp - - -
    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(nthreads)
    slicer.set_max_slicer_interations(N_MAX_SLICER_ITERATIONS)
    slicer.set_proj_error_bound( (EPS2*(dist_sq_bnd)) )
    slicer.set_Nt(len(target_candidates))
    slicer.set_saturation_scalar(SATURATION_SCALAR)
    # - - - END prepare Slicer for batch cvp - - -

    nrand_, _ = batchCVPP_cost(sieve_dim,1,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(NRAND_FACTOR*(1./nrand_)**sieve_dim) #min( 250, target_list_size / len(target_candidates ) )

    print(f"len(target_candidates): {len(target_candidates)} nrand: {nrand}")
    t_gs_list = []
    t_gs_reduced_list = []
    shift_babai_c_list = []
    for target in target_candidates:
        t_gs = from_canonical_scaled( G,target,offset=sieve_dim, scale_fact=gh_sub )

        t_gs_non_scaled = G.from_canonical(target)[dim-sieve_dim:]
        shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, gso=True) )
        shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
        t_gs_reduced = from_canonical_scaled( G,np.array(target, dtype=DTYPE)-shift_babai,offset=sieve_dim,scale_fact=gh_sub ) #this is the actual reduced target

        t_gs_list.append(t_gs)
        shift_babai_c_list.append(shift_babai_c)
        t_gs_reduced_list.append(t_gs_reduced)
        slicer.grow_db_with_target(t_gs_reduced, n_per_target=nrand) #add a candidate to the Slicer

    print(f"running slicer")
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)

    print(f"t_gs_reduced norm: {t_gs_reduced@t_gs_reduced}")
    iterator = slicer.itervalues_cdb_t(return_with_index=True)
    best_bab_01 = np.array( g6k.M.d*[0] )
    attemptcntr = 0
    for tmp, index in iterator:
        out_gs_reduced = np.array(tmp, dtype=DTYPE)  #db_t[0] is expected to contain the error vector
        if (out_gs_reduced@out_gs_reduced) > 1.00001*dist_sq_bnd:
            break
        attemptcntr += 1
        print(f"out_gs_reduced norm: {(out_gs_reduced@out_gs_reduced)**0.5} vs {dist_sq_bnd**0.5}")

        #Now we deduce which target candidate the error vector corresponds to.
        #The idea is that if t_gs is an answer then t_gs_reduced - out_gs_reduced is in the projective lat
        #and is (close to) zero.
        min_norm_err_sq = float("inf")
        out_reduced = np.array( to_canonical_scaled( G, out_gs_reduced, offset=sieve_dim, scale_fact=gh_sub ), dtype=DTYPE )
        # the line below projects the error away from first basis vectors
        out_reduced = G.to_canonical( (G.d-sieve_dim)*[0] + list( G.from_canonical( out_reduced,start=G.d-sieve_dim ) ), start=0 )

        assert not (index is None), f"Impossible!"
        t = np.array( target_candidates[index], dtype=DTYPE )
        bab_01 = np.array( G.babai(t-out_reduced) )
        solution_candidate = np.array( G.B.multiply_left( bab_01 ), dtype=DTYPE )
        diff = t - solution_candidate
        diff_nrm_sq = diff@diff

        if diff_nrm_sq <= min_norm_err_sq:
            min_norm_err_sq = diff_nrm_sq
            best_bab_01 = bab_01
            if not tracer_alg2 is None:
                tracer_alg2["walltime"] = time.perf_counter() - startt
                tracer_alg2["len(target_candidates)"] = len(target_candidates)
                tracer_alg2["nrand"] = nrand
            yield best_bab_01


        print(f"min_norm_err_sq: {min_norm_err_sq}")


    print(f"alg2 terminates after {attemptcntr} searches")


    return best_bab_01


