from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys
from time import perf_counter
from experiments.lwe_gen import *

from sample import *

from g6k.siever import SaturationError

from preprocessing import load_lwe

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from test_hyb_att import alg_3_debug_v2 #, generateLWEInstances, se_gen, kyberGen
from global_consts import *
from copy import copy

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/" 


def run_experiment(lat_index, params, stats_dict, delta_slicer_coord=0):
    nthreads = params["nthreads"]
    n, q, dist, dist_param = params["n"], params["q"], params["dist"], params["dist_param"]
    n_guess_coord, n_slicer_coord = params["n_guess_coord"], params["n_slicer_coord"]
    beta_pre = params["beta_pre"]

    ft = "dd" #"ld" if 2*n<140 else ( "dd" if config.have_qd else "mpfr")
    FPLLL.set_precision(210)
    dim = 2*n

    print(f"float_type: {ft}")
    succ_cntr = 0
    ex_cntr = 0

    
    # A, _, _, _, bse = load_lwe(n,q,eta,k,lat_index)
    A, q, bse = load_lwe(params)

    # we don't store the whole lattice basis Binit since it is fairly large for github
    Binit = [ [int(0) for i in range(2*n)] for j in range(2*n) ]
    for i in range( n ):
        Binit[i][i] = int( q )
    for i in range(n, 2*n):
        Binit[i][i] = 1
    for i in range(n, 2*n):
        for j in range(n):
            Binit[i][j] = int( A[i-n,j] )

    then = perf_counter()
    filename_g6kdump = f'g6kdump_{n}_{q}_{dist}_{dist_param:.04f}_{lat_index}_{n_guess_coord}_{n_slicer_coord}_{beta_pre}.pkl'
    #restore precomputed g6k and initialize ittest_vect_proj(G, n_slicer_coord, n_tests=NPROJ_TESTS, eta=eta)
    g6k = Siever.restore_from_file( out_path + filename_g6kdump )
    # Needed to ensure that all locals are correct.
    # Ideally, already done.
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve

    G = g6k.M
    G.update_gso()
    # bkz_performed = False
    # LR = LatticeReduction( G.B, threads_bkz=nthreads )
    if dist=="binomial":
        distrib = centeredBinomial(dist_param)
    elif dist=="ternary":
         print(f"dist_param: {dist_param}")
         distrib = ternaryDist(dist_param)
    for delta in range(n_slicer_coord,n_slicer_coord+delta_slicer_coord+1):
        lens = test_vect_proj(G, delta, NPROJ_TESTS, distrib)
        est_norm = np.percentile(lens,50)
        print(f"#{lat_index} est_proj_norm is: {est_norm} for dim={delta}",flush=True)
        if est_norm <= HYB_PROJ_THRESHOLD:
            break

    print(f"#{lat_index} final est_proj_norm is: {est_norm} @dim={delta}")

    # - - - when we chose the slicing dimension, we are ready to go
    overhead_tsieve = time.perf_counter()
    assert n_slicer_coord <= G.d, f"Too many slicer coords: {n_slicer_coord}>{G.d}"

    # G = GSO.Mat( G.B, U=IntegerMatrix.identity(g6k.M.d,int_type="mpz"), UinvT=IntegerMatrix.identity(g6k.M.d,int_type="mpz"), float_type=ft )
    G = g6k.M
    g6k = Siever(G,param_sieve)
    print(g6k.M.d-delta)
    g6k.initialize_local(g6k.M.d-delta,g6k.M.d-delta,g6k.M.d)
    print("Running bdgl2...")
    then = time.perf_counter()
    g6k(alg="bdgl2") #alg="bdgl2"
    print(f"bdgl2 done in {time.perf_counter()-then}")

    H11 = g6k.M.B

    overhead_tsieve = time.perf_counter() - overhead_tsieve
    n_slicer_coord = delta
    print(f"n_slic_c: {n_slicer_coord}")

    # assert n_slicer_coord == g6k.r-g6k.l-1, f"No | n_slicer_coord: {n_slicer_coord} l:{g6k.l} r:{g6k.r} g6k.r-g6k.l-1: {g6k.r-g6k.l-1}"

    # Gaussian heuristic for the last sieve_dim dimensioal projective lattice of G.
    # ALL {from/to}_canonical_scaled calls must use scale_fact=gh_sub, or things go out of hand.
    gh_sub = gaussian_heuristic(G.r()[-n_slicer_coord:])

    print(f"Sieving-1 done in {perf_counter() - then}")
    # lambda1 = (b0@b0)**0.5

    print(f"r / r = {(g6k.M.r()[-n_slicer_coord] / g6k.M.r()[-1])**0.5}")
    for (b, s, e) in bse:
        ex_cntr+=1
        print(f"running exp # {ex_cntr}")
        ex_timer = perf_counter()
        assert ( all( (s@A+e)%q == b ) ), f"wrong lwe instance! {(A@s+e)%q , b}"
        print(f"len {len(Binit), len(Binit[0])}")

        answer = np.concatenate( [b-e,s] )

        print(f"Database size: {len(g6k)}")

        t = np.concatenate([b,n*[0]])
        e_ = np.concatenate([e,-s])[:-n_guess_coord]
        # project the error vector onto the last n_sieve_dim GS-vectors.
        e_ = from_canonical_scaled( G,e_,offset=n_slicer_coord,scale_fact=gh_sub )

        # print(f"prog e_: {e_}")

        #deduce the projected error norm
        dist_sq_bnd = e_@e_
        dist_bnd = dist_sq_bnd**0.5
        dist_threshold = ( G.r()[-n_slicer_coord] / gh_sub )**0.5
        print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
        print(f"dist_sq_bnd: {dist_sq_bnd}")
        print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")

        B = IntegerMatrix.from_matrix(Binit)

        tracer = {}
        # v = alg_3_debug(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg3=None)
        # iter_v = alg_3_debug_v2(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg3=tracer)
        with open("progvar","wb") as file:
            pickle.dump([n_slicer_coord,t,e,s,EPS2 * dist_sq_bnd, g6k.M.r(), gh_sub], file)
        iter_v = alg_3_debug_v2(g6k,H11,B,t,n_guess_coord, dist, dist_param, s, dist_sq_bnd=EPS2 * dist_sq_bnd, nthreads=nthreads, tracer_alg3=tracer)
        guess_cntr = 0
        sli_succ = False
        v2 = None
        for v in iter_v:
            if v is None:
                v = np.array( len(answer)*[0] )
            guess_cntr+=1

            v2 = v

            sli_succ = all(answer==v2)
            if sli_succ:
                succ_cntr+=1
                print(f"Success in experiment! @{guess_cntr} guess - - - - - - - - - - - - - - - - - - - - - - !!!")
                break
        if not sli_succ:
            print(f"Fail @{lat_index, ex_cntr}")
        print(f"v2 is none: {v2 is None}")
        fail_reason = "other" if guess_cntr<1 else "parasites"
        a0, a1 = tracer["wrong_guess_time_alg3"] , tracer["wrong_guess_time_alg2"]
        print(f"a0, a1: {a0,a1}")
        walltime, walltime_observed = tracer["wrong_guess_time_alg3"] + tracer["wrong_guess_time_alg2"], perf_counter() - ex_timer
        stats_dict[(n, lat_index, n_slicer_coord, n_guess_coord, ex_cntr)] = {
            "walltime": walltime,
            "dist_bnd": dist_bnd,
            "succ": sli_succ,
            "fail_reason": None if sli_succ else fail_reason,
            "key_num": tracer["key_num"], #number of guessed keys
            "g6k_len": len(g6k),
            "g6k_dim": g6k.r-g6k.l,
            "wrong_guess_time_alg3": tracer["wrong_guess_time_alg3"],
            "correct_guess_time_alg3": tracer["correct_guess_time_alg3"],
            "wrong_guess_time_alg2": tracer["wrong_guess_time_alg2"],
            "correct_guess_time_alg2": tracer["correct_guess_time_alg2"],
            "walltime_observed": walltime_observed,
            # "overhead_tbkz": overhead_tbkz, #no longer exists
            "overhead_tsieve": overhead_tsieve,
        }

        print(f"walltime: {walltime} | walltime_observed: {walltime_observed}")
        print(f" - - - {all(answer==v2)} - - - ")
    return stats_dict

if __name__=="__main__":
    """
    This file implements the hybrid attack on preprocessed Kyber instances.
    To generate ones, one needs to run attack_on_kyber.py (generating instances), run
    preprocessing.py (preprocess the data) and then run this file.
    The attack is relaxed -- we do not guess all the subkeys, but rather consider a single batch.
    """
    n = 144
    q = 3329
    # dist, dist_param = "ternary", 1/6.
    dist, 
    

    
    params={}
    params["nthreads"] = nthreads
    params["n"], params["dist"], params["dist_param"], params["q"] = n, dist, dist_param, q
    params["n_guess_coord"], params["n_slicer_coord"] = n_guess_coord, n_slicer_coord
    params["beta_pre"] = beta_pre

    succ_cntr = 0
    ex_cntr = 0

    output = []
    pool = Pool( processes = nworkers )
    tasks = []
    for lat_index in range(latnum):
        output.append({})
        params["seed"] = (lat_index,0)
        tasks.append( pool.apply_async(
            run_experiment, (lat_index, copy(params), output[lat_index],delta_slicer_coord)
            ) )

    stats_dict_agr = {}
    for t in tasks:
            stats_dict_agr.update(t.get())

    # print(ex_cntr, succ_cntr)
    print(stats_dict_agr)

    filename = f"tph_{n}_{dist}_{dist_param:0.4f}_{n_guess_coord}_{beta_pre}_{n_slicer_coord+delta_slicer_coord}.pkl"
    print(f"saving results to {filename}")
    with open(filename, "wb") as file:
        pickle.dump( stats_dict_agr, file )
    pool.close()