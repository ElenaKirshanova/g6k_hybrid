from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
import argparse
from utils import *
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
    if dist=="binomial":
        dist_param = int(dist_param)
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

        #deduce the projected error norm
        dist_sq_bnd = e_@e_
        dist_bnd = dist_sq_bnd**0.5
        dist_threshold = ( G.r()[-n_slicer_coord] / gh_sub )**0.5
        print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
        print(f"dist_sq_bnd: {dist_sq_bnd}")
        print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")

        B = IntegerMatrix.from_matrix(Binit)

        tracer = {}
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

def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocessing for hybrid attack."
    )
    parser.add_argument(
    "--nthreads", default=N_SIEVE_THREADS, type=int, help="Threads per slicer."
    )
    parser.add_argument(
    "--nworkers", default=1, type=int, help="Workers for experiments."
    )
    parser.add_argument(
    "--lats_per_dim", default=1, type=int, help="Number of lattices."
    )
    parser.add_argument(
    "--n", default=125, type=int, help="LWE dimension"
    )
    parser.add_argument(
    "--q", default=3329, type=int, help="LWE modulus"
    )
    parser.add_argument(
    "--dist", default="binomial", type=str, help="LWE distribution"
    )
    parser.add_argument(
    "--dist_param", default=2.0, type=float, help="LWE distribution's parameter (as float)"
    )
    parser.add_argument(
    "--beta_pre", default=46, type=int, help="BKZ blocksize."
    )
    parser.add_argument(
    "--n_guess_coord", default=2, type=int, help="Number of guessing coordinates"
    )
    parser.add_argument(
    "--n_slicer_coord", default=47, type=int, help="Minimal dimension of slicer."
    )
    parser.add_argument(
    "--delta_slicer_coord", default=3, type=int, help="Maximal dimension of slicer will be n_slicer_coord+delta_slicer_coord."
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

if __name__=="__main__":
    """
    This file implements the hybrid attack on preprocessed Kyber instances.
    To generate ones, one needs to run attack_on_kyber.py (generating instances), run
    preprocessing.py (preprocess the data) and then run this file.
    The attack is relaxed -- we do not guess all the subkeys, but rather consider a single batch.
    """
    parser = get_parser()
    args = parser.parse_args()

    n = args.n
    q = args.q
    dist, dist_param = args.dist, args.dist_param
    latnum = args.lats_per_dim
    n_guess_coord, n_slicer_coord = args.n_guess_coord, args.n_slicer_coord
    beta_pre = args.beta_pre
    delta_slicer_coord = args.delta_slicer_coord #integer >=0, n_slicer_coord + delta_slicer_coord is the cap on slicer dimension
    nthreads = args.nthreads
    nworkers = args.nworkers

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

    print(stats_dict_agr)

    filename = f"tph_{n}_{dist}_{dist_param:0.4f}_{n_guess_coord}_{beta_pre}_{n_slicer_coord+delta_slicer_coord}.pkl"
    print(f"saving results to {filename}")
    with open(filename, "wb") as file:
        pickle.dump( stats_dict_agr, file )
    pool.close()