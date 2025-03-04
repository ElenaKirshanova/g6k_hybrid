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

from LatticeReduction import LatticeReduction
from test_hyb_att import alg_3_debug, alg_3_debug_v2 #, generateLWEInstances, se_gen, kyberGen
from global_consts import *

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"
max_nsampl = 2**10


def run_experiment(lat_index, params, stats_dict, bkz_beta_range=None, delta_slicer_coord=0):
    nthreads = params["nthreads"]
    n, k, q, eta = params["n"], params["k"], params["q"], params["eta"]
    n_guess_coord, n_slicer_coord = params["n_guess_coord"], params["n_slicer_coord"]

    ft = "ld" if 2*k*n<140 else ( "dd" if config.have_qd else "mpfr")
    FPLLL.set_precision(210)
    dim = 2*k*n

    print(f"float_type: {ft}")
    succ_cntr = 0
    ex_cntr = 0

    filename_g6kdump = f"g6kdump_{n}_{q}_{eta}_{k}_{lat_index}_{n_guess_coord}_{n_slicer_coord}.pkl"
    A, _, _, _, bse = load_lwe(n,q,eta,k,lat_index)

    # we don't store the whole lattice basis Binit since it is fairly large for github
    Binit = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ] 
    for i in range( k*n ):
        Binit[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        Binit[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            Binit[i][j] = int( A[i-k*n,j] )

    then = perf_counter()
    #restore precomputed g6k and initialize it
    g6k = Siever.restore_from_file( out_path + filename_g6kdump ) 
    # Needed to ensure that all locals are correct.
    # Ideally, already done.
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve

    #if we need to reduce the basis further, we do so and throw the precomputed database away
    #since it will be altered by the reduction. 
    # TODO: we can try inserting a vector from siever into the basis, since it was already computed.
    # n_slicer_coord += delta_slicer_coord
    overhead_tbkz = time.perf_counter()
    beta = 0
    G = g6k.M #the GSO obj. for first k*n-kappa vectors.
    bkz_performed = False
    LR = LatticeReduction( G.B, threads_bkz=nthreads )
    overhead_tbkz = 0
    for beta in bkz_beta_range:
        lens = test_vect_proj(G, n_slicer_coord, n_tests=NPROJ_TESTS, eta=eta)
        est_norm = np.percentile(lens,50)
        print(f"#{lat_index} est_proj_norm is: {est_norm}")
        # if est_norm <= HYB_PROJ_THRESHOLD:
        #     break
        
        then_round=time.perf_counter()
        LR.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        bkz_performed = True
        print(f"#{lat_index} Additional BKZ-{beta} done in {round_time}")
        sys.stdout.flush()
        G = LR.gso

        overhead_tbkz_ = time.perf_counter() - overhead_tbkz
        overhead_tbkz += overhead_tbkz_
        lens = test_vect_proj(G, n_slicer_coord, n_tests=NPROJ_TESTS, eta=eta)
        est_norm = np.percentile(lens,50)
        print(f"#{lat_index} final est_proj_norm is: {est_norm}")
        for delta in range(beta,beta+delta_slicer_coord+1):
            n_slicer_coord = delta
            overhead_tsieve = time.perf_counter()
            assert n_slicer_coord <= G.d, f"Too many slicer coords: {n_slicer_coord}>{G.d}"

            g6k = Siever(G,param_sieve)
            print(g6k.M.d-n_slicer_coord)
            g6k.initialize_local(g6k.M.d-n_slicer_coord,g6k.M.d-n_slicer_coord,g6k.M.d)
            print("Running bdgl2...")
            then = time.perf_counter()
            g6k(alg="bdgl2")
            print(f"bdgl2 done in {time.perf_counter()-then}")

            overhead_tsieve = time.perf_counter() - overhead_tsieve  
            H11 = g6k.M.B 

            # Gaussian heuristic for the last sieve_dim dimensioal projective lattice of G.
            # ALL {from/to}_canonical_scaled calls must use scale_fact=gh_sub, or things go out of hand.
            gh_sub = gaussian_heuristic(G.r()[-n_slicer_coord:])
            print(f"Sieving-1 done in {perf_counter() - then}")
            b0 = None
            for tmp in g6k.itervalues():
                b0 = G.B[-n_slicer_coord:].multiply_left( tmp )
                break
            b0 = from_canonical_scaled( G, b0, offset=n_slicer_coord,scale_fact=gh_sub )
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

                #deduce the projected error norm
                dist_sq_bnd = e_@e_
                dist_bnd = dist_sq_bnd**0.5
                dist_threshold = ( G.r()[-n_slicer_coord] / gh_sub )**0.5
                print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
                print(f"dist_sq_bnd: {dist_sq_bnd}")
                print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")

                B = IntegerMatrix.from_matrix(Binit)

                len_bound = dist_sq_bnd
                tracer = {}
                # v = alg_3_debug(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg3=None)
                iter_v = alg_3_debug_v2(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg3=tracer)
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
                        print(f"Success in experiment! @{guess_cntr} guess")
                        break
                fail_reason = "other" if guess_cntr<1 else "parasites"
                stats_dict[(n, lat_index, beta, n_slicer_coord, n_guess_coord, ex_cntr)] = {
                    "walltime": tracer["wrong_guess_time_alg3"] + tracer["wrong_guess_time_alg2"],
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
                    "walltime_observed": perf_counter() - ex_timer, 
                    "overhead_tbkz": overhead_tbkz,
                    "overhead_tsieve": overhead_tsieve,
                }
                print(f"walltime: {tracer["walltime"]} | walltime_observed: {tracer["walltime_observed"]}")
                print(f" - - - {all(answer==v2)} - - - ")
    return stats_dict

if __name__=="__main__":
    """
    This file implements the hybrid attack on preprocessed Kyber instances.
    To generate ones, one needs to run attack_on_kyber.py (generating instances), run
    preprocessing.py (preprocess the data) and then run this file. 
    The attack is relaxed -- we do not guess all the subkeys, but rather consider a single batch.
    """
    n, k = 130, 1
    q, eta = 3329, 3
    latnum = 2
    n_guess_coord, n_slicer_coord = 4, 53
    # bkz_beta_range = range(n_slicer_coord-1,n_slicer_coord+4) #range of values of beta or None if no additional reduction to be performed
    bkz_beta_range = range(52,54) #range(60,62,1)
    delta_slicer_coord = 2 #integer >=0, n_slicer_coord + delta_slicer_coord will be the slicer dimension
    nthreads = 5
    nworkers = 3

    params={}
    params["nthreads"] = nthreads
    params["n"], params["k"], params["q"], params["eta"] = n, k, q, eta
    params["n_guess_coord"], params["n_slicer_coord"] = n_guess_coord, n_slicer_coord

    succ_cntr = 0
    ex_cntr = 0

    output = []
    pool = Pool( processes = nworkers )
    tasks = []
    for lat_index in range(latnum):
        output.append({})
        tasks.append( pool.apply_async(
            run_experiment, (lat_index, params, output[lat_index],bkz_beta_range,delta_slicer_coord)
            ) )
        
    stats_dict_agr = {}
    for t in tasks:
            stats_dict_agr.update(t.get())

    # print(ex_cntr, succ_cntr)
    print(stats_dict_agr)

    filename = f"tph_{n}_{n_guess_coord}_{n_slicer_coord+delta_slicer_coord}.pkl"
    print(f"saving results to {filename}")
    with open(filename, "wb") as file:
        pickle.dump( stats_dict_agr, file )
    pool.close()