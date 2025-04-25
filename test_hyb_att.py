from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys
from time import perf_counter
from experiments.lwe_gen import *

from hyb_att_on_kyber import alg_2_batched
from sample import *

from g6k.siever import SaturationError

from preprocessing import load_lwe

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from global_consts import *
from copy import copy

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"

def kyberGen(n, q = 3329, eta = 3, k=1):
    polys = []
    for i in range(k*k):
        polys.append( uniform_vec(n,0,q) )
    A = module(polys, k, k)

    return A,q

def se_gen(k,n,eta):
    s = binomial_vec(k*n, eta)
    e = binomial_vec(k*n, eta)
    return s, e

def generateLWEInstances(n, q = 3329, eta = 3, k=1, ntar=5):
    A,q = kyberGen(n,q = q, eta = eta, k=k)
    bse = []
    for _ in range(ntar):
        s, e = se_gen(k,n,eta)
        b = (s.dot(A) + e) % q
        bse.append( (b,s,e) )

    return A,q,bse

def batch_babai( g6k,target_candidates, dist_sq_bnd ):
    G = g6k.M

    bs = []
    index = 0
    minnorm, best_index = 10**32, 0
    for t in target_candidates:
        cb = G.babai(t)
        b = np.array( G.B.multiply_left( cb ) )
        bs.append(b)

        t = np.array(t)
        curnrm = (b-t)@(b-t)
        if curnrm < minnorm:
            minnorm = curnrm
            best_index = index
            best_cb = cb
        index+=1
    print(f"minnorm: {minnorm**0.5}")
    print(f"best_cb: {best_cb}")
    return best_cb

def alg_3_debug_v2(g6k,H11,B,target,n_guess_coord, dist, dist_param, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # Emulates batch CVPP with guessing.
    # - - - prepare targets - - -
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r-g6k.l):])
    dim = B.nrows
    print(f"dim: {dim}")

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    if dist=="binomial":
        distrib = centeredBinomial(dist_param)
    elif dist=="ternary":
         print(f"dist_param: {dist_param}")
         distrib = ternaryDist(dist_param)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    tracer_alg3["key_num"] = nsampl

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    sieve_dim = g6k.r-g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(NRAND_FACTOR*(1./nrand_)**sieve_dim)
    print(f"times: {ceil( len(g6k) / nrand )}")
    times = ceil( len(g6k) / nrand )

    tracer_alg2_correct, tracer_alg2_wrong = {}, {}
    # - - - BEGIN INCORRECT GUESS - - -
    wrong_guess_time_alg3 = time.perf_counter()
    target_candidates = []
    vtilde2s = []
    wrong_guess_time = time.perf_counter()
    for times in range(times): #Alg 3 steps 4-7 ceil( (nrand * nsampl) / len(g6k) )
        if times!=0 and times%1000 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        if times>0:
            etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        else:
            etilde2 = np.array( distrib.sample( n_guess_coord ) )
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        tmp = np.array( H12.multiply_left(vtilde2) )

        t1_ = np.array( list(t1) ) - tmp
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: deduce what is the betamax
    # def of alg_2_batched is in hyb_att_on_kyber.py
    print(f"- - - alg 2 on incorrect guess - - -")
    # ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_wrong )
    it = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_wrong )
    ctilde1 = np.zeros( dim-n_guess_coord )
    for ctilde1 in it: #what's returned is not quite relevant. The guess is wrong by design.
        break

    v1 = np.array( H11.multiply_left( ctilde1 ) )
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:
        v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
        babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
        v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

        v_t = v-np.array( target )
        vv = v_t@v_t
        if vv < minv:
            minv = vv
            argminv = v
        cntr+=1
    wrong_guess_time = time.perf_counter() - wrong_guess_time
    wrong_guess_time_alg3 = time.perf_counter() - wrong_guess_time_alg3
    # - - - END INCORRECT GUESS - - -
    if not tracer_alg3 is None: #this belongs here since this point is always reached
                    tracer_alg3["wrong_guess_time_alg3"] = wrong_guess_time
                    tracer_alg3["wrong_guess_time_alg2"] = tracer_alg2_wrong["walltime"]
    # - - - BEGIN CORRECT GUESS - - -
    target_candidates = []
    vtilde2s = []
    correct_guess_time_start = time.perf_counter()
    for times in range(times): #Alg 3 steps 4-7 ceil( (nrand * nsampl) / len(g6k) )
        if times!=0 and times%1000 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        if times>0:
            etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        else:
            etilde2 = np.array(-s[-n_guess_coord:])
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        tmp = np.array( H12.multiply_left(vtilde2) )

        t1_ = np.array( list(t1) ) - tmp
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: deduce what is the betamax
    # def of alg_2_batched is in hyb_att_on_kyber.py
    print(f"- - - alg 2 on correct guess - - -")
    it = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg2_correct )
    if not tracer_alg3 is None: #this belongs here since we may never start the loop
                    tracer_alg3["correct_guess_time_alg3"] = 0
                    tracer_alg3["correct_guess_time_alg2"] = 0
    for ctilde1 in it: #we do not quite care what it
        v1 = np.array( H11.multiply_left( ctilde1 ) )
        argminv_correct = None
        minv = 10**12
        cntr = 0
        for vtilde2 in vtilde2s:
            v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
            babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
            v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

            v_t = v-np.array( target )
            vv = v_t@v_t
            if vv < minv:
                minv = vv
                argminv_correct = v
                correct_guess_time = time.perf_counter() - correct_guess_time_start
                if not tracer_alg3 is None: #this belongs here since we may never reach the end of yield
                    tracer_alg3["correct_guess_time_alg3"] = correct_guess_time
                    tracer_alg3["correct_guess_time_alg2"] = tracer_alg2_correct["walltime"]
                yield argminv_correct
            cntr+=1

    # correct_guess_time = time.perf_counter() - correct_guess_time
    # - - - END CORRECT GUESS - - -


def alg_3_debug(g6k,H11,B,target,n_guess_coord, distrib, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # Emulates batch CVPP with guessing.
    # - - - prepare targets - - -
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r-g6k.l):])
    dim = B.nrows
    print(f"dim: {dim}")

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    # distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    target_candidates = []
    vtilde2s = []

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    sieve_dim = g6k.r-g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(NRAND_FACTOR*(1./nrand_)**sieve_dim)
    print(f"times: {ceil( len(g6k) / nrand )}")
    sieve_dim = g6k.r-g6k.l

    print(f"times: {ceil( len(g6k) / nrand )}")
    for times in [0]: #Alg 3 steps 4-7 ceil( (nrand * nsampl) / len(g6k) )
        if times!=0 and times%1000 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        if times>0:
            etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        else:
            etilde2 = np.array(-s[-n_guess_coord:])
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        print(f"vtilde2 babai norm: {(vtilde2@vtilde2)**0.5}")
        print(f"tmp babai norm: {(tmp@tmp)**0.5}")

        t1_ = np.array( list(t1) ) - tmp
        if not tracer_alg3 is None:
            tracer_alg3["es"] -= tmp
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    # def of alg_2_batched is in hyb_att_on_kyber.py
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=tracer_alg3 )

    v1 = np.array( H11.multiply_left( ctilde1 ) )
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:
        v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
        babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
        v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

        v_t = v-np.array( target )
        vv = v_t@v_t
        if vv < minv:
            minv = vv
            argminv = v
        cntr+=1
    return argminv

def run_experiment(params, stats_dict, tracer=None):
    nthreads = params["nthreads"]
    n, q, dist, dist_param = params["n"], params["q"], params["dist"], params["dist_param"]
    n_guess_coord, n_slicer_coord = params["n_guess_coord"], params["n_slicer_coord"]
    beta_pre = params["beta_pre"]
    seed = params["seed"]
    lat_index = seed[0]

    # if dist == "binomial":
    #     distrib = centeredBinomial(dist_param)
    # elif dist=="ternary":
    #     distrib = ternaryDist(dist_param)
    # else:
    #      raise ValueError(f"distrib: expected \"binomial\" or \"ternary\", got {distrib}")

    ft = "ld" if 2*n<140 else ( "dd" if config.have_qd else "mpfr")
    FPLLL.set_precision(210)
    dim = 2*n

    print(f"float_type: {ft}")
    succ_cntr = 0
    ex_cntr = 0

    #n,q,dist,dist_param,lat_index
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
    #restore precomputed g6k and initialize it
    g6k = Siever.restore_from_file( out_path + filename_g6kdump )
    # Needed to ensure that all locals are correct.
    # Ideally, already done.
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve
    H11 = g6k.M.B

    G = g6k.M #the GSO obj. for first k*n-kappa vectors.
    # Gaussian heuristic for the last sieve_dim dimensioal projective lattice of G.
    # ALL {from/to}_canonical_scaled calls must use scale_fact=gh_sub, or things will go out of hand.
    gh_sub = gaussian_heuristic(G.r()[-n_slicer_coord:])

    for (b, s, e) in bse:
        ex_cntr+=1
        print(f"running exp # {ex_cntr} out of {len(bse)} lat ind: {lat_index}")
        ex_timer = perf_counter()
        assert ( all( (s@A+e)%q == b ) ), f"wrong lwe instance! {(A@s+e)%q , b}"
        print(f"len {len(Binit), len(Binit[0])}")

        answer = np.concatenate( [b-e,s] )

        print(f"Database size: {len(g6k)}")

        t = np.concatenate([b,n*[0]])
        e_ = np.concatenate([e,-s])[:-n_guess_coord]
        # project the error vector onto the last n_sieve_dim GS-vectors.
        e_ = from_canonical_scaled( G,e_,offset=n_slicer_coord,scale_fact=gh_sub )
        # print(f"hyb e_: {e_}")

        #deduce the projected error norm
        dist_sq_bnd = e_@e_
        dist_bnd = dist_sq_bnd**0.5
        dist_threshold = ( G.r()[-n_slicer_coord] / gh_sub )**0.5


        print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
        print(f"dist_sq_bnd: {dist_sq_bnd}")
        print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")

        B = IntegerMatrix.from_matrix(Binit)


        # project the error vector onto the last n_sieve_dim GS-vectors.
        tracer = {}
        with open("hybHvar","wb") as file:
            pickle.dump(pickle.dump([n_slicer_coord,t,e,s,EPS2 * dist_sq_bnd, g6k.M.r(), gh_sub], file), file)
        iter_v = alg_3_debug_v2(g6k,H11,B,t,n_guess_coord, dist, dist_param, s, dist_sq_bnd=EPS2 * dist_sq_bnd, nthreads=nthreads, tracer_alg3=tracer)
        guess_cntr = 0
        sli_succ = False
        for v in iter_v:
            if v is None:
                v = np.array( len(answer)*[0] )
            guess_cntr+=1
            # print(f"v: {v}")
            # print(f"vs: {answer}")
            print(f" - - - - - - ")

            v2 = v

            sli_succ = all(answer==v2)
            # print(f"slicer:\n {sli_succ}")
            if (sli_succ):
                succ_cntr+=1
                print(f"Success in experiment! @{guess_cntr} guess")
                break

        fail_reason = "other" if guess_cntr<1 else "parasites"
        a0, a1 = tracer["wrong_guess_time_alg3"] , tracer["wrong_guess_time_alg2"]
        print(f"a0, a1: {a0,a1}")
        walltime, walltime_observed = tracer["wrong_guess_time_alg3"] + tracer["wrong_guess_time_alg2"], perf_counter() - ex_timer
        stats_dict[(n, lat_index, n_slicer_coord, n_guess_coord, ex_cntr)] = {
            "walltime": walltime,
            "dist_bnd": dist_bnd,
            "succ": (sli_succ),
            "fail_reason": None if (sli_succ) else fail_reason,
            "key_num": tracer["key_num"], #number of guessed keys
            "g6k_len": len(g6k),
            "wrong_guess_time_alg3": tracer["wrong_guess_time_alg3"],
            "correct_guess_time_alg3": tracer["correct_guess_time_alg3"],
            "wrong_guess_time_alg2": tracer["wrong_guess_time_alg2"],
            "correct_guess_time_alg2": tracer["correct_guess_time_alg2"],
            "walltime_observed": walltime_observed,
        }
        print(f"walltime: {walltime} | walltime_observed: {walltime_observed}")

        print(f" - - - {sli_succ} after {guess_cntr} guesses - - - ")
    return stats_dict

if __name__=="__main__":
    """
    This file implements the hybrid attack on preprocessed Kyber instances.
    To generate ones, one needs to run attack_on_kyber.py (generating instances), run
    preprocessing.py (preprocess the data) and then run this file.
    The attack is relaxed -- we do not guess all the subkeys, but rather consider a single batch.
    """
    n = 135
    q, eta = 3329, 3
    # dist, dist_param = "ternary", 1/6.
    dist, dist_param = "binomial", 2
    n_guess_coord, n_slicer_coord = 6, 53
    beta_pre = 52
    nthreads = 5
    nworkers = 1
    latnum = 2

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
            run_experiment, (copy(params), output[lat_index])
            ) )

    stats_dict_agr = {}
    for t in tasks:
            stats_dict_agr.update(t.get())

    print(stats_dict_agr)
    #filename = f"tha_{n}_{n_guess_coord}_{n_slicer_coord}.pkl"
    filename = f"tha_{n}_{n_guess_coord}_{n_slicer_coord}_{beta_pre}.pkl"
    print(f"Saving to {filename}")
    with open(filename, "wb") as file:
        pickle.dump( stats_dict_agr, file )
    pool.close()
    print( stats_dict_agr )