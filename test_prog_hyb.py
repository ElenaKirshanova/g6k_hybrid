from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys
from time import perf_counter
from experiments.lwe_gen import *

from hyb_att_on_kyber import alg_3, alg_2_batched
from sample import *

from g6k.siever import SaturationError
from test_alg2 import alg_2_batched_debug

from preprocessing import load_lwe

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from LatticeReduction import LatticeReduction

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"
max_nsampl = 2**10

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

def alg_3_debug_v2(g6k,H11,B,target,n_guess_coord, eta, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # Emulates batch CVPP with guessing.
    # - - - prepare targets - - -
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r-g6k.l):])
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = []
    vtilde2s = []

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    sieve_dim = g6k.r-g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(5*(1./nrand_)**sieve_dim)
    print(f"times: {ceil( len(g6k) / nrand )}")
    for times in range( ceil( len(g6k) / nrand ) ): #Alg 3 steps 4-7 ceil( (nrand * nsampl) / len(g6k) )
        if times!=0 and times%64 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        if times>0:
            etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        else:
            etilde2 = np.array(-s[-n_guess_coord:])
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        print(f"vtilde2 babai norm: {vtilde2@vtilde2}")
        print(f"tmp babai norm: {tmp@tmp}")

        t1_ = np.array( list(t1) ) - tmp
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: deduce what is the betamax
    # def alg_2_batched is in hyb_att_on_kyber.py
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

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

def alg_3_debug(g6k,H11,B,target,n_guess_coord, eta, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # Emulates batch CVPP with guessing.
    # - - - prepare targets - - -
    then_start = perf_counter()
    gh_sub = gaussian_heuristic(g6k.M.r()[-(g6k.r-g6k.l):])
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = []
    vtilde2s = []

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    sieve_dim = g6k.r-g6k.l

    from hybrid_estimator.batchCVP import batchCVPP_cost
    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(5*(1./nrand_)**sieve_dim)
    print(f"times: {ceil( len(g6k) / nrand )}")
    for times in [0]: #Alg 3 steps 4-7 ceil( (nrand * nsampl) / len(g6k) )
        if times!=0 and times%64 == 0:
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
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: deduce what is the betamax
    # def alg_2_batched is in hyb_att_on_kyber.py
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

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

    # loading the preprocessed H11 (see alg. 3 in the paper)
    #TODO: the next number after n_guess_coord does not carry any meaningful info. Consider deleting.
    # with open(out_path+f"kyb_prehybrid_{n}_{q}_{eta}_{k}_{lat_index}_{n_guess_coord}.pkl", "rb") as file:
    #     H11 = pickle.load(file)["B"]
    # H11 = Binit[:len(Binit)-kappa] #the part of basis to be reduced
    # H11 = IntegerMatrix.from_matrix( [ h11[:len(Binit)-kappa] for h11 in H11  ] )

    # H11r, H11c = H11.nrows, H11.ncols

    then = perf_counter()
    #restore precomputed g6k and initialize it
    g6k = Siever.restore_from_file( out_path + filename_g6kdump ) 
    # g6k.initialize_local(g6k.M.d-n_slicer_coord, g6k.M.d-n_slicer_coord, g6k.M.d)
    # Needed to ensure that all locals are correct.
    # Ideally, already done.
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve

    # g6k = Siever(g6k.M,param_sieve) #temporary solution
    # print(g6k.M.d-n_slicer_coord)
    # g6k.initialize_local(g6k.M.d-n_slicer_coord,g6k.M.d-n_slicer_coord,g6k.M.d)
    # print("Running bdgl2...")
    # then = time.perf_counter()
    # g6k(alg="bdgl2")
    # print(f"bdgl2 done in {time.perf_counter()-then}")
    # g6k.M.update_gso()

    #if we need to reduce the basis further, we do so and throw the precomputed database away
    #since it will be altered by the reduction. 
    # TODO: we can try inserting a vector from siever into the basis, since it was already computed.
    n_slicer_coord += delta_slicer_coord
    overhead_t_start = time.perf_counter()
    beta = 0
    if not bkz_beta_range is None:
        G = g6k.M #the GSO obj. for first k*n-kappa vectors.
        LR = LatticeReduction( G.B, threads_bkz=nthreads )
        bkz_start = time.perf_counter()
        for beta in bkz_beta_range:
            lens = test_vect_proj(G, n_slicer_coord, n_tests=2048, eta=eta)
            est_norm = np.percentile(lens,50)
            print(f"#{lat_index} est_proj_norm is: {est_norm}")
            if est_norm <= 0.95:
                break
            
            then_round=time.perf_counter()
            LR.BKZ(beta,tours=2)
            round_time = time.perf_counter()-then_round
            print(f"#{lat_index} Additional BKZ-{beta} done in {round_time}")
            sys.stdout.flush()
            G = LR.gso
    
    if (delta_slicer_coord>0) or (not bkz_beta_range is None): #if context grows, or we did bkz, we need to reinstantiate g6k
        assert n_slicer_coord <= G.d, f"Too many slicer coords: {n_slicer_coord}>{G.d}"

        g6k = Siever(G,param_sieve) #temporary solution
        print(g6k.M.d-n_slicer_coord)
        g6k.initialize_local(g6k.M.d-n_slicer_coord,g6k.M.d-n_slicer_coord,g6k.M.d)
        print("Running bdgl2...")
        then = time.perf_counter()
        g6k(alg="bdgl2")
        print(f"bdgl2 done in {time.perf_counter()-then}")
    overhead_t = time.perf_counter() - overhead_t_start  
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
    lambda1 = (b0@b0)**0.5

    gh = gaussian_heuristic( g6k.M.r()[-n_slicer_coord:] )
    print(f"r / r = {(g6k.M.r()[-n_slicer_coord] / g6k.M.r()[-1])**0.5}")
    for (b, s, e) in bse:
        ex_cntr+=1
        print(f"running exp # {ex_cntr}")
        #TODO: n=160, kappa=16, n_slicer_coord=71 returns large output far from the target when slicer fails.
        # Investigate, if this is correct. One can use the code below to encounter this issue immediately.
        # if not ex_cntr==9: 
        #     print(f"debug, omitting exp {ex_cntr}")
        #     continue
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
        print(f"lambda1: {lambda1}")
        print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
        print(f"dist_sq_bnd: {dist_sq_bnd}")
        print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")

        B = IntegerMatrix.from_matrix(Binit)

        len_bound = dist_sq_bnd
        # no guessing version of alg_3
        v = alg_3_debug(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=len_bound, nthreads=nthreads, tracer_alg3=None)
        # v = alg_3_debug_v2(g6k,H11,B,t,n_guess_coord, eta, s, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg3=None)
        print(f"e_: {e_}")
        if v is None:
            v = np.array( len(answer)*[0] )
        print(f"v: {v}")
        print(f"vs: {answer}")
        print(f" - - - - - - ")

        # LR2 = LatticeReduction( B )
        # cv = LR2.gso.babai( v )
        # v2 = LR2.basis.multiply_left( cv )
        v2 = v
        succ_alg_3_debug = all( answer==v2 )

        sli_succ = answer==v2
        print(f"slicer:\n {sli_succ}")
        if all(sli_succ):
            succ_cntr+=1
        stats_dict[(n,lat_index, n_slicer_coord, n_guess_coord, ex_cntr)] = {
            "walltime": perf_counter() - ex_timer, 
            "dist_bnd": dist_bnd, 
            "succ": all(sli_succ),
            "key_num": 0, #number of guessed keys
            "beta": beta,
            "overhead_t": overhead_t,
        }

        print(f" - - - {all(answer==v2)} - - - ")
    return stats_dict

if __name__=="__main__":
    """
    This file implements the hybrid attack on preprocessed Kyber instances.
    To generate ones, one needs to run attack_on_kyber.py (generating instances), run
    preprocessing.py (preprocess the data) and then run this file. 
    The attack is relaxed -- we do not guess all the subkeys, but rather consider a single batch.
    """
    n, k = 140, 1
    q, eta = 3329, 3
    latnum = 10
    n_guess_coord, n_slicer_coord = 5, 55
    bkz_beta_range = range(n_slicer_coord,n_slicer_coord+6) #range of values of beta or None if no additional reduction to be performed
    delta_slicer_coord = 5 #integer >=0, n_slicer_coord + delta_slicer_coord will be the slicer dimension
    nthreads = 2
    nworkers = 2

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

    filename = f"tha_{n}_{n_guess_coord}_{n_slicer_coord+delta_slicer_coord}.pkl"
    print(f"saving results to {filename}")
    with open(filename, "wb") as file:
        pickle.dump( stats_dict_agr, file )
    pool.close()