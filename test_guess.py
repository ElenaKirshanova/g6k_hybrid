from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys
import multiprocessing
from time import perf_counter
from experiments.lwe_gen import *

from hyb_att_on_kyber import alg_3, alg_2_batched
from sample import *

from g6k.siever import SaturationError
from test_alg2 import alg_2_batched_debug

import gc
# from Entropy.Distribution_Classes import *
# from Entropy.Entropy_stuff import *
# from Entropy.Multinomial import *
# from Entropy.Largelog import *
# from Entropy.Root_sum import approx_sum_of_roots
# from Entropy.Compact_Dictionary import *
from Entropy import Distribution_Classes
def centeredBinomialDict(eta):
    n = 2*eta
    D = {}
    for i in range(-eta,eta+1):
        D[i] = comb(n, eta+i) / 2**n
    # print(D)
    # return Distribution(D)
    return D

inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"
max_nsampl = 2**8

def guess_keys(n_guess_coord, left, right, t1, t2, H12, D, storage, s):
    then = perf_counter()
    keys = D.GetKeys(left,right,n_guess_coord)
    print(f"GetKeys done in {perf_counter()-then}", flush=True)
    vtilde2s, target_candidates = [], []
    key_encount = False
    then = perf_counter()
    for tmp in keys:
        etilde2 = np.array( tmp[1] )
        if all(etilde2 == -s[-n_guess_coord:]):
            print(f"key encountered! {tmp[0]}")
            key_encount = True

        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        t1_ = np.array( list(t1) ) - tmp
        target_candidates.append( t1_ )
    print(f"Constructing {len(keys)} vectors done in {perf_counter()-then}", flush=True)
    then = perf_counter()
    storage[(left,right)] = vtilde2s, target_candidates
    print(f"Copy done in {perf_counter()-then}", flush=True)
    return key_encount

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

def alg_3_debug_v2(g6k,H11,target,n_guess_coord, eta, s, ee, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # - - - prepare targets - - -
    then_start = perf_counter()
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}", flush=True)
    # nsampl = min(max_nsampl, nsampl)
    target_candidates = []
    vtilde2s = []

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    guessed_key_position = -1

    D = centeredBinomialDict(3)
    D = Distribution_Classes.distribution( D )
    then = perf_counter()
    keys = D.GetKeys(0,nsampl,n_guess_coord)
    print(f"GetKeys done in {perf_counter()-then}", flush=True)
    for times in range(nsampl): #Alg 3 steps 4-7
        if times%20000==0:
            print(f"{times} of {nsampl} targets precomputed")
        # etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        etilde2 = np.array( keys[times][1] )
        if all(etilde2 == -s[-n_guess_coord:]):
            print(f"Encountered key after {times} guesses", flush=True)
            if guessed_key_position<0:
                guessed_key_position = times

        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        t1_ = np.array( list(t1) ) - tmp
        target_candidates.append( t1_ )
    msg = "Key not guessed"
    print(f"Guessed key position: {msg if guessed_key_position<0 else guessed_key_position}")

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    if guessed_key_position>=0:
        for batches in range(len(target_candidates)//max_nsampl+1):
            left, right = batches*max_nsampl, min( (batches+1)*max_nsampl, len(target_candidates))
            if not( guessed_key_position >= left and guessed_key_position<right ):
                continue
            print(f"left, right, len: {left, right, len(target_candidates)}", flush=True)
            tar_cand = target_candidates[left:right]
            ctilde1 = alg_2_batched( g6k,tar_cand, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

            v1 = np.array( H11.multiply_left( ctilde1 ) )
            #keep a track of v2?
            argminv = None
            minv = 10**12
            cntr = 0
            # print("vv__: ", end="")
            for vtilde2 in vtilde2s:
                v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
                babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
                v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

                # print(v)
                # t = target_candidates[cntr]
                v_t = v-np.array( target ) #+ tmp
                vv = v_t@v_t
                # print(f"{vv**0.5}", end = ", ")
                if vv < minv:
                    minv = vv
                    argminv = v
                if minv < ee:
                    break
                cntr+=1
            print()
            print(f"minv: {minv}")
            return argminv
    return None

def alg_3_debug(g6k,H11,target,n_guess_coord, eta, s, ee, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # - - - prepare targets - - -
    then_start = perf_counter()
    dim = B.nrows
    print(f"dim: {dim}")
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = target[:-n_guess_coord], target[-n_guess_coord:]
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( distrib.entropy * n_guess_coord ) )
    print(f"nsampl: {nsampl}", flush=True)
    # nsampl = min(max_nsampl, nsampl)
    vtilde2s, target_candidates = [], []

    H12 = IntegerMatrix.from_matrix( [list(b)[:dim-n_guess_coord] for b in B[dim-n_guess_coord:]] )
    guessed_key_position = -1

    D = centeredBinomialDict(3)
    D = Distribution_Classes.distribution( D )
    # - - -
    processes = []
    manager = multiprocessing.Manager()
    storage = manager.dict()
    # storage["vtilde2s"] = []
    # storage["target_candidates"] = []

    blocksize = nsampl // nthreads
    for cntr in range( ceil(nsampl / blocksize) ):
        left = cntr * blocksize
        right = min(nsampl, (cntr+1) * blocksize)
        process = multiprocessing.Process(
            target=guess_keys, args=(n_guess_coord, left, right, t1, t2, H12, D, storage, s)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    for key in storage.keys():
        vtilde2s += storage[key][0]
        target_candidates += storage[key][1]

    # msg = "Key not guessed"
    # print(f"Guessed key position: {msg if guessed_key_position<0 else guessed_key_position}")

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    print(f"len: {len(target_candidates)//max_nsampl+1}")
    argminv = None
    minv = 10**12
    cntr = 0
    for batches in range(len(target_candidates)//max_nsampl+1):
        left, right = batches*max_nsampl, min( (batches+1)*max_nsampl, len(target_candidates))
        print(f"left, right, len: {left, right, len(target_candidates)}", flush=True)
        tar_cand = target_candidates[left:right]
        ctilde1 = alg_2_batched( g6k,tar_cand, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )

        v1 = np.array( H11.multiply_left( ctilde1 ) )
        #keep a track of v2?
        # print("vv__: ", end="")
        for vtilde2 in vtilde2s:
            v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
            babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
            v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

            # print(v)
            # t = target_candidates[cntr]
            v_t = v-np.array( target ) #+ tmp
            vv = v_t@v_t
            # print(f"{vv**0.5}", end = ", ")
            if vv < minv:
                minv = vv
                argminv = v
            if minv < ee:
                print(f"Breaking after success!")
                break
            cntr+=1
    print()
    print(f"minv: {minv}")
    return argminv

if __name__=="__main__":
    n, k = 125, 1
    eta = 3
    n_guess_coord, n_slicer_coord = 5, 50
    betamax = 47
    sieve_dim_max = n_slicer_coord
    nsieves = 2
    nthreads = 5
    dim = 2*k*n
    ft = "ld" if 2*k*n<140 else ( "dd" if config.have_qd else "mpfr")
    print(f"float_type: {ft}")
    # FPLLL.set_precision(250)
    # ft = "qd"

    load_flag = True
    filename = f"testlat_{n}_g{n_guess_coord}_b{betamax}.pkl"
    if not load_flag:
        A,q,bse = generateLWEInstances(n, q = 3329, eta = eta, k=k, ntar=10)
        b, s, e = bse[0]

        Binit = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
        for i in range( k*n ):
            Binit[i][i] = int( q )
        for i in range(k*n, 2*k*n):
            Binit[i][i] = 1
        for i in range(k*n, 2*k*n):
            for j in range(k*n):
                Binit[i][j] = int( A[i-k*n,j] )
        H11 = IntegerMatrix.from_matrix( [b[:dim - n_guess_coord] for b in Binit[:dim - n_guess_coord] ] )

        LR = LatticeReduction( H11, nthreads )
        for beta in range(5,betamax+1):
            then = perf_counter()
            LR.BKZ( beta, tours=5 )
            print(f"BKZ-{beta} done in {perf_counter()-then}", flush=True)
        H11 = LR.basis

        with open(filename, "wb") as file:
            pickle.dump( [ Binit, H11, A,q,bse ], file )
    else:
        with open(filename, "rb") as file:
            Binit, H11, A,q,bse = pickle.load( file)
        # b, s, e = bse[7]

    H11r, H11c = H11.nrows, H11.ncols
    G = GSO.Mat( H11,U=IntegerMatrix.identity(H11r,int_type=H11.int_type), UinvT=IntegerMatrix.identity(H11r,int_type=H11.int_type), float_type=ft )
    H11r, H11c = H11.nrows, H11.ncols
    G.update_gso()
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['db_size_base'] = (4/3.)**0.5 #(4/3.)**0.5 ~ 1.1547
    param_sieve['db_size_factor'] = 3.35 #3.2


    param_sieve['saturation_ratio'] = 0.5
    param_sieve['saturation_radius'] = 1.32
    print(f"Running sieving: {param_sieve}", flush=True)
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(H11r-n_slicer_coord, H11r-n_slicer_coord, H11r)
    then = perf_counter()
    g6k(alg="bdgl2")
    print(f"Sieving-1 done in {perf_counter() - then}")

    gh = gaussian_heuristic( g6k.M.r()[-n_slicer_coord:] )
    print(f"r / r = {(g6k.M.r()[-n_slicer_coord] / g6k.M.r()[-1])**0.5}")
    for (b, s, e) in bse:
        answer = np.concatenate( [b-e,s] )
        # print(f"Solving...")
        # Bnp = np.array( [ np.array(b) for b in Binit ] )
        # print(np.linalg.lstsq(Bnp.transpose(),answer))

        print(f"Database size: {len(g6k)}")

        t = np.concatenate([b,n*[0]])
        e_ = np.concatenate([e,-s])[:-n_guess_coord]
        # e_ = from_canonical_scaled( G,e_,offset=n_slicer_coord )[-n_slicer_coord:]
        e_ = from_canonical_scaled( G,e_,offset=n_slicer_coord )

        # for it in g6k.itervalues():
        #     v = g6k.M.B[-n_slicer_coord:].multiply_left( it )
        #     v = np.array( from_canonical_scaled( g6k.M,v,offset=n_slicer_coord ) )
        #     lambda1 = ( v@v )**0.5
        #     break

        dist_sq_bnd = e_@e_
        gh_sub = gaussian_heuristic(G.r()[-n_slicer_coord:])
        dist_bnd = dist_sq_bnd**0.5
        dist_threshold = ( G.r()[-n_slicer_coord] / gh_sub )**0.5
        print(f"dist_bnd: {dist_bnd} | dist_threshold: {dist_threshold} | ratio: {dist_bnd/dist_threshold}")
        print(f"dist_sq_bnd: {dist_sq_bnd}")
        print(f"len(e_): {len(e_)} G.M.nrows(): {G.B.nrows}")
        rs = np.array( G.r()[-n_slicer_coord:] ) / gh_sub
        rs = np.array( [ sqrt(rr) for rr in rs ] )
        # print(f"Checking errs:")
        # print(np.abs(e_) / rs)

        B = IntegerMatrix.from_matrix(Binit)

        len_bound = dist_sq_bnd
        v = alg_3_debug(g6k,H11,t,n_guess_coord, eta, s, ee=e@e, dist_sq_bnd=len_bound, nthreads=nthreads, tracer_alg3=None)
        print(f" - - - - - - ")

        LR2 = LatticeReduction( B )
        for beta in range(4,15):
            LR2.BKZ(beta, tours=2)
        cv = LR2.gso.babai( v )
        v2 = LR2.basis.multiply_left( cv )
        succ_alg_3_debug = all( answer==v2 )

        print(f"slicer:\n {answer==v2}")
        print(f"succ_alg_3_debug: {succ_alg_3_debug}")
        print(f"Next vector...")

        # print(f"- - - Now slicer with guessing - - -")
        # len_bound = dist_sq_bnd
        # vbab = np.array( alg_3_debug_v2( g6k,H11,t,n_guess_coord, eta, s, ee=e@e, dist_sq_bnd=len_bound, nthreads=nthreads, tracer_alg3=None ) )
        #
        # print(f"babai:\n {answer==vbab}")
        # print(f"Next vector...")
        # succ_alg_3_debug_v2 = all( answer==vbab )
        # print(f"succ_alg_3_debug vs succ_alg_3_debug_v2: {succ_alg_3_debug, succ_alg_3_debug_v2}")
