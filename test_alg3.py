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

inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"
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

def alg_3_debug_v2(g6k,H11,target,n_guess_coord, eta, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # - - - prepare targets - - -
    then_start = perf_counter()
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
    for times in range(max_nsampl): #Alg 3 steps 4-7
        # if times!=0 and times%64 == 0:
        #     print(f"{times} done out of {nsampl}", end=", ")
        # if times>0:
        #     etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        # else:
        #     etilde2 = np.array(-s[-n_guess_coord:])
        etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)

        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        # print(f"vtilde2 babai norm: {vtilde2@vtilde2}")
        # print(f"tmp babai norm: {tmp@tmp}")

        # print(f"len(vtilde2): {len(vtilde2)} len(t1): {len(t1)}")
        # print(f"dim: {dim} n_guess_coord: {n_guess_coord}")
        t1_ = np.array( list(t1) ) - tmp
        # print(t1_)
        # print(f"len t1_: {len(t1_)}")
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: dist_sq_bnd might have changed at this point (or even in attacker)
    #TODO: deduce what is the betamax
    # betamax = 48
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )
    # ctilde1 = batch_babai( g6k,target_candidates, dist_sq_bnd )
    # print(f"target_candidates babai = {target_candidates}")

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
        cntr+=1
    print()
    return argminv

def alg_3_debug(g6k,H11,target,n_guess_coord, eta, s, dist_sq_bnd=1.0, nthreads=1, tracer_alg3=None):
    # - - - prepare targets - - -
    then_start = perf_counter()
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
    for times in range(1): #Alg 3 steps 4-7
        if times!=0 and times%64 == 0:
            print(f"{times} done out of {nsampl}", end=", ")
        if times>0:
            etilde2 = np.array( distrib.sample( n_guess_coord ) ) #= (0 | e2)
        else:
            etilde2 = np.array(-s[-n_guess_coord:])
        # print(f"len etilde2: {len(etilde2)}")
        # print(f"etilde2 babai: {etilde2}")
        vtilde2 = np.array(t2)-etilde2
        vtilde2s.append( vtilde2  )
        #compute H12*H22^-1 * vtilde2 = H12*vtilde2 since H22 is identity
        tmp = np.array( H12.multiply_left(vtilde2) )
        print(f"vtilde2 babai norm: {vtilde2@vtilde2}")
        print(f"tmp babai norm: {tmp@tmp}")

        # print(f"len(vtilde2): {len(vtilde2)} len(t1): {len(t1)}")
        # print(f"dim: {dim} n_guess_coord: {n_guess_coord}")
        t1_ = np.array( list(t1) ) - tmp
        # print(t1_)
        # print(f"len t1_: {len(t1_)}")
        target_candidates.append( t1_ )
    print()

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:dim-kappa] to avoid fp errors.
    """
    #TODO: dist_sq_bnd might have changed at this point (or even in attacker)
    #TODO: deduce what is the betamax
    # betamax = 48
    ctilde1 = alg_2_batched( g6k,target_candidates, dist_sq_bnd=dist_sq_bnd, nthreads=nthreads, tracer_alg2=None )
    # ctilde1 = batch_babai( g6k,target_candidates, dist_sq_bnd )
    # print(f"target_candidates babai = {target_candidates}")

    v1 = np.array( H11.multiply_left( ctilde1 ) )
    #keep a track of v2?
    argminv = None
    minv = 10**12
    cntr = 0
    for vtilde2 in vtilde2s:
        v2 = np.concatenate( [(dim-n_guess_coord)*[0],vtilde2] )
        babshift = np.concatenate( [ np.array( H12.multiply_left(vtilde2) ), n_guess_coord*[0] ] )
        v = np.concatenate([v1,n_guess_coord*[0]]) + v2 + babshift

        # print(v)
        # t = target_candidates[cntr]
        v_t = v-np.array( target ) #+ tmp
        vv = v_t@v_t
        print(f"vv__: {vv**0.5}")
        # print(f"babshift babai: {babshift}")
        print(f"v babai: {v}")
        if vv < minv:
            minv = vv
            argminv = v
        cntr+=1
    return argminv

if __name__=="__main__":
    n, k = 144, 1
    eta = 3
    n_guess_coord, n_slicer_coord = 10, 68
    betamax = 67
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

    # try: #db_size_base
    #     param_sieve['saturation_ratio'] = 0.95
    #     param_sieve['saturation_radius'] = 1.335
    #     # g6k = Siever(G,param_sieve)
    #     g6k.params = param_sieve
    #     print(f"Running sieving: {param_st_gs_reducedieve}")
    #     g6k.initialize_local(H11r-n_slicer_coord, H11r-n_slicer_coord, H11r)
    #     then = perf_counter()
    #     g6k(alg="bdgl2")
    #     print(f"Sieving-2 done in {perf_counter() - then}")
    # except SaturationError:
    #     print("Saturation error...")

    # - - - checking the database - - -
    # v_nrms = []
    gh = gaussian_heuristic( g6k.M.r()[-n_slicer_coord:] )
    # cntr = 0
    # print(f"Processing {len(g6k)} norms...")
    # for it in g6k.itervalues():
    #     # if cntr%200 == 0:
    #         # print(f"{cntr} dumped", end=", ", flush=True)
    #     v = g6k.M.B[-n_slicer_coord:].multiply_left( it )
    #     v = np.array( from_canonical_scaled( g6k.M,v,offset=n_slicer_coord ) )
    #     if ( v@v ) > 1.09**2 * (4/3.):
    #         break
    #     cntr+=1
        # v_nrms.append( ( v@v )**0.5 )
    # with open("tmp.pkl", "wb") as file:
    #     pickle.dump(v_nrms, file)
    # print()
    # print(f"Was: {len(g6k)}", end=", ")
    # g6k.shrink_db( cntr )
    # print(f"Is: {len(g6k)} corr. alpha: {len(g6k)**(1./n_slicer_coord)}")
    # - - - end checking the database - - -
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
        v = alg_3_debug(g6k,H11,t,n_guess_coord, eta, s, dist_sq_bnd=len_bound, nthreads=nthreads, tracer_alg3=None)
        print(f" - - - - - - ")

        LR2 = LatticeReduction( B )
        for beta in range(4,15):
            LR2.BKZ(beta, tours=2)
        cv = LR2.gso.babai( v )
        v2 = LR2.basis.multiply_left( cv )
        succ_alg_3_debug = all( answer==v2 )

        print(f"slicer:\n {answer==v2}")

        print(f" - - - {all(answer==v2)} - - - ")
        # print(f"- - - Now slicer with guessing - - -")
        # len_bound = dist_sq_bnd
        # vbab = np.array( alg_3_debug_v2( g6k,H11,t,n_guess_coord, eta, s, dist_sq_bnd=len_bound, nthreads=nthreads, tracer_alg3=None ) )
        # print(f"babai:\n {answer==vbab}")
        # print(f"Next vector...")
        # succ_alg_3_debug_v2 = all( answer==vbab )
        # print(f"succ_alg_3_debug vs succ_alg_3_debug_v2: {succ_alg_3_debug, succ_alg_3_debug_v2}")
        #
        # H11prime = g6k.M.B
        # for ii in range(H11.nrows):
        #     for jj in range(H11.ncols):
        #         assert( H11[ii][jj] == g6k.M.B[ii][jj] )
