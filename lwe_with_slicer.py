
import sys
import math

from fpylll import BKZ as fplll_bkz, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

import g6k.siever
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.siever import Siever
from g6k.slicer import RandomizedSlicer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from utils import *
from hybrid_estimator.batchCVP import batchCVPP_cost




if __name__=='__main__':
    fpylll_crossover = 55
    goal_margin = 1.5
    bkz_tours = 2

    alpha = 0.01
    n = 40
    A, c, q = load_lwe_challenge(n=n, alpha=alpha)

    ft = "ld" if 2*n<145 else ( "dd" if config.have_qd else "mpfr")


    try:
        min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                    samples=A.nrows, decouple=True)
        print(min_cost_param)
        (b, s, m) = min_cost_param #bkz_dim, svp_dim, num_samples
    except TypeError:
        raise TypeError("No winning parameters.") #for alpha=0.045 it fails to find the params

    target_norm = goal_margin * (alpha*q)**2 * m

    if m is None:
        m = A.nrows
    elif m > A.nrows:
        raise ValueError("Only m=%d samples available." % A.nrows)
    n = A.ncols

    B = IntegerMatrix(m+n, m)
    for i in range(m):
        for j in range(n):
            B[j, i] = A[i, j]
        B[i+n, i] = q


    B = LLL.reduction(B)
    assert(B[:n] == IntegerMatrix(n, m))
    B = B[n:]

    g6k = Siever(B)
    g6k.initialize_local(0,0,m)


    sieve_dim = s - 5
    print(g6k.full_n - sieve_dim)

    if sieve_dim<40:
        raise ValueError("The sieve dim is too small for the slicer")

    bkz_blocksize = list(range(10, 20)) + [b-20, b-17] + list(range(b - 14, b, 2))
    print(bkz_blocksize)

    for blocksize in bkz_blocksize:
        for tt in range(bkz_tours):
            if blocksize < fpylll_crossover:
                #print("Starting a fpylll BKZ-%d tour. " % (blocksize), end=' ')
                sys.stdout.flush()
                bkz = BKZReduction(g6k.M)
                par = fplll_bkz.Param(blocksize,
                                      strategies=fplll_bkz.DEFAULT_STRATEGY,
                                      max_loops=1)
                bkz(par)
            else:
                print("Starting a pnjBKZ-%d tour. " % (blocksize))
                pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=1,
                                     verbose=False,
                                     extra_dim4free=12,
                                     dim4free_fun="default_dim4free_fun",
                                     goal_r0=target_norm)
            g6k.lll(0, g6k.full_n)

            if g6k.M.get_r(0, 0) <= target_norm:
                break

    print("finished bkz preprocessing")

    #-------- starting bdgl on sublattice [n-sieve_dim, n] -------
    g6k.initialize_local(m-sieve_dim,m-sieve_dim,m)
    g6k.M.update_gso()

    print("starting bdgl2...")
    g6k(alg="bdgl2")
    print("finished bdgl2...")


    #project the target and babai-reduced it

    c = c[:m]
    GSO = g6k.M

    t_gs_non_scaled = GSO.from_canonical(c)[m-sieve_dim:]
    shift_babai_c =  list( GSO.babai( list(t_gs_non_scaled), start=m-sieve_dim, dimension=sieve_dim, gso=True) )
    shift_babai = GSO.B.multiply_left( (m-sieve_dim)*[0] + list( shift_babai_c ) )
    t_gs_reduced = from_canonical_scaled( GSO,np.array(c)-shift_babai,offset=sieve_dim )

    assert len(t_gs_reduced) == sieve_dim
    assert all( abs( t_gs_reduced[m-sieve_dim:] ) <0.501 )
    t_gs_shift = from_canonical_scaled( GSO,shift_babai,offset=sieve_dim )

    #print("t_gs_reduced:", t_gs_reduced)


    #check if Babai was successful
    #TODO!


    dbsize_start = g6k.db_size()
    nrand_, _ = batchCVPP_cost(sieve_dim,100,dbsize_start**(1./sieve_dim),1)
    nrand = math.ceil(1./nrand_)+100

    scaling_vec = np.array( [tmp**0.5 for tmp in GSO.r()[m-sieve_dim:]] )


    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(1);
    slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

    #To be properly done on the C-lvl
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    try:
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], 1.8) #TODO:set up 1.8 correctly
        iterator = slicer.itervalues_t()
        for tmp in iterator:
            out_gs_reduced = np.array(tmp)  #cdb[0]
            break

        print("out_gs_reduced:", out_gs_reduced)
        #print(len(c))
        #print(len(to_canonical_scaled( GSO, np.concatenate( [(m-sieve_dim)*[0], out_gs_reduced] ) )))
        c = np.array(c)
        c_new = c - to_canonical_scaled( GSO, np.concatenate( [(m-sieve_dim)*[0], out_gs_reduced] ) )
        bab_01 = GSO.babai(c_new)
        print(bab_01)

        #shift_babai_c_reduced = GSO.babai((m-sieve_dim)*[0] + list(t_gs_non_scaled), start=dim-sieve_dim,gso=True)
        #shift_babai_reduced = GSO.B.multiply_left( (m-sieve_dim)*[0] + list( shift_babai_c_reduced ) )
        #shift_babai_reduced = shift_babai_reduced[m-sieve_dim:]
        #shift_babai_reduced *= scaling_vec
        #t_gs_bab = np.array(t_gs_reduced) - shift_babai_reduced


    except Exception as e: print(f" - - - {e} - - -")





















