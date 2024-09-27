
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
    g6k.initialize_local(g6k.full_n-sieve_dim,g6k.full_n-sieve_dim,g6k.full_n)
    g6k.M.update_gso()

    print("starting bdgl2...")
    g6k(alg="bdgl2")
    print("finished bdgl2...")


    #project the target and babai-reduced it

    c = c[:m]
    GSO = g6k.M

    t_gs_non_scaled = GSO.from_canonical(c)[-sieve_dim:]
    shift_babai_c = GSO.babai((g6k.full_n-sieve_dim)*[0] + list(t_gs_non_scaled), gso=True)

    print(shift_babai_c)
    babai_coeff = GSO.babai(c)
    print(GSO.babai(c))


    shift_babai = GSO.B.multiply_left( (g6k.full_n-sieve_dim)*[0] + list( shift_babai_c ) )
    t_gs_reduced = from_canonical_scaled( GSO,np.array(c)-shift_babai,offset=sieve_dim )
    t_gs_shift = from_canonical_scaled( GSO,shift_babai,offset=sieve_dim )

    #print("t_gs_reduced:", t_gs_reduced)


    #check if Babai was successful
    #TODO!


    dbsize_start = g6k.db_size()
    nrand_, _ = batchCVPP_cost(sieve_dim,100,dbsize_start**(1./sieve_dim),1)
    nrand = math.ceil(1./nrand_)+100

    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(2);
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
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], 10)
        iterator = slicer.itervalues_t()
        for tmp in iterator:
            out_gs_reduced = tmp  #cdb[0]
            break
        out_gs = out_gs_reduced + t_gs_shift

        out = to_canonical_scaled( G,out_gs,offset=sieve_dim )
        N = GSO.Mat( GSO.B[:g6k.full_n-sieve_dim], float_type=ft )
        N.update_gso()
        bab_1 = GSO.babai(c-np.array(out),start=g6k.full_n-sieve_dim) #last sieve_dim coordinates of s
        tmp = c - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
        tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=g6k.full_n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
        bab_0 = N.babai(tmp)

        bab_01 =  np.array( bab_0+bab_1 ) #shifted answer. Good since it is smaller, thus less rounding error
        bab_01 += np.array(shift_babai_c)
        print(f"Success: {all(c==bab_01)}")


    except Exception as e: print(f" - - - {e} - - -")





















