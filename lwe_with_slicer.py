
import sys

from fpylll import BKZ as fplll_bkz, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.siever import Siever
from g6k.slicer import RandomizedSlicer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from utils import *




if __name__=='__main__':
    fpylll_crossover = 55
    goal_margin = 1.5
    bkz_tours = 2

    alpha = 0.01
    n = 40
    A, c, q = load_lwe_challenge(n=n, alpha=alpha)


    try:
        min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                    samples=A.nrows, decouple=True)
        print(min_cost_param)
        (b, s, m) = min_cost_param #bkz_dim, svp_dim, num_samples
    except TypeError:
        raise TypeError("No winning parameters.")

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
    sieve_dim = s - 5

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


    #To be properly done on the C-lvl
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))


    #project the target and babai-reduced it
    t = [ int(cc) for cc in c ]
    G = g6k.M
    t_gs_non_scaled = G.from_canonical(t)[-sieve_dim:]
    shift_babai_c = G.babai((g6k.full_n-sieve_dim)*[0] + list(t_gs_non_scaled), start=g6k.full_n-sieve_dim,gso=True)
    shift_babai = G.B.multiply_left( (g6k.full_n-sieve_dim)*[0] + list( shift_babai_c ) )
    t_gs_reduced = from_canonical_scaled( G,np.array(t)-shift_babai,offset=sieve_dim ) #TODO: this line fails, check the dimensions
    t_gs_shift = from_canonical_scaled( G,shift_babai,offset=sieve_dim )















