from __future__ import absolute_import
from __future__ import print_function
import copy
import re
import sys, os
import time

from collections import OrderedDict # noqa
from math import log

from fpylll import BKZ as fplll_bkz
from fpylll import IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.algorithms.pump import pump
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.utils.cli import parse_args, run_all, pop_prefixed_params
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from six.moves import range
import numpy as np

from sample import Distribution, centeredBinomial
from primal_kyber import gen_and_dump_lwe, load_lwe
from utils import get_filename

import pickle
from global_consts import *
import argparse
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

def lwe_kernel(params=None, seed=None):
    """
    Run the primal attack against Darmstadt LWE instance (n, alpha).

    :param n: the dimension of the LWE-challenge secret
    :param params: parameters for LWE:

        - lwe/alpha: the noise rate of the LWE-challenge

        - lwe/m: the number of samples to use for the primal attack

        - lwe/goal_margin: accept anything that is
          goal_margin * estimate(length of embedded vector)
          as an lwe solution

        - lwe/svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - bkz/blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - bkz/tours: the number of tours to do for each blocksize

        - bkz/jump: the number of blocks to jump in a BKZ tour after
          each pump

        - bkz/extra_dim4free: lift to indices extra_dim4free earlier in
          the lattice than the currently sieved block

        - bkz/fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - bkz/dim4free_fun: in blocksize x, try f(x) dimensions for free,
          give as 'lambda x: f(x)', e.g. 'lambda x: 11.5 + 0.075*x'

        - pump/down_sieve: sieve after each insert in the pump-down
          phase of the pump

        - dummy_tracer: use a dummy tracer which captures less information

        - verbose: print information throughout the lwe challenge attempt

    """

    params = copy.copy(params)
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"]
    seed = params["seed"]
    
    match dist:
        case "binomial":
            dist_param = int(dist_param)
            distrib = centeredBinomial(dist_param)
        case "ternary":
            print(f"dist_param: {dist_param}")
            distrib = ternaryDist(dist_param)
        case "ternary_sparse":
            distrib = centeredBinomial(dist_param)
        case _:
            raise NotImplementedError(f"Bad distribution")

    alpha = distrib.variance**0.5/q

    # -------------------------------- preparing --------------------------------
    try: #try load lwe instance
        A, q, bse = load_lwe(params) #D["A"], D["q"], D["bse"]
    except FileNotFoundError: #if no such, create one
        print(f"No kyber instance found... generating.")
        gen_and_dump_lwe(params) #ntar = 5
        A, q, bse = load_lwe(params) #D["A"], D["q"], D["bse"]


    B = [ [int(0) for i in range(2*n)] for j in range(2*n) ]
    for i in range( n ):
        B[i][i] = int( q )
    for i in range(n, 2*n):
        B[i][i] = 1
    for i in range(n, 2*n):
        for j in range(n):
            B[i][j] = int( A[i-n,j] )

    B = IntegerMatrix.from_matrix( B )
    b_, s, e = bse[seed[1]]

    c = ( np.array(A@(s)) + np.array(e) )%q #the target
    sec = np.concatenate([e,-s,[1]])

    goal_margin = params["goal_margin"]
    target_norm = goal_margin * (sec@sec)

    # --------------------------------end preparing -------------------------------

    # params for underlying BKZ
    extra_dim4free = params["extra_dim4free"]
    dim4free_fun = "default_dim4free_fun"
    jump = params["jump"]
    pump_params = {} #pop_prefixed_params("pump", params)
    fpylll_crossover = params["fpylll_crossover"]
    blocksizes = params["blocksizes"]
    tours = params["tours"]
    jump = params["jump"]

    # flow of the lwe solver
    svp_bkz_time_factor = params["svp_bkz_time_factor"]

    # generation of lwe instance and Kannan's embedding

    m = params["m"]
    decouple = svp_bkz_time_factor > 0

    # misc
    dont_trace = True #params["dummy_tracer"]
    verbose = params["verbose"]

    # A, c, q = load_lwe_challenge(n=n, alpha=alpha)
    print("-------------------------")
    print("Primal attack, LWE challenge n=%d, alpha=%.4f" % (n, alpha))

    if m is None:
        try:
            min_cost_param = gsa_params(n=len(A), alpha=alpha, q=q,
                                        samples=len(A[0]), d=2*n, decouple=decouple)
            (b, s, m) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    else:
        try:
            min_cost_param = gsa_params(n=len(A), alpha=alpha, q=q, samples=m, d=2*n,
                                        decouple=decouple)
            (b, s, _) = min_cost_param
        except TypeError:
            raise TypeError("No winning parameters.")
    print("Chose %d samples. Predict solution at bkz-%d + svp-%d" % (m, b, s))
    print()

    if blocksizes is not None:
        blocksizes = list(range(10, 40)) + list( eval("range(%s)" % re.sub(":", ",", blocksizes)) ) # noqa
    else:
        blocksizes = list(range(10, 50)) + [b-20, b-17] + list(range(b - 14, b + 25, 2))

    A = IntegerMatrix.from_matrix(A)
    # B = primal_lattice_basis(A, c, q, m=m)
    B = [ [int(0) for i in range(2*n)] for j in range(2*n) ]
    for i in range( n ):
        B[i][i] = int( q )
    for i in range(n, 2*n):
        B[i][i] = 1
    for i in range(n, 2*n):
        for j in range(n):
            B[i][j] = int( A[i-n,j] )

    B = [ [ bb for bb in b ]+[0] for b in B ] + [ (2*n)*[0] + [1] ]
    for j in range(n):
        B[-1][j] = int( c[j] )

    B = IntegerMatrix.from_matrix( B )

    T_overall_0 = time.time()
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k = Siever(B, param_sieve)
    print("GSO precision: ", g6k.M.float_type)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("lwe"), start_clocks=True)

    d = g6k.full_n
    blocksizes = [blocksize for blocksize in blocksizes if blocksize <= d]
    g6k.lll(0, g6k.full_n)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    T0 = time.time()
    T0_BKZ = time.time()
    for blocksize in blocksizes:
        for tt in range(tours):
            # BKZ tours

            if blocksize < fpylll_crossover:
                if verbose:
                    print("Starting a fpylll BKZ-%d tour. " % (blocksize), end=' ')
                    sys.stdout.flush()
                bkz = BKZReduction(g6k.M)
                par = fplll_bkz.Param(blocksize,
                                      strategies=fplll_bkz.DEFAULT_STRATEGY,
                                      max_loops=1)
                bkz(par)
                print(f"basis_quality: {basis_quality(bkz.M)}")

            else:
                if verbose:
                    print("Starting a pnjBKZ-%d tour. " % (blocksize), flush=True)

                pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                     verbose=verbose,
                                     extra_dim4free=extra_dim4free,
                                     dim4free_fun=dim4free_fun,
                                     goal_r0=target_norm,
                                     pump_params=pump_params)
                print(f"basis_quality: {basis_quality(bkz.M)}")

            T_BKZ = time.time() - T0_BKZ

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))

            g6k.lll(0, g6k.full_n)

            if g6k.M.get_r(0, 0) <= target_norm:
                break

            # overdoing n_max would allocate too much memory, so we are careful
            svp_Tmax = svp_bkz_time_factor * T_BKZ
            n_max = int(58 + 2.85 * log(svp_Tmax * nthreads)/log(2.))

            rr = [g6k.M.get_r(i, i) for i in range(d)]
            continue_flag = False
            for n_expected in range(2, d-2):
                x = (target_norm/goal_margin) * n_expected/(1.*d)
                # if 4./3 * gaussian_heuristic(rr[d-n_expected:]) > x:
                #     break
                if 0.96 * gaussian_heuristic(rr[d-n_expected:]) > x: #the estimation above is not for BDD
                    break

            #but underdoing won`t solve the bdd instance at all
            if 1.02 * gaussian_heuristic(rr[d-n_expected:]) < x: #the estimation above is not for BDD
                print(f"Solution unlikely: {1.02 * gaussian_heuristic(rr[d-n_expected:])} < {x}")
                continue

            print("Without otf, would expect solution at pump-%d. n_max=%d in the given time." % (n_expected, n_max)) # noqa
            if n_expected >= n_max - 1:
                continue

            n_max += 1

            # Larger SVP

            llb = d - blocksize
            while gaussian_heuristic([g6k.M.get_r(i, i) for i in range(llb, d)]) < target_norm * (d - llb)/(1.*d): # noqa
                llb -= 1
                if llb < 0:
                    break

            # catch small cases where selections above give nonsensical suggestions
            llb = max(0, llb)
            f = max(d-llb-n_max, 0)

            if verbose:
                print("Starting svp pump_{%d, %d, %d}, n_max = %d, Tmax= %.2f sec" % (llb, d-llb, f, n_max, svp_Tmax)) # noqa
            pump(g6k, tracer, llb, d-llb, f, verbose=verbose,
                 goal_r0=target_norm * (d - llb)/(1.*d))

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "\n slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))
                print()

            g6k.lll(0, g6k.full_n)
            T0_BKZ = time.time()
            if g6k.M.get_r(0, 0) <= target_norm:
                break

        if g6k.M.get_r(0, 0) <= target_norm:
            print("Finished! TT=%.2f sec" % (time.time() - T0))
            print(g6k.M.B[0])
            alpha_ = int(alpha*1000)
            filename = 'lwechallenge/%03d-%03d-solution.txt' % (n, alpha_)
            fn = open(filename, "w")
            fn.write(str(g6k.M.B[0]))
            fn.close()
            T_overall = T_overall_0 - time.time()
            return True, T_overall, T_BKZ 
    T_overall = T_overall_0 - time.time()
    return False, T_overall, T_BKZ

def get_parser():
    parser = argparse.ArgumentParser(
        description="Experiments for primal attack."
    )
    parser.add_argument(
    "--nthreads", default=1, type=int, help="Threads per slicer."
    )
    parser.add_argument(
    "--nworkers", default=1, type=int, help="Workers for experiments."
    )
    parser.add_argument(
    "--inst_per_lat", default=1, type=int, help="Number of instances per lattice."
    )
    parser.add_argument(
    "--lats_per_dim", default=1, type=int, help="Number of lattices."
    )
    parser.add_argument(
    "--n", default= 144, type=int, help="LWE dimension."
    )
    parser.add_argument(
    "--m", default= 144, type=int, help="LWE ambient dimension."
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
    "--betapre", default=45, type=int, help="Preprocessing BKZ blocksize."
    )
    parser.add_argument(
    "--blocksizes", default="50:61:5", type=str, help="Upper bound on the BKZ blocksize."
    )
    parser.add_argument(
    "--tours", default=5, type=int, help="BKZ tours"
    )
    parser.add_argument(
    "--jump", default=1, type=int, help="BKZ jump"
    )
    parser.add_argument(
    "--extra_dim4free", default=12, type=int, help="Upper bound on the BKZ blocksize."
    )
    parser.add_argument(
    "--fpylll_crossover", default=55, type=int, help="Upper bound on the BKZ blocksize."
    ) 
    parser.add_argument(
    "--svp_bkz_time_factor", default=1.0, type=float, help="svp_bkz_time_factor (as float)"
    )
    parser.add_argument(
    "--goal_margin", default=1., type=float, help="goal_margin (as float)"
    ) 

    parser.add_argument("--recompute_instance", action="store_true", help="Recomputes instances. WARNING deletes previous instance irreversibly.")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

if __name__ == "__main__":
    out_path = "lwe_instances/reduced_lattices/"
    isExist = os.path.exists(out_path)
    if not isExist:
        try:
            os.makedirs(out_path)
        except:
            pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.
    
    parser = get_parser()
    args = parser.parse_args()

    nthreads = args.nthreads
    nworkers = args.nworkers
    lats_per_dim = args.lats_per_dim
    inst_per_lat = args.inst_per_lat #10 #how many instances per A, q
    dist, dist_param = args.dist, args.dist_param
    q = args.q
    n = args.n

    output = []
    pool = Pool( processes = nworkers )
    tasks = []
    RECOMPUTE_INSTANCE = args.recompute_instance
    RECOMPUTE_KYBER = False
    if RECOMPUTE_INSTANCE:
        print(f"Generating Kyber...")
        for latnum in range(lats_per_dim):
            params = {
                "n": n,
                "q": q,
                "m": args.m,
                "dist": dist,
                "dist_param": dist_param,
                "ntar": inst_per_lat,
                "blocksizes": args.blocksizes,
                "tours": args.tours,
                "extra_dim4free": args.extra_dim4free,
                "fpylll_crossover": args.fpylll_crossover,
                "seed": [latnum,0],
                "nthreads": nthreads
            }
            gen_and_dump_lwe(params)

    for latnum in range(lats_per_dim):
            for tstnum in range(inst_per_lat):
                print("lol")
                params = {
                "n": n,
                "q": q,
                "m": args.m,
                "dist": dist,
                "dist_param": dist_param,
                "ntar": inst_per_lat,
                "blocksizes": args.blocksizes,
                "tours": args.tours,
                "jump": args.jump,
                "extra_dim4free": args.extra_dim4free,
                "fpylll_crossover": args.fpylll_crossover,
                "svp_bkz_time_factor": args.svp_bkz_time_factor,
                "goal_margin": args.goal_margin,
                "seed": [latnum,tstnum],
                "nthreads": nthreads,
                "verbose": True
                }
                tasks.append( pool.apply_async(
                    lwe_kernel, ( params,None )
                    ) )
                
    for t in tasks:
            output.append( t.get() )

    pool.close()

    print(output)
    