"""
Invoke as:
python lwe_g6k.py --nthreads 2 --nworkers 2 --inst_per_lat 2 --lats_per_dim 2 --n 125 --m 125 --dist "binomial" --dist_param 3 --use_pnj_strat_instead --extra_dim4free 3 --recompute_instance --goal_margin 1.05 --svp_bkz_time_factor 1.0 --verbose
"""

from __future__ import absolute_import
from __future__ import print_function
import warnings

warnings.filterwarnings("ignore", message=".*Dimension of lattice is larger than.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API..*")
import copy
import re
import sys, os
import time

from collections import OrderedDict # noqa
from math import log

from fpylll import BKZ as fplll_bkz
from fpylll import IntegerMatrix, GSO
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

from sample import Distribution, centeredBinomial, ternaryDist
from preprocessing import gen_and_dump_lwe, load_lwe
from utils import get_filename
from pnj_strat import strats_kyber

import pickle
from global_consts import *
import argparse
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

def lwe_kernel(params=None, seed=None, my_tracer={}):
    """
    Run the primal attack against an LWE instance.

    :param params: parameters for LWE:

        - n: the dimension of the LWE-challenge secret
    
        - q: the LWE modulus

        - m: the number of samples to use for the primal attack

        - dist: secret distribution

        - blocksizes: given as low:high:inc perform BKZ reduction
          with blocksizes in range(low, high, inc) (after some light)
          prereduction

        - tours: the number of tours to do for each blocksize

        - jump: the number of blocks to jump in a BKZ tour after
          each pump

        - ntar: numper of LWE instances per lattice

        - fpylll_crossover: use enumeration based BKZ from fpylll
          below this blocksize

        - svp_bkz_time_factor: if > 0, run a larger pump when
          svp_bkz_time_factor * time(BKZ tours so far) is expected
          to be enough time to find a solution

        - goal_margin: an approximation factor

        - nthreads: number of threads allocated to the sieve and bkz

        - verbose: print information throughout the lwe challenge attempt

    """
    my_tracer = {
                    "bkz_invoked": {},
                    "svp_calls": [],
                    "succ": False,
                    "T_overall": 0,
                    "T_BKZ": 0
                }

    params = copy.copy(params)
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"]
    seed = params["seed"]
    nthreads = params["nthreads"]
    
    match dist:
        case "binomial":
            dist_param = int(dist_param)
            distrib = centeredBinomial(dist_param)
        case "ternary":
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

    b_, s, e = bse[seed[1]]
    c = np.concatenate([b_,[0]*n])

    B = [ [ bb for bb in b ]+[0] for b in B ] + [ (2*n)*[0] + [1] ]

    for j in range(n):
        B[-1][j] = int( c[j] )

    B = IntegerMatrix.from_matrix( B )
    
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

    print("-------------------------")
    print("Primal attack, LWE instance n=%d" % (n,))

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
    U=IntegerMatrix.identity(B.nrows)
    UinvT=IntegerMatrix.identity(B.nrows)
    G=GSO.Mat(B,float_type="dd",U=U,UinvT=UinvT)
    
    g6k = Siever(G, param_sieve)

    if dont_trace:
        tracer = dummy_tracer
    else:
        tracer = SieveTreeTracer(g6k, root_label=("lwe"), start_clocks=True)

    d = g6k.full_n
    blocksizes = [blocksize for blocksize in blocksizes if blocksize <= d]
    g6k.lll(0, g6k.full_n)
    slope = basis_quality(g6k.M)["/"]
    print("Intial Slope = %.5f\n" % slope)

    use_pnj_strat_instead = params["use_pnj_strat_instead"]
    # if not use_pnj_strat_instead:
    iter_strat = [ (tmp,jump,tours) for tmp in blocksizes ]
    # else:
    #     iter_strat = [ (tmp,1,2) for tmp in range(5,46) ] + strats_kyber[(dist,dist_param)][n]

    T0 = time.time()
    T0_BKZ = time.time()
    cntr = 0
    for blocksize, jump, ntours in iter_strat:
        for tt in range(ntours):
            T_tour_0 = time.time()
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

            else:
                if verbose or blocksize>69:
                    print("Starting a pnjBKZ-%d-%d tour. " % (blocksize,jump))

                pump_n_jump_bkz_tour(g6k, tracer, blocksize, jump=jump,
                                     verbose=verbose,
                                     extra_dim4free=extra_dim4free,
                                     dim4free_fun=dim4free_fun,
                                     goal_r0=target_norm,
                                     pump_params=pump_params)

            T_tour = time.time() - T_tour_0
            if not blocksize in my_tracer["bkz_invoked"].keys():
                my_tracer["bkz_invoked"][blocksize] = {"iters": 1, "times":[T_tour]}
            else:
                my_tracer["bkz_invoked"][blocksize]["iters"]+=1
                my_tracer["bkz_invoked"][blocksize]["times"] += [T_tour]

            T_BKZ = time.time() - T0_BKZ

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))

            g6k.lll(0, g6k.full_n)

            if g6k.M.get_r(0, 0) <= 1.01 * target_norm:
                break

            # overdoing n_max would allocate too much memory, so we are careful
            svp_Tmax = svp_bkz_time_factor * T_BKZ
            n_max = int(58 + 2.85 * log(svp_Tmax * nthreads)/log(2.))

            rr = [g6k.M.get_r(i, i) for i in range(d)]
            
            for n_expected in range(2, d-2):
                x = (target_norm/goal_margin) * n_expected/(1.*d)
                # if 4./3 * gaussian_heuristic(rr[d-n_expected:]) > x:
                #     break
                if 1.02 * gaussian_heuristic(rr[d-n_expected:]) > x: #the estimation above is not for BDD
                    break

            #but underdoing won`t solve the bdd instance at all
            # if 1.02 * gaussian_heuristic(rr[d-n_expected:]) < x: #the estimation above is not for BDD
            #     print(f"Solution unlikely: {1.02 * gaussian_heuristic(rr[d-n_expected:])} < {x}")
            #     continue

            if verbose:
                print("Without otf, would expect solution at pump-%d. n_max=%d in the given time." % (n_expected, n_max)) # noqa
            if (n_expected >= n_max - 1 and not cntr>=len(iter_strat)-1) or n_expected<45:
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

            if f>12:
                print(f"Was about to svp pump_{llb, d-llb, f, n_max}... Aborted")
                continue

            if verbose:
                print("Starting svp pump_{%d, %d, %d}, n_max = %d, Tmax= %.2f sec" % (llb, d-llb, f, n_max, svp_Tmax)) # noqa
            T_sieve = time.time()
            pump(g6k, tracer, llb, d-llb, f, verbose=verbose,
                 goal_r0=target_norm * (d - llb)/(1.*d))
            T_sieve = time.time() - T_sieve
            my_tracer["svp_calls"].append( (d-llb, T_sieve) )

            if verbose:
                slope = basis_quality(g6k.M)["/"]
                fmt = "\n slope: %.5f, walltime: %.3f sec"
                print(fmt % (slope, time.time() - T0))
                print()

            g6k.lll(0, g6k.full_n)
            T0_BKZ = time.time()
            if g6k.M.get_r(0, 0) <= target_norm:
                break
        cntr+=1
        if g6k.M.get_r(0, 0) <= target_norm:
            print("Finished! TT=%.2f sec" % (time.time() - T0))
            print(f"Solution: {g6k.M.B[0]}")
            alpha_ = int(alpha*1000)
            T_overall = time.time() - T_overall_0
            my_tracer["succ"] = True
            my_tracer["T_BKZ"] = T_BKZ
            my_tracer["T_overall"] = T_overall
            return my_tracer
    T_overall = T_overall_0 - time.time()
    my_tracer["T_BKZ"] = T_BKZ
    my_tracer["T_overall"] = T_overall
    print(f"FAIL: basis_quality: {basis_quality(bkz.M)}")
    return my_tracer

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
    "--blocksizes", default="50:61:5", type=str, help="Bounds on the BKZ blocksize."
    )
    parser.add_argument(
    "--tours", default=5, type=int, help="BKZ tours"
    )
    parser.add_argument(
    "--jump", default=1, type=int, help="BKZ jump"
    )
    parser.add_argument("--use_pnj_strat_instead", action="store_true", help="Overrides blocksizes tours and jump. Uses strategies from pnj_strat.py instead.")
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
    my_tracers = []
    for latnum in range(lats_per_dim):
            for tstnum in range(inst_per_lat):

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
                "use_pnj_strat_instead": args.use_pnj_strat_instead,
                "extra_dim4free": args.extra_dim4free,
                "fpylll_crossover": args.fpylll_crossover,
                "svp_bkz_time_factor": args.svp_bkz_time_factor,
                "goal_margin": args.goal_margin,
                "seed": [latnum,tstnum],
                "nthreads": nthreads,
                "verbose": args.verbose,
                }
                tasks.append( pool.apply_async(
                    lwe_kernel, ( params,None )
                    ) )
                
    for t in tasks:
            my_tracers.append( t.get() )

    pool.close()

    filename = f"exp_{n}.pkl" if not args.use_pnj_strat_instead else f"exp_{n}_pnj.pkl"
    with open(out_path + filename,"wb") as file:
        pickle.dump(my_tracers,file)

    print( "Results dumped to "+ out_path + filename )
    print( my_tracers )
    