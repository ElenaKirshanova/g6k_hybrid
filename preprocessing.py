import sys,os
import time
import argparse
from fpylll import *
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from utils import *
from lattice_reduction import LatticeReduction
from copy import deepcopy
from primal_kyber import gen_and_dump_lwe

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from global_consts import *
from utils import get_filename

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"
does_exist = os.path.exists(inp_path)
if not does_exist:
    sys.exit('cannot find path for input lattices')

os.makedirs(out_path,exist_ok=True)


def load_lwe(params):
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"]
    seed = params["seed"][0]
    print(f"- - - n,seed={n,seed} - - - load")
    filename = f"lwe_instance_{dist}_{dist_param}_{n}_{q}_{dist_param:.04f}_{seed}"
    filename = get_filename( "lwe_instance", params )
    with open(inp_path + filename, "rb") as fl:
        D = pickle.load(fl)
    A_, q_,  bse_ = D["A"], D["q"], D["bse"]
    return A_, q_, bse_


def run_preprocessing(params):
    n, q, dist, dist_param = params["n"], params["q"], params["dist"], params["dist_param"]
    seed,beta_bkz,sieve_dim_max,nsieves,kappa = params["seed"],params["beta_bkz"],params["sieve_dim_max"],params["nsieves"],params["kappa"]
    beta_bkz_offset = params["beta_bkz_offset"]
    nthreads=N_SIEVE_THREADS
    dump_bkz=True
    report = {
        "params": (n,q,dist,dist_param,seed),
        "beta_bkz": beta_bkz,
        "sieve_dim_max": sieve_dim_max,
        "sieve_dim_min": sieve_dim_max-nsieves,
        "kappa": kappa,
        "bkz_runtime": 0,
        "bdgl_runtime": [0]*(nsieves+1),
    }
    A, q, bse = load_lwe(params) #D["A"], D["q"], D["bse"]

    B = [ [int(0) for i in range(2*n)] for j in range(2*n) ]
    for i in range( n ):
        B[i][i] = int( q )
    for i in range(n, 2*n):
        B[i][i] = 1
    for i in range(n, 2*n):
        for j in range(n):
            B[i][j] = int( A[i-n,j] )

    if sieve_dim_max<60:
        nthreads = 1
    elif sieve_dim_max<80:
        nthreads = 2


    H11 = B[:len(B)-kappa] #the part of basis to be reduced
    H11 = IntegerMatrix.from_matrix( [ h11[:len(B)-kappa] for h11 in H11  ] )
    H11r, H11c = H11.nrows, H11.ncols
    assert(H11r==H11c)

    LR = LatticeReduction( H11, threads_bkz=nthreads )
    bkz_start = time.perf_counter()
    for beta in range(5,beta_bkz+1):
        then_round=time.perf_counter()
        LR.BKZ(beta)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
        sys.stdout.flush()
    
    report["bkz_runtime"] = [ time.perf_counter() - bkz_start ]

    for beta in range(beta_bkz, beta_bkz+beta_bkz_offset):
        then_round=time.perf_counter()
        LR.BKZ(beta)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time} seed {seed[0]}")
        sys.stdout.flush()
        report["bkz_runtime"].append( time.perf_counter() - then_round )

        H11 = deepcopy( LR.basis ) #precautinary measure -- g6k = Siever(G,param_sieve) may call LLL on H11 = LR.basis and alter LR.basis???
        #---------run sieving------------
        int_type = H11.int_type
        FPLLL.set_precision(210)
        ft = "dd" if config.have_qd else "mpfr"
        G = GSO.Mat( H11, U=IntegerMatrix.identity(H11r,int_type=int_type), UinvT=IntegerMatrix.identity(H11r,int_type=int_type), float_type=ft )
        G.update_gso()
        param_sieve = SieverParams()
        param_sieve['threads'] = nthreads
        param_sieve['otf_lift'] = False
        g6k = Siever(G,param_sieve)
        g6k.initialize_local(H11r-sieve_dim_max, H11r-sieve_dim_max+nsieves ,H11r)

        sieve_start = time.perf_counter()
        g6k(alg="bdgl2")
        i = 0
        report["bdgl_runtime"][i] = time.perf_counter()-sieve_start
        print(f"siever-{seed[0]}-{kappa}-{sieve_dim_max-nsieves+i} for beta={beta} finished in added time {time.perf_counter()-sieve_start}\n" )
        sys.stdout.flush()
        #NOTE: this dumps
        assert g6k.r - g6k.l == sieve_dim_max-nsieves+i, f"g6k context: {g6k.r - g6k.l} != {sieve_dim_max-nsieves+i}"
        g6kdumppath = f'g6kdump_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{kappa}_{g6k.n}_{beta}.pkl'
        g6k.dump_on_disk(out_path+g6kdumppath)
        for i in range(1,nsieves+1):
            g6k.extend_left(1)
            sieve_start = time.perf_counter()
            g6k(alg="bdgl2")
            report["bdgl_runtime"][i] = time.perf_counter()-sieve_start
            print(f"siever-{seed[0]}-{kappa}-{sieve_dim_max-nsieves+i} for beta={beta} finished in added time {time.perf_counter()-sieve_start}\n", flush=True )
            sys.stdout.flush()
            #NOTE: this dumps
            assert g6k.r - g6k.l == sieve_dim_max-nsieves+i, f"g6k context: {g6k.r - g6k.l} != {sieve_dim_max-nsieves+i}"
            g6kdumppath = f'g6kdump_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{kappa}_{g6k.n}_{beta}.pkl'
            g6k.dump_on_disk(out_path+g6kdumppath)


    print(report)
    sys.stdout.flush()
    return report

def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocessing for hybrid attack."
    )
    parser.add_argument(
    "--nthreads", default=N_SIEVE_THREADS, type=int, help="Threads per slicer."
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
    "--params", default= "[ (125,2,46) ]", type=str, help="String that evaluates to the list of triples (n, n_guess_coordinates, bkzbeta)."
    )
    parser.add_argument(
    "--q", default=3329, type=int, help="LWE modulus"
    )
    parser.add_argument(
    "--dist", default="binomial", type=str, help="LWE distribution"
    )
    parser.add_argument(
    "--dist_param", default=2.0, type=float, help="LWE distribution's parameter (as float). For binomial should be an integer for ternary should be in (0,1/2)."
    )
    parser.add_argument(
    "--beta_bkz_offset", default=1, type=int, help="BKZ-beta reduced bases will be computed for beta in [sieve_dim,...,sieve_dim+beta_bkz_offset]."
    )
    parser.add_argument(
    "--sieve_dim_max_offset", default=1, type=int, help="he largest slicer will work on dim=predicted beta + this offset."
    )
    parser.add_argument(
    "--nsieves", default=1, type=int, help="Number of sieves performed."
    )
    parser.add_argument("--recompute_instance", action="store_true", help="Recomputes instances. WARNING deletes previous instance irreversibly.")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

if __name__=="__main__":
    # (dimension, predicted kappa, predicted beta)
    parser = get_parser()
    args = parser.parse_args()

    params = [ i for i in eval(args.params) ]
    nworkers, nthreads =  args.nworkers, args.nworkers #5 (to be changed for kyber 190, 200 !!!)

    beta_bkz_offset = args.beta_bkz_offset #
    sieve_dim_max_offset = args.sieve_dim_max_offset 

    lats_per_dim = args.lats_per_dim
    inst_per_lat = args.inst_per_lat #how many instances per A, q
    dist, dist_param = args.dist, args.dist_param
    q = args.q
    output = []
    pool = Pool(processes = nworkers )

    RECOMPUTE_INSTANCE = args.recompute_instance
    RECOMPUTE_KYBER = True
    if RECOMPUTE_INSTANCE:
        print(f"Generating Kyber...")
        for n in [pp[0] for pp in params]:
            for latnum in range(lats_per_dim):
                params_ = {
                    "n": n,
                    "q": q,
                    "dist": dist,
                    "dist_param": dist_param,
                    "ntar": inst_per_lat,
                    "seed": [latnum,0],
                    "nthreads": nthreads
                }
                gen_and_dump_lwe(params_)

    tasks = []
    for param in params:
        for latnum in range(lats_per_dim):
            kappa = param[1]
            params_inp ={
                    "n": param[0], #n
                    "q": q, #q
                    "dist": dist,
                    "dist_param": dist_param,
                    "seed": [latnum,0], #seed, second value is irrelevant
                    "beta_bkz": param[2], #beta_bkz
                    "beta_bkz_offset": beta_bkz_offset,
                    "sieve_dim_max": param[2]+sieve_dim_max_offset, #sieve_dim_max
                    "nsieves": args.nsieves,  #nsieves
                    "kappa": kappa, #kappa
                    "nthreads": nthreads, #nthreads
                }
            tasks.append( pool.apply_async(
                run_preprocessing, (params_inp,)
            ) )

    for t in tasks:
        output.append( t.get() )
    pool.close()

    for o_ in output:
        print(o_)
        n,q,dist, dist_param,seed = o_["params"]
        kappa = o_["kappa"]
        beta_bkz = o_["beta_bkz"]
        sieve_dim_max = o_["sieve_dim_max"]
        sieve_dim_min = o_["sieve_dim_min"]
        filename = f"report_prehyb_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}_{beta_bkz}.pkl" if dist=="ternary" else f"report_prehyb_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}_{beta_bkz}.pkl"
        filename = out_path + filename

        with open(filename, "wb") as file:
            pickle.dump( o_,file )

    sys.stdout.flush()
