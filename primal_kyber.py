from experiments.lwe_gen import *

import os
import time
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from math import log

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

try:
  from g6k import SieverParams
except ImportError:
  raise ImportError("g6k not installed")

from lattice_reduction import LatticeReduction
from experiments.lwe_gen import *
from utils import get_filename

import pickle
from global_consts import *
import argparse

from signal import signal, SIGPIPE, SIG_DFL  
signal(SIGPIPE,SIG_DFL) 

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"

def flatter_interface( fpylllB ):
    flatter_is_installed = os.system( "flatter -h > /dev/null" ) == 0

    if flatter_is_installed:
        basis = '[' + fpylllB.__str__() + ']'
        seed = randrange(2**32)
        filename = f"lat{seed}.txt"
        filename_out = f"redlat{seed}.txt"
        with open(filename, 'w') as file:
            file.write( "["+fpylllB.__str__()+"]" )

        out = os.system( "flatter " + filename + " > " + filename_out )
        time.sleep(float(0.05))
        os.remove( filename )

        B = IntegerMatrix.from_file( filename_out )
        os.remove( filename_out )
    else:
        print("Flatter issues")
    return B

def gen_and_dump_lwe(params):
    # n, q, dist, dist_param,  ntar, seed=0
    n = params["n"]
    q = params["q"]
    ntar = params["ntar"]
    dist = params["dist"] 
    dist_param = params["dist_param"] if dist!="binomial" else int(params["dist_param"])
    seed = params["seed"][0]
    print(f"- - - n,seed={n,seed} - - - gen")
    A,q,bse= generateLWEInstances(n, q, dist, dist_param, ntar)

    # filename = f"lwe_instance_{dist}_{n}_{q}_{dist_param:.04f}_{seed}"
    filename = get_filename( "lwe_instance", params )
    with open(inp_path + filename, "wb") as fl:
        pickle.dump({"A": A, "q": q, "dist": dist, "dist_param":dist_param,  "bse": bse}, fl)

def load_lwe(params):
    # n,q,dist,dist_param,seed=0
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"]
    seed = params["seed"][0]
    print(f"- - - n,seed={n,seed} - - - load")
    filename = f"lwe_instance_{dist}_{n}_{q}_{dist_param:.04f}_{seed}"
    filename = get_filename( "lwe_instance", params )
    with open(inp_path + filename, "rb") as fl:
        D = pickle.load(fl)
    A_, q_,  bse_ = D["A"], D["q"], D["bse"]
    return A_, q_, bse_

def prepare_kyber(params): #for debug purposes
    # n,q,dist,dist_param,betapre,seed=[0,0], nthreads=5
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"] 
    betapre = params["betapre"]
    seed = params["seed"]
    nthreads = params["nthreads"]
    """
    Prepares a kyber instance. Attempts to call load_lwe to load instances, then extracts
    the instanse bse[seed[1]]. If load fails, calls gen_and_dump_lwe. Then it attenmnpts to load
    an already preperocessed lattice. If fails, it preprocesses one.
    """
    report = {
        "kyb": ( n,q,dist, dist_param,seed ),
        "beta": betapre,
        "time": 0
    }

    try: #try load lwe instance
        A, q, bse = load_lwe(params) #D["A"], D["q"], D["bse"]
    except FileNotFoundError: #if no such, create one
        print(f"No kyber instance found... generating.")
        gen_and_dump_lwe(params) #ntar = 5
        A, q, bse = load_lwe(params) #D["A"], D["q"], D["bse"]
    #try load reduced kyber
    try:
        with open(out_path + f"kyb_preprimal_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{betapre}.pkl", "rb") as file:
            B = pickle.load(file)
            print(f"Kyber located")
    except (FileNotFoundError, EOFError): #if no such, create one
        B = [ [int(0) for i in range(2*n)] for j in range(2*n) ]
        for i in range( n ):
            B[i][i] = int( q )
        for i in range(n, 2*n):
            B[i][i] = 1
        for i in range(n, 2*n):
            for j in range(n):
                B[i][j] = int( A[i-n,j] )

        B = IntegerMatrix.from_matrix( B )
        #nthreads=5 by default since preprocessing operates with small blocksizes
        LR = LatticeReduction( B,threads_bkz=nthreads )
        for beta in range(4,betapre+1):
            then = time.perf_counter()
            LR.BKZ( beta )
            round_time = time.perf_counter()-then
            print(f"Preprocess BKZ-{beta} done in {round_time}", flush=True)
            report["time"] += round_time

        with open(out_path + f"kyb_preprimal_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{betapre}.pkl", "wb") as file:
            pickle.dump( LR.basis, file )
        B = LR.basis
        with open(out_path + f"report_pre_{n}_{q}_{dist}_{dist_param:.04f}_{seed[0]}_{betapre}.pkl", "wb") as file:
            pickle.dump( report, file )

    return B, A, q, dist, dist_param, bse

def attack_on_kyber(params):
    # prepeare the lattice
    n = params["n"]
    q = params["q"]
    dist = params["dist"]
    dist_param = params["dist_param"]
    betapre = params["betapre"]
    betamax = params["betamax"]
    seed = params["seed"]
    nthreads = params["nthreads"]
    print( f"launching {n,q,dist,dist_param,seed}" )
    B, A, q, dist, dist_param, bse = prepare_kyber(params)
    dim = B.nrows+1 #dimension of Kannan

    print(f"Total instances per lat: {len(bse)} seed={seed[1]}")
    b, s, e = bse[seed[1]]

    r,c = A.shape
    print(f"Shape: {A.shape}, n")
    t = np.concatenate([b,[0]*r]) #BDD target
    x = np.concatenate([b-e,s,[-1]]) #BBD solution
    sol = np.concatenate([e,-s,[1]])

    B = [ [ bb for bb in b ]+[0] for b in B ] + [ (dim-1)*[0] + [1] ]

    for j in range(n):
        B[-1][j] = int( t[j] )
    C = IntegerMatrix.from_matrix( B )
    B = np.array( B )
    tarnrmsq = 1.01*(sol.dot(sol))

    ft = "dd" if (config.have_qd and C.nrows<450) else "mpfr"
    FPLLL.set_precision(208)
    G = GSO.Mat(C,float_type=ft, U=IntegerMatrix.identity(dim,int_type=C.int_type), UinvT=IntegerMatrix.identity(dim,int_type=C.int_type))
    G.update_gso()

    print(G.get_r(0,0)**0.5)

    report = {
        "kyb": ( n,q,dist, dist_param ),
        "beta": 2,
        "time": 0,
        "projinfo": {}
    }
    lll = LLL.Reduction(G)
    then = time.perf_counter()
    lll()
    llltime = time.perf_counter() - then
    report = {
        "kyb": ( n,q,dist, dist_param ),
        "beta": 2,
        "time": llltime,
        "projinfo": {}
    }
    beta = betapre
    if lll.M.get_r(0,0) <= tarnrmsq:
        print(f"LLL recovered secret!")
        report["beta"] = beta
        return report

    flags = BKZ.AUTO_ABORT|BKZ.MAX_LOOPS|BKZ.GH_BND
    bkz = BKZReduction(G)

    for beta in range(betapre-1,min(betamax+1,BKZ_SIEVING_CROSSOVER)):    #BKZ reduce the basis
        par = BKZ.Param(beta,
                               max_loops=BKZ_MAX_LOOPS,
                               flags=flags,
                               strategies=BKZ.DEFAULT_STRATEGY
                               )
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        curnrm = np.array( bkz.M.B[0] ).dot( np.array( bkz.M.B[0] ) )**(0.5)
        # print(f"BKZ-{beta} done in {round_time} | {curnrm}")
        slope = basis_quality(bkz.M)["/"]
        print(f"Enum beta: {beta:}, done in: {round_time : 0.4f}, slope: {slope}  log r00: {log( bkz.M.get_r(0,0),2 )/2 : 0.5f} task_id = {seed}", flush=True)
        report["time"] += round_time

        if bkz.M.get_r(0,0) <= tarnrmsq:
            print(f"successes! beta={beta}")
            report["beta"] = beta
            return report

    M = bkz.M
    try:
        param_sieve = SieverParams()
        param_sieve['threads'] = nthreads #10
        param_sieve['default_sieve'] = "bgj1" #"bgj1" "bdgl2"

        #we do not use lattice_reduction here since we do not neccesarily
        #want to run all the tours and can interupt after any given one.
        LR = LatticeReduction( M.B, threads_bkz=nthreads )
        for beta in range(max(BKZ_SIEVING_CROSSOVER,betapre-1),betamax+1):
            then_round=time.perf_counter()
            LR.BKZ(beta,tours=5)
            round_time = time.perf_counter()-then_round
            slope = basis_quality(M)["/"]
            print(f"beta: {beta:}, done in: {round_time : 0.4f}, slope: {slope : 0.6f}, log r00: {log( M.get_r(0,0),2 )/2 : 0.5f} task_id = {seed}", flush=True)
            report["time"] += round_time
            
            M = LR.gso
            if M.get_r(0,0) <= tarnrmsq:
                print(f"successes! beta={beta}")
                report["beta"] = beta
                return report

    except Exception as excpt:
        print( excpt )
        print("Sieving died!")
        pass

    return report

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
    "--ns", default= "range(125,126,1)", type=str, help="String that evaluates list of LWE dimensions."
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
    "--betamax", default=60, type=int, help="Upper bound on the BKZ blocksize."
    )
    parser.add_argument("--recompute_instance", action="store_true", help="Recomputes instances. WARNING deletes previous instance irreversibly.")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

if __name__ == "__main__":
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
    ns = [n for n in eval( args.ns )]
    betapre,betamax = args.betapre, args.betamax

    output = []
    pool = Pool( processes = nworkers )
    tasks = []
    RECOMPUTE_INSTANCE = args.recompute_instance
    RECOMPUTE_KYBER = False
    if RECOMPUTE_INSTANCE:
        print(f"Generating Kyber...")
        for n in ns:
            for latnum in range(lats_per_dim):
                params = {
                    "n": n,
                    "q": q,
                    "dist": dist,
                    "dist_param": dist_param,
                    "ntar": inst_per_lat,
                    "betapre": betapre,
                    "betamax": betamax,
                    "seed": [latnum,0],
                    "nthreads": nthreads
                }
                gen_and_dump_lwe(params)

    if RECOMPUTE_KYBER or RECOMPUTE_INSTANCE:
        pretasks = []
        for n in ns:
            for latnum in range(lats_per_dim):
                params = {
                    "n": n,
                    "q": q,
                    "dist": dist,
                    "dist_param": dist_param,
                    "ntar": inst_per_lat,
                    "betapre": betapre,
                    "betamax": betamax,
                    "seed": [latnum,0],
                    "nthreads": nthreads
                }
                pretasks.append( pool.apply_async(
                prepare_kyber, (params,) #NOTE: comma is crucial here
                ) )
        print(f"Preprocessing Kyber...", flush=True)
        for t in pretasks:
            t.get()

    for n in ns:
        for latnum in range(lats_per_dim):
            for tstnum in range(inst_per_lat):
                params = {
                    "n": n,
                    "q": q,
                    "dist": dist,
                    "dist_param": dist_param,
                    "betapre": betapre,
                    "betamax": betamax,
                    "seed": [latnum,tstnum],
                    "nthreads": nthreads
                }
                tasks.append( pool.apply_async(
                    attack_on_kyber, ( params, )
                    ) )


    for t in tasks:
            output.append( t.get() )

    pool.close()

    name = f"exp{ns}_{q}_{dist}_{dist_param:.04f}.pkl" if dist=="ternary" else f"exp{ns}_{q}_{dist}_{dist_param}.pkl"
    with open( out_path+name, "wb" ) as file:
        pickle.dump( output,file )

    print(f"Experimental data dumped to {out_path+name}")

    print(f"- - - output - - -")
    print(output)

    time.sleep(0.5)
