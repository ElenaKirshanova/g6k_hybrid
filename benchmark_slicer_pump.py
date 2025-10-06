import warnings
import re

warnings.filterwarnings("ignore", message=".*Dimension of lattice is larger than.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API..*")

from fpylll.util import gaussian_heuristic
from fpylll import *
from g6k.siever import Siever
from g6k.utils.stats import dummy_tracer
from g6k.siever_params import SieverParams
from g6k.algorithms.pro_randslicer import pro_randslicer
from g6k.algorithms.pump import pump
from math import log, ceil
from random import randrange

from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
import sys, os, time

FPLLL.set_random_seed(0x1337)
from g6k.siever import SaturationError
from g6k.siever_params import SieverParams

import numpy as np
# from utils import *
import pickle
from multiprocessing import Pool

from lattice_reduction import LatticeReduction
from utils import random_on_sphere

verbose = True

def dot_product(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])

def norm(a):
    return sqrt(dot_product(a,a))

def gen_cvp_chal_w_bkz(n, inst_per_lat, betamax=50, apprr_fact=0.999):
    ft = "ld" if n<50 else ( "dd" if config.have_qd else "mpfr")

    B = IntegerMatrix(n,n)
    B.randomize("qary", k=n//2+1, bits=11.705)

    G=GSO.Mat( B, float_type=ft )
    G.update_gso()
    lll = LLL.Reduction(G)
    then = time.perf_counter()
    lll()
    print(f"LLL done in {time.perf_counter()-then}")

    for beta in range(5,betamax+1):
        bkz = LatticeReduction(G.B)
        then = time.perf_counter()
        bkz.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then
        if verbose: print(f"BKZ-{beta} done in {round_time}")
        sys.stdout.flush()

    gh = gaussian_heuristic( bkz.gso.r() )
    cb = []
    B = bkz.gso.B
    for i in range(inst_per_lat):
        c = np.array( [randrange(-256,257) for j in range(inst_per_lat)], dtype=np.int64 )
        e = np.round( random_on_sphere( n, (apprr_fact*gh)**0.5 ) )
        b = np.array( B.multiply_left( c ) ) + e
        cb.append( {'c': c,'b': b} )
    return B, cb

def solve_cvp(B, t, params):
    sieve_dim =  B.nrows
    nrand_fact = params["nrand_fact"]
    n = sieve_dim

    ft = "ld" if n<50 else ( "dd" if config.have_qd else "mpfr")
    G=GSO.Mat( B, float_type=ft,
              U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
              UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type) 
      )
    G.update_gso()
    gh = gaussian_heuristic(G.r())

    param_sieve = SieverParams()
    param_sieve = SieverParams(threads = params["nthreads"] ,  saturation_ratio = 1. )

    if(B.nrows <=10):
            T0 =time.time()
            #babai
            c = round(dot_product(t,B[0])/ dot_product(B[0],B[0]))
            close_vector = tuple([c*B[0][i] for i in range(B.ncols)])
            Tslice = time.time() - T0
            return close_vector, 1, 0, Tslice, 0, gh
    elif(B.nrows <= 30):
            #CVP enumeration
            A = IntegerMatrix.from_matrix(B, int_type="mpz")
            T0 =time.time()
            close_vector = CVP.closest_vector(A,t)
            Tslice = time.time() - T0
            return close_vector, 1, 0, Tslice, 0, gh

    Tpre_pump = time.perf_counter()
    g6k = Siever(G,param_sieve)
    f=0
    close_vector,_,sample_times, Tpump, Tslice, db_size = pro_randslicer(g6k,t,dummy_tracer,f,verbose=False, )

    return close_vector, sample_times, Tpump, Tslice, db_size, gh

def run_experiment(B,cb,myparams,expid):
    c, b = cb['c'], cb['b']
    then = time.perf_counter()
    close_vector, nrand, Tpump, Tslice, db_size, gh = solve_cvp(B,cb['b'], myparams)
    print(f"experiment {expid} is finished in {time.perf_counter()-then}", flush=True)
    v = B.multiply_left( c )
    dt = (sum([(b[i] - close_vector[i])**2 for i in range(len(b))]))
    return [nrand, Tpump, Tslice, db_size, dt, gh]

if __name__ == "__main__":
    n, lat_num, inst_per_lat, betamax, appr_fact = 64, 2, 5, 50, 0.999
    n_workers = 2
    myparams = {
    "max_slicer_interations": 150,
    "proj_err_bound": 0.95,
    "saturation_scalar": 1.0,
    "nrand_fact": 50,
    "nthreads": 1,
    "nthreads_sieve": 5,
    }

    os.makedirs("./saved_lattices", exist_ok=True)
    filename = f"./saved_lattices/cvp_lats_{n}_{lat_num}_{inst_per_lat}_{betamax}_{appr_fact:0.4f}.pkl"
    L = []
    try:
        with open(filename,"rb") as file:
            print(f"Found {filename}... proceeding")
            L = pickle.load( file )
    except FileNotFoundError:
        print(f"No {filename}... computing")
        for _ in range(lat_num):
            B, cb = gen_cvp_chal_w_bkz(n,inst_per_lat,betamax,appr_fact)
            L.append( [B,cb] )
        with open(filename,"wb") as file:
            pickle.dump( L, file )

    print("Running experiments.")
    pool = Pool(processes=n_workers)
    tasks = []
    results = []
    expid = [0,0]
    for B, cbs in L:
        for cb in cbs: 
            # c, b = cb['c'], cb['b']
            # close_vector, nrand, Tpump, Tslice, db_size, gh = solve_cvp(B,cb['b'], myparams)
            # v = B.multiply_left( c )
            # dt = (sum([(b[i] - close_vector[i])**2 for i in range(len(b))]))
            tasks.append( pool.apply_async(
                run_experiment, (B, cb, myparams,[tmp for tmp in expid])
            ) )
            expid[1]+=1
        expid[0]+=1
        expid[1]=0

    for tsk in tasks:
            results.append( tsk.get() )

    path = f"../cvp_comp/pump/"
    os.makedirs(path,exist_ok=True)
    filename = path+f"cvp_comp_{n}_{lat_num}_{inst_per_lat}_{betamax}_{appr_fact:0.4f}.pkl"
    with open(filename,"wb") as file:
        pickle.dump(results, file)
    print(results)
    print(f"Save to {filename}")