from fpylll.util import gaussian_heuristic
from fpylll import *
from g6k.siever import Siever
from g6k.utils.stats import dummy_tracer
from g6k.siever_params import SieverParams
# from g6k.algorithms.pro_randslicer import pro_randslicer
from g6k.algorithms.pump import pump
from math import log, ceil

from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
import time

FPLLL.set_random_seed(0x1337)
from g6k.siever import SaturationError
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer

import numpy as np
from utils import *
import pickle
from hybrid_estimator.batchCVP import batchCVPP_cost
from multiprocessing import Pool

from cvpp_exp import gen_cvpp_g6k
from LatticeReduction import LatticeReduction
# def gen_cvpp_g6k(n,betamax=None,k=None,bits=11.705,seed=0,threads=1,verbose=False):

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

    ft = "ld" if n<50 else ( "dd" if config.have_qd else "mpfr")
    G=GSO.Mat( B, float_type=ft,
              U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
              UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type) 
      )
    G.update_gso()

    param_sieve = SieverParams()
    param_sieve['threads'] = params["nthreads_sieve"]
    params["sieve"] = "bdgl2"
    params["saturation_ratio"] = 0.95
    params["saturation_radius"] = 4/3.
    params["dbsize_factor"] = 5.

    Tpre_pump = time.perf_counter()
    g6k = Siever(G)
    g6k.lll(0, n)
    g6k.initialize_local(0,0,n)
    g6k.update_gso(0, n)
    f=0
    pump(g6k, dummy_tracer, 0, g6k.r, f, saturation_error="ignore", verbose=False)
    # print( f"context: {g6k.ll, g6k.l, g6k.r}" )
    while not g6k.l==0:
        g6k.extend_left()
        g6k()

    Tpump = time.perf_counter() - Tpre_pump

    Tpre_slice = time.perf_counter()
    gh = gaussian_heuristic(g6k.M.r())
    t_gs = to_canonical_scaled(g6k.M,t,offset=g6k.M.d,scale_fact=gh)
    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(1) #one since the slicer we are comparing against is not parralelized

    G = g6k.M
    dim = G.d
    sieve_dim = G.d

    t_gs_non_scaled = G.from_canonical(t)[dim-sieve_dim:]
    shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, gso=True) )
    shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
    t_gs_reduced = from_canonical_scaled( G,np.array(t, dtype=DTYPE)-shift_babai,offset=sieve_dim,scale_fact=gh ) #this is the actual reduced target
    
    nrand_, _ = batchCVPP_cost(g6k.M.d,1,len(g6k)**(1./g6k.M.d),1)
    nrand = ceil(nrand_fact*(1./nrand_)**sieve_dim)
    slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = SieverParams()
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    slicer.set_proj_error_bound(params["proj_err_bound"])
    slicer.set_max_slicer_interations(params["max_slicer_interations"])
    slicer.set_Nt(1)
    slicer.set_saturation_scalar(params["saturation_scalar"])
    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], True) #slicer_verbosity

    iterator = slicer.itervalues_cdb_t()
    out_gs_reduced = None
    for tmp, _ in iterator:
        out_gs_reduced = np.array(tmp)  #cdb[0]
        break
    assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"

    out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh )
    bab_01 = np.round( np.array( G.babai( np.array(t)-out ) ) )
    close_vector = G.B.multiply_left(bab_01)

    Tslice = time.perf_counter() - Tpre_slice

    return close_vector, nrand, Tpump, Tslice, len(g6k), gh

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

    # tasks = []
    # results = []
    # for B, cbs in L:
    #     for cb in cbs: 
    #         c, b = cb['c'], cb['b']
    #         close_vector, nrand, Tpump, Tslice, db_size, gh = solve_cvp(B,cb['b'], myparams)
    #         v = B.multiply_left( c )
    #         dt = (sum([(b[i] - close_vector[i])**2 for i in range(len(b))]))
    #         results.append( [nrand, Tpump, Tslice, db_size, dt, gh] )
    
    print("Running experiments.", flush=True)
    pool = Pool(processes=n_workers)
    tasks = []
    results = []
    expid = [0,0]
    for B, cbs in L:
        for cb in cbs: 
            tasks.append( pool.apply_async(
                run_experiment, (B, cb, myparams, [tmp for tmp in expid])
            ) )
            expid[1]+=1
        expid[0]+=1
        expid[1]=0


    for tsk in tasks:
            results.append( tsk.get() )

    filename = f"cvp_comp_{n}_{lat_num}_{inst_per_lat}_{betamax}_{appr_fact:0.4f}.pkl"
    with open(filename,"wb") as file:
        pickle.dump(results, file)
    print(results)
    print(f"Save to {filename}")

    
