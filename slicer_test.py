import warnings
import re

warnings.filterwarnings("ignore", message=".*Dimension of lattice is larger than.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API..*")

from fpylll.util import gaussian_heuristic
from fpylll import *
from g6k.siever import Siever
from g6k.utils.stats import dummy_tracer
from g6k.siever_params import SieverParams
# from g6k.algorithms.pro_randslicer import pro_randslicer
from g6k.algorithms.pump import pump
from math import sqrt
from g6k.utils.util_pump import load_cvp_instance
from copy import deepcopy
# from DistEstColattice import DistEstDistEstColattice
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

import warnings

warnings.filterwarnings('ignore')

def dot_product(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])

def norm(a):
    return sqrt(dot_product(a,a))


#Use the simulator like BKZ 2.0
def draw_cvp_bound_simulation():
    return


def cvp_test(A,t, params, myparams):
    nrand_fact =  myparams["nrand_fact"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        close_vector = tuple([0]*A.ncols)
        sample_times = 1
        T_sieve = 0
        db_size = 0
        if(A.nrows == 1):
            T0 =time.time()
            #babai
            c = round(dot_product(t,A[0])/ dot_product(A[0],A[0]))
            close_vector = tuple([c*A[0][i] for i in range(A.ncols)])
            T_slicer = time.time() - T0
        elif(A.nrows <= 30):
            #CVP enumeration
            A = IntegerMatrix.from_matrix(A, int_type="mpz")
            T0 =time.time()
            close_vector = CVP.closest_vector(A,t)
            T_slicer = time.time() - T0
        else:
            #randomlized slicer
            # params = SieverParams(threads = 1,saturation_ratio = 1.)
            T0 = time.time()
            params["threads"] = myparams["nthreads_sieve"]
            params["sieve"] = "bdgl2"
            params["saturation_ratio"] = 0.95
            params["saturation_radius"] = 4/3.
            params["dbsize_factor"] = 5.
            A = IntegerMatrix.from_matrix(A,int_type="mpz")
            g6k = Siever(A,params)
            d = g6k.full_n
            g6k.lll(0, d)
            g6k.initialize_local(0,0,d)
            g6k.update_gso(0, d)
            # g6k.initialize_local(0,max(0,A.nrows-50),A.nrows)
            
            """
            def pump(g6k, tracer, kappa, blocksize, dim4free, down_sieve=False,                                 # Main parameters
                     goal_r0=None, max_up_time=None, down_stop=None, start_up_n=30, saturation_error="weaken",  # Flow control of the pump
                     increasing_insert_index=True, prefer_left_insert=1.04,                                     # Insertion policy
                     verbose=False,                                                                             # Misc
                    ):
            """
            f = 0
            print(f"g6k.r: {g6k.r}")
            pump(g6k, dummy_tracer, 0, g6k.r, f, saturation_error="ignore", verbose=False)
            print( f"context: {g6k.ll, g6k.l, g6k.r}" )
            g6k.extend_left()
            print(f"lolpre: {len(g6k)}")
            for i in range(1):
                g6k.grow_db(ceil(1.5*len(g6k)))
                g6k()
            
            T_sieve = time.time() - T0

            T0 = time.time()
            gh = gaussian_heuristic(g6k.M.r())
            t_gs = to_canonical_scaled(g6k.M,t,offset=g6k.M.d,scale_fact=gh)
            slicer = RandomizedSlicer(g6k)
            slicer.set_nthreads(myparams["nthreads"])

            G = g6k.M
            dim = G.d
            sieve_dim = G.d

            t_gs_non_scaled = G.from_canonical(t)[dim-sieve_dim:]
            shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, gso=True) )
            shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
            t_gs_reduced = from_canonical_scaled( G,np.array(t, dtype=DTYPE)-shift_babai,offset=sieve_dim,scale_fact=gh ) #this is the actual reduced target
            

            # B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
            # t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
            # t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

            # bab_1 = G.babai(t-np.array(out),start=sieve_dim) #last sieve_dim coordinates of s
            # tmp = t - np.array( G.B[sieve_dim:].multiply_left(bab_1) )
            # tmp = G.to_canonical( G.from_canonical( tmp, start=0, dimension=sieve_dim ) ) #project onto span(B[-sieve_dim:])
            # bab_0 = G.babai(tmp)

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

            slicer.set_proj_error_bound(myparams["proj_err_bound"])
            slicer.set_max_slicer_interations(myparams["max_slicer_interations"])
            slicer.set_Nt(1)
            slicer.set_saturation_scalar(myparams["saturation_scalar"])
            slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False) #slicer_verbosity

            iterator = slicer.itervalues_cdb_t()
            out_gs_reduced = None
            for tmp, _ in iterator:
                out_gs_reduced = np.array(tmp)  #cdb[0]
                break
            assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"

            out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh )
            bab_01 = np.round( np.array( G.babai( np.array(t)-out ) ) )
            close_vector = G.B.multiply_left(bab_01)

            T_slicer, sample_times = time.time()- T0, nrand
            db_size = len(g6k)
            # f = 0
            # # print(g6k.M.B.nrows,g6k.M.B.ncols)
            # close_vector,_,sample_times, T_sieve, T_slicer, db_size = pro_randslicer(g6k,t,dummy_tracer,f,verbose=False, )


            T_slicer = time.time() - T0

    return close_vector, sample_times, T_sieve, T_slicer, db_size


def run_exp(n,index,A,t,myparams,tracer=None):
    param_sieve = SieverParams()
    param_sieve['otf_lift'] = False
    params = param_sieve

    print("{0: <10} | {1: <10} | {2: <15} | {3: <30} | {4: <15} | {5: <15} | {6: <15} | {7: <15} | {8: <15} | {9: <15}".format("dim", "index", "sample times", "estimated sample times", "T_pump (sec)", "T_slicer (sec)", "dt", "gh", "db_size", "satisfied vectors"))
    exp_results = []

    # A, t = load_cvp_instance(n)
    A = LLL.reduction(A)


    g6k = Siever(A,None)
    for blocksize in range(10, 5, 31):
        bkz = BKZReduction(g6k.M)
        par = fplll_bkz.Param(blocksize,
                                    strategies=fplll_bkz.DEFAULT_STRATEGY,
                                    max_loops=1)
        bkz(par)


    rr = [g6k.M.get_r(i, i) for i in range(n)]


    w, sample_times,T_pump, T_slicer, db_size = cvp_test(A,t, params, myparams)
    max_sample_times = ceil((16/13.)**(n//2.))


    gh = sqrt(gaussian_heuristic(rr))
    dt = sqrt(sum([(w[i] - t[i])**2 for i in range(len(t))]))
    # simDist = DistEstDistEstColattice([log(_)/2. for _ in rr[:n]], [n])

    print("{0: <10} | {1:<10} | {2: <15} | {3: <30} | {4: <15} | {5: <15} | {6: <15} | {7: <15} | {8: <15} | {9: <15}".format(n,index, sample_times, max_sample_times, round(T_pump,4), round(T_slicer,4), round(dt,3), round(gh,3), db_size, int(.5 * params.saturation_ratio * params.db_size_base ** n )))
    return( [T_pump, T_slicer, dt, gh, db_size] )

# params = SieverParams(threads = 1 ,  saturation_ratio = 1. )#, saturation_ratio = 0.75)#, saturation_ratio = 1.,  db_size_factor = 5, default_sieve = "bgj1" )#, db_size_factor = 1.5 )

rngs = (55, 66, 5)
tours = 10
myparams = {
    "max_slicer_interations": 150,
    "proj_err_bound": 0.7,
    "saturation_scalar": 1.0,
    "nrand_fact": 100,
    "nthreads": 1,
    "nthreads_sieve": 5,
}
pool = Pool(processes = 10)

filename = f"prec_cvp_chal_{rngs}_{tours}.pkl"
loaded = False
try:
    with open(filename,"rb") as file:
        At_dict = pickle.load(file)
        loaded = True
    print("Precomputed challenges found.")
except FileNotFoundError:
    print("No precomputed challenges found.")
    At_dict = {}
    for n in range(rngs[0], rngs[1], rngs[2]):
        At_dict[n] = []
        for index in range(tours):
            A, t = load_cvp_instance(n)
            At_dict[n].append([A,t])
    with open(filename,"wb") as file:
        pickle.dump(At_dict,file)

tasks = {}
exp_results = {}
for n in range(rngs[0], rngs[1], rngs[2]):
    tasks[n] = []
    exp_results[n] = []
    for index in range(tours):
            exp_results[n].append( None )
            A, t = At_dict[n][index]
            tasks[n].append( 
                pool.apply_async( run_exp, (n,index,A,t,myparams,None) )
             )
            
for n in range(rngs[0], rngs[1], rngs[2]):
    for index in range(tours):
        exp_results[n][index] = tasks[n][index].get()

filename_exp = f"cvp_chal_bdgl__{rngs}_{tours}.pkl"
with open(filename_exp,"wb") as file:
        pickle.dump(exp_results,file)