from fpylll.util import gaussian_heuristic
from fpylll import *
from g6k.siever import Siever
from g6k.utils.stats import dummy_tracer
from g6k.siever_params import SieverParams
# from g6k.algorithms.pro_randslicer import pro_randslicer
from math import sqrt
from g6k.utils.util_chi import load_cvp_instance
from copy import deepcopy
# from DistEstColattice import DistEstDistEstColattice
from math import log, ceil
from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
import time

FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer

import numpy as np
from utils import *
import pickle
from hybrid_estimator.batchCVP import batchCVPP_cost

import warnings

warnings.filterwarnings('ignore')

def dot_product(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])

def norm(a):
    return sqrt(dot_product(a,a))


#Use the simulator like BKZ 2.0
def draw_cvp_bound_simulation():
    return


def cvp_test(A,t, params):
    nrand_fact =  10

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
            g6k = Siever(A,params)
            g6k.initialize_local(0,0,A.nrows)
            g6k(alg="bdgl2")
            T_sieve = time.time() - T0

            T0 = time.time()
            gh = gaussian_heuristic(g6k.M.r())
            t_gs = to_canonical_scaled(g6k.M,t,offset=g6k.M.d,scale_fact=gh)
            slicer = RandomizedSlicer(g6k)
            slicer.set_nthreads(5)

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

            slicer.set_proj_error_bound(0.7)
            slicer.set_max_slicer_interations(50)
            slicer.set_Nt(1)
            slicer.set_saturation_scalar(1.0)
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

            T_slicer, sample_times = time.time()- T0, nrand
            db_size = len(g6k)
            # f = 0
            # # print(g6k.M.B.nrows,g6k.M.B.ncols)
            # close_vector,_,sample_times, T_sieve, T_slicer, db_size = pro_randslicer(g6k,t,dummy_tracer,f,verbose=False, )


            T_slicer = time.time() - T0

    return close_vector, sample_times, T_sieve, T_slicer, db_size




# params = SieverParams(threads = 1 ,  saturation_ratio = 1. )#, saturation_ratio = 0.75)#, saturation_ratio = 1.,  db_size_factor = 5, default_sieve = "bgj1" )#, db_size_factor = 1.5 )
param_sieve = SieverParams()
param_sieve['otf_lift'] = False
params = param_sieve

rngs = (60, 81, 10)
tours = 10

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
    loaded = True

print("{0: <10} | {1: <10} | {2: <15} | {3: <30} | {4: <15} | {5: <15} | {6: <15} | {7: <15} | {8: <15} | {9: <15}".format("dim", "index", "sample times", "estimated sample times", "T_pump (sec)", "T_slicer (sec)", "dt", "gh", "db_size", "satisfied vectors"))
filename_exp = f"cvp_chal_bdgl__{rngs}_{tours}.pkl"
exp_results = {}
for n in range(rngs[0], rngs[1], rngs[2]):
    exp_results[n] = []
    for index in range(tours):
        A, t = At_dict[n][index]
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


        w, sample_times,T_pump, T_slicer, db_size = cvp_test(A,t, params)
        max_sample_times = ceil((16/13.)**(n//2.))


        gh = sqrt(gaussian_heuristic(rr))
        dt = sqrt(sum([(w[i] - t[i])**2 for i in range(len(t))]))
        # simDist = DistEstDistEstColattice([log(_)/2. for _ in rr[:n]], [n])

        print("{0: <10} | {1:<10} | {2: <15} | {3: <30} | {4: <15} | {5: <15} | {6: <15} | {7: <15} | {8: <15} | {9: <15}".format(n,index, sample_times, max_sample_times, round(T_pump,4), round(T_slicer,4), round(dt,3), round(gh,3), db_size, int(.5 * params.saturation_ratio * params.db_size_base ** n )))
        exp_results[n].append( [T_pump, T_slicer, dt, gh, db_size] )

with open(filename_exp,"wb") as file:
    pickle.dump(exp_results,file)