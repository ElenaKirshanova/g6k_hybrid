"""
BKZ-beta reduces Nlats lattice bases. Solves ntests Tail-Batch-BDD instances (with appr. factor approx_factor) each consisting of n_uniq_targets BDD instances.

python tailBDD.py --n 120 --beta 55 --approx_factor 0.43 --Nlats 5  --ntests 5 --n_uniq_targets 10
"""
from experiments.lwe_gen import *

import sys,os
import argparse
from time import perf_counter
from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever, SaturationError
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *

from global_consts import *

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from sample import *

from preprocessing import load_lwe
from hybrid_estimator.batchCVP import batchCVPP_cost
from lattice_reduction import LatticeReduction

def get_parser():
    parser = argparse.ArgumentParser(
        description="CVPP experiments."
    )
    parser.add_argument(
    "--nthreads", default=1, type=int, help="Threads per slicer."
    )
    parser.add_argument(
    "--nworkers", default=1, type=int, help="Workers for experiments."
    )
    parser.add_argument(
    "--ntests", default=1, type=int, help="Number of tests per lattice."
    )
    parser.add_argument(
    "--Nlats", default=1, type=int, help="TNumber of lattices."
    )
    parser.add_argument(
    "--n", default=80, type=int, help="Lattice dimension"
    )
    parser.add_argument(
    "--beta", default=50, type=int, help="Lattice dimension"
    )
    parser.add_argument(
    "--approx_factor", default=0.43, type=float, help="Lattice dimension"
    )
    parser.add_argument(
    "--nrand_param", default=10., type=float, help="Lattice dimension"
    )
    parser.add_argument(
    "--n_uniq_targets", default=5, type=int, help="Lattice dimension"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

def gen_cvpp_g6k(n,betamax=None,n_slicer_coord=None,k=None,bits=11.705,seed=0):
    betamax=n if betamax is None else betamax
    n_slicer_coord=n if n_slicer_coord is None else n_slicer_coord


    k = n//2 if k is None else k
    B = IntegerMatrix(n,n)
    B.randomize("qary", bits=bits, k = k)

    LR = LatticeReduction( B )
    for beta in range(5,betamax+1):
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}", flush=True)

    int_type = LR.gso.B.int_type
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    G = GSO.Mat( LR.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
    param_sieve = SieverParams()
    param_sieve['threads'] = 1
    param_sieve['db_size_base'] = (4/3.)**0.5 #(4/3.)**0.5 ~ 1.1547
    param_sieve['db_size_factor'] = 3.2 #3.2
    param_sieve['saturation_ratio'] = 0.5
    param_sieve['saturation_radius'] = 1.32

    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-n_slicer_coord,n-n_slicer_coord,n)
    print("Running bdgl2...")
    then=perf_counter()
    try:
        g6k(alg="bdgl2")
    except SaturationError:
        pass
    print(f"bdgl2-{n} done in {perf_counter()-then}")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")
    g6k.dump_on_disk(f"cvppg6k_n{n}_b{betamax}_d{n_slicer_coord}_{seed}_test.pkl")

def run_experiment( lat_index, params, stats_dict, verbose=False ):
    n, beta = params["n"],  params["beta"]
    n_uniq_targets = params["n_uniq_targets"]
    slicer_iterations, nrand_param, approx_factor = params["slicer_iterations"] , params["nrand_param"] , params["approx_factor"]
    nthreads, seed = params["nthreads"] , lat_index
    n_slicer_coord = beta
    ntests = params["ntests"]

    sieve_filename = f"cvppg6k_n{n}_b{beta}_d{beta}_{seed}_test.pkl"
    nothing_to_load = True
    try:
        g6k = Siever.restore_from_file(sieve_filename)
        G = g6k.M
        param_sieve = SieverParams()
        param_sieve['threads'] = nthreads
        param_sieve['otf_lift'] = False
        g6k.params = param_sieve
        nothing_to_load = False
    except Exception as excpt:
        print(excpt)
        gen_cvpp_g6k(n,betamax=beta,n_slicer_coord=beta,k=None,bits=11.705,seed=seed)
        pass

    g6k = Siever.restore_from_file(sieve_filename)
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve
    G = g6k.M
    B = G.B

    ft = "ld" if n<50 else ( "dd" if config.have_qd else "mpfr")
    if verbose: print(f"launching n, beta, sieve_dim = {n, beta, n_slicer_coord}")
    sieve_dim = beta
    gh = gaussian_heuristic(G.r())
    lambda1 = min( [G.get_r(0, 0)**0.5, gh**0.5] )

    aggregated_data = []
    #retrieve the projective sublattice
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]

    D = { (n,beta,approx_factor): [] }
    gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
    for tstnum in range(ntests):
        print(f"- - - TSTNUM: {tstnum} LATSEED {seed} - - - ")
        Ts, Cs, Bs = [], [], []
        e_gs_mmin = 2**64
        for i in range(n_uniq_targets):
            c = [ randrange(-2,3) for j in range(n) ]
            e = np.array( random_on_sphere(n,approx_factor*lambda1) )
            b = np.array( B.multiply_left( c ) )
            t = b+e
            Ts.append(t); Cs.append(c); Bs.append(b)

            e_gs = from_canonical_scaled( G,e,offset=beta,scale_fact=gh_sub )
            e_gs = (e_gs@e_gs)**0.5
            if e_gs < e_gs_mmin:
                e_gs_mmin = e_gs
            D[(n,beta,approx_factor)].append( [e_gs,False] ) #proj. approx fact and whether it was a success
        dist_sq_bnd = 0.99*e_gs_mmin**2

        TGSs = []
        for i in range(n_uniq_targets):
            t = Ts[i]
            t_gs_non_scaled = G.from_canonical(t)[n-sieve_dim:]
            shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=n-sieve_dim, gso=True) )
            shift_babai = G.B.multiply_left( (n-sieve_dim)*[0] + list( shift_babai_c ) )
            t_gs_reduced = from_canonical_scaled( G,np.array(t, dtype=DTYPE)-shift_babai,offset=sieve_dim,scale_fact=gh_sub ) #this is the actual reduced target

            TGSs.append(t_gs_reduced)

        slicer = RandomizedSlicer(g6k)
        slicer.set_nthreads(nthreads)
        nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
        nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
        print(f"total randomizations: {nrand*n_uniq_targets} len(g6k): {len(g6k)}")

        for i in range(n_uniq_targets):
            t_gs_reduced = TGSs[i]
            slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

        blocks = 2 # should be the same as in siever
        blocks = min(3, max(1, blocks))
        blocks = min(int(sieve_dim / 28), blocks)
        sp = SieverParams()
        N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
        buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
        buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
        buckets = max(buckets, 2**(blocks-1))
 
        slicer.set_nthreads(nthreads)
        slicer.set_proj_error_bound( (EPS2*(dist_sq_bnd)) )
        slicer.set_Nt(ceil(n_uniq_targets))
        slicer.set_saturation_scalar(10.)
        slicer.set_max_slicer_interations(slicer_iterations)
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)

        iterator = slicer.itervalues_cdb_t()

        for tmp, indx in iterator:
            out_gs_reduced = np.array(tmp)
            out_reduced = np.array( to_canonical_scaled( G, out_gs_reduced, offset=sieve_dim, scale_fact=gh_sub ) )
            # the line below projects the error away from first basis vectors
            out_reduced = G.to_canonical( (G.d-sieve_dim)*[0] + list( G.from_canonical( out_reduced,start=G.d-sieve_dim ) ), start=0 )

            t = np.array( Ts[indx], )
            bab_01 = np.array( G.babai(t-out_reduced) )
            c = Cs[indx]
            succ = all(c==bab_01)

            if succ:
                D[(n,beta,approx_factor)][tstnum*n_uniq_targets+indx][1] = True #batch no. tstnum*ntests+indx successfull
                # print(f"SUCC! {indx} | {tstnum*n_uniq_targets+indx}")
    
    return D



if __name__ == '__main__':
    verbose = True
    parser = get_parser()
    args = parser.parse_args()

    nworkers = args.nworkers # number of workers

    params = {
        "n": args.n,
        "beta": args.beta,
        "n_uniq_targets": args.n_uniq_targets,
        "ntests": args.ntests,
        "slicer_iterations": 100,
        "nrand_param": args.nrand_param,
        "approx_factor": args.approx_factor,
        "nthreads": args.nthreads,
    }

    pool = Pool(processes = nworkers )
    tasks = []

    stats_dict = {}
    for lat_index in range(args.Nlats):
        tasks.append( pool.apply_async(
            run_experiment, ( lat_index, params, stats_dict, args.verbose )
        ) )

    output = []
    for t in tasks:
        tmp = t.get()
        output.append(tmp)

    print(f"output: \n {output}")

    filename=f"tail_bdd_n{args.n}_b{args.beta}.pkl"
    with open("./lwe_instances/reduced_lattices/"+filename,"wb") as file:
        pickle.dump(output,file)
