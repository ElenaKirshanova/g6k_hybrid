from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from hybrid_estimator.batchCVP import batchCVPP_cost
from utils import *
import sys
import numpy as np
import time, pickle

from global_consts import *

def load_or_gen_lat(n,latind,sieve_dim, betamax):
    ft = "ld" if n<90 else ( "dd" if config.have_qd else "mpfr")
    # - - - try load a lattice - - -
    filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
    nothing_to_load = True
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    try:
        g6k = Siever.restore_from_file(filename)
        g6k.params = param_sieve
        nothing_to_load = False
        print(f"Load seems to succseed...")
    except Exception as excpt:
        print(excpt)
        pass
    # - - - end try load a lattice - - -

    # - - - Make all fpylll objects - - -
    if nothing_to_load:
        print(f"Nothing to load. Computing")
        B = IntegerMatrix(n,n)
        B.randomize("qary", k=n//2, bits=11.705)
        G = GSO.Mat(B, float_type=ft)
        G.update_gso()

        if sieve_dim<30: print("Slicer is not implemented on dim < 30")
        if sieve_dim<40: print("LSH won't work on dim < 40")

        lll = LLL.Reduction(G)
        lll()

        bkz = LatticeReduction(B)
        for beta in range(5,betamax+1):
            then_round=time.perf_counter()
            bkz.BKZ(beta)
            round_time = time.perf_counter()-then_round
            print(f"BKZ-{beta} done in {round_time}")
            sys.stdout.flush()

        int_type = bkz.gso.B.int_type
        G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
        G.update_gso()
        lll = LLL.Reduction( G )
        lll()

        g6k = Siever(G)
        g6k.params = param_sieve
        g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
        print("Running bdgl2...")
        then = time.perf_counter()
        g6k(alg="bdgl2")
        print(f"siever done in {time.perf_counter()-then}")
        g6k.M.update_gso()
        # filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
        g6k.dump_on_disk( filename )
    # - - - end Make all fpylll objects - - -
    return g6k

slicer_interations = 256
nrand_param = 5
nthreads = 5
nexp = 20
nlats = 2

n, betamax, sieve_dim = 65, 50, 65
stat_dict = {}

for latind in range(nlats):
    g6k = load_or_gen_lat( n, latind, sieve_dim, betamax )
    G = g6k.M

    gh = min( [G.r()[0], gaussian_heuristic(G.r())] )
    gh_sub = gaussian_heuristic(G.r()[-sieve_dim:]) #min( [G.r()[-sieve_dim], gaussian_heuristic(G.r()[-sieve_dim:])] )
    print(f"gh: {gh**0.5}, gh_sub: {gh_sub**0.5}")
    print(f"dbsize: {len(g6k)}")

    dblens = []
    iterator = g6k.itervalues()
    for tmp in iterator:
        coeffs = np.array(tmp)  #cdb[0]
        vi = np.array( G.B.multiply_left( coeffs ) )
        vi_gs = from_canonical_scaled( G,vi,offset=sieve_dim,scale_fact=gh_sub )
        dblens.append( vi_gs@vi_gs )

    stat_dict[n,latind] = {
        "dblens": dblens,
        "succs": [],
        "niters": [],
        "nrmt_evolution": [],
    }

    es_ = []
    for _ in range(nexp):
        c = [ randrange(-33,34) for j in range(n) ]
        e = np.array( random_on_sphere(n,0.9*gh**0.5) )
        e = np.round(e)
        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t = e+b_

        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh_sub) )
        dist_sq_bnd = e_@e_

        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh_sub )
        #retrieve the projective sublattice
        B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh_sub), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
        t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

        slicer = RandomizedSlicer(g6k)
        slicer.set_nthreads(nthreads)
        nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
        nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
        slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

        blocks = 2 # should be the same as in siever
        blocks = min(3, max(1, blocks))
        blocks = min(int(sieve_dim / 28), blocks)
        sp = SieverParams()
        N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
        buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
        buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
        buckets = max(buckets, 2**(blocks-1))

        slicer.set_proj_error_bound(EPS2*dist_sq_bnd)
        slicer.set_max_slicer_interations(slicer_interations)
        then = time.perf_counter()
        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False)
        endtime = time.perf_counter()-then
        print(f"slicer w. nthreads: {nthreads} done in {endtime}")

        iterator = slicer.itervalues_cdb_t()
        out_gs_reduced = None
        for tmp in iterator:
            out_gs_reduced = np.array(tmp)  #cdb[0]
            break
        assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"
        print(f"|out_gs_reduced|: {out_gs_reduced@out_gs_reduced}")

        succ = out_gs_reduced@out_gs_reduced <= dist_sq_bnd
        print(f"Success: {succ}")

        stat_dict[n,latind]["succs"].append({
            "succ": succ,
        })

print(stat_dict)
