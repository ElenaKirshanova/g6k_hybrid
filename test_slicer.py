from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
import argparse
from g6k.slicer import RandomizedSlicer
from hybrid_estimator.batchCVP import batchCVPP_cost
from utils import *
import sys

from lattice_reduction import LatticeReduction

import numpy as np

import time

def get_parser():
    parser = argparse.ArgumentParser(
        description="Test run for slicer."
    )
    parser.add_argument(
    "--approx_factor", default=0.95, type=float, help="CVP approx factor"
    )
    parser.add_argument(
    "--nthreads", default=1, type=int, help="Threads per slicer."
    )
    parser.add_argument(
    "--n", default=50, type=int, help="Lattice dimension"
    )
    parser.add_argument(
    "--nexp", default=5, type=int, help="Number of experiments"
    )
    parser.add_argument(
    "--betamax", default=30, type=int, help="BKZ blocksize"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
   
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    slicer_interations = 140
    norm_slack = 1.01      #terminate slicer if norm_slack*||e_projected|| is found
    approx_factor = args.approx_factor
    nrand_param = 10
    nthreads = args.nthreads
    nexp = args.nexp
    verbose = args.verbose
    slicer_verbosity = True


    FPLLL.set_precision(200)
    n, betamax,  = args.n, args.betamax
    sieve_dim = n
    ft = "ld" if n<90 else ( "dd" if config.have_qd else "mpfr")
    # - - - try load a lattice - - -
    filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
    nothing_to_load = True
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    try:
        g6k = Siever.restore_from_file(filename)
        g6k.params = param_sieve
        G = g6k.M
        B = G.B
        nothing_to_load = False
        if verbose: print(f"Load seems to succeed...")
    except Exception as excpt:
        print(excpt)
        pass
    # - - - end try load a lattice - - -

    # - - - Make all fpylll objects - - -
    if nothing_to_load:
        if verbose: print(f"Nothing to load. Computing")
        B = IntegerMatrix(n,n)
        B.randomize("qary", k=n//2, bits=11.705)
        G = GSO.Mat(B, float_type=ft)
        G.update_gso()

        if sieve_dim<30: print("Slicer is not implemented on dim < 30") #TODO: change to warning
        if sieve_dim<40: print("LSH won't work on dim < 40") #TODO: change to warning

        lll = LLL.Reduction(G)
        lll()

        bkz = LatticeReduction(B)
        for beta in range(5,betamax+1):
            then_round=time.perf_counter()
            bkz.BKZ(beta,tours=5)
            round_time = time.perf_counter()-then_round
            if verbose: print(f"BKZ-{beta} done in {round_time}")
            sys.stdout.flush()

        int_type = bkz.gso.B.int_type
        G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
        G.update_gso()
        lll = LLL.Reduction( G )
        lll()

        g6k = Siever(G)
        g6k.params = param_sieve
        g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
        if verbose: print("Running bdgl2...")
        then = time.perf_counter()
        g6k(alg="bdgl2")
        print(f"siever done in {time.perf_counter()-then}")
        g6k.M.update_gso()
        g6k.dump_on_disk( filename )
    # - - - end Make all fpylll objects - - -
    gh = min( [G.r()[0], gaussian_heuristic(G.r())] )
    gh_sub = gaussian_heuristic(G.r()[-sieve_dim:]) 
    if verbose: print(f"gh: {gh**0.5}, gh_sub: {gh_sub**0.5}")


    if verbose: print(f"dbsize: {len(g6k)}")

    nbab_succ, nsli_succ = 0, 0
    runtimes=[]

    es_ = []
    for ctr_experiment in range(nexp):
        c = [ randrange(-33,34) for j in range(n) ]
        e = np.array( random_on_sphere(n,approx_factor*gh**0.5) )
        e = np.round(e)

        if verbose: print(f"gauss: {gh**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh_sub) ) #,scale_fact=gh_sub
        e_llr = np.array( from_canonical_scaled(G,e,scale_fact=gh_sub) ) #,scale_fact=gh_sub
        dist_sq_bnd = e_@e_,
        if verbose: print(f"projected (e_@e_): {(e_@e_)} vs r/gh: {G.get_r(n-sieve_dim, n-sieve_dim)/gh}")
        if verbose: print("projected target squared length:", (e_@e_))

        if verbose: print(f"e_: {e_}")

        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t_ = e+b_
        t = [ int(tt) for tt in t_ ]

        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh_sub )
        #retrieve the projective sublattice
        B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh_sub), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
        t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

        # - - - prelim check - - -
        out = to_canonical_scaled( G,t_gs_reduced,offset=sieve_dim,scale_fact=gh_sub )


        N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
        N.update_gso()
        bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
        tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
        tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
        bab_0 = N.babai(tmp)

        bab_01=np.array( bab_0+bab_1 )
        succbab = all(c==bab_01)
        if verbose: print(f"Babai Success: {succbab}")
        # - - - end prelim check - - -
        # - - - extra check - - -
        bab_t = np.array( g6k.M.babai(t) )
        #print(f"Coeffs of b found: {(c==bab_t)}")
        succ = all(c==bab_t)
        if verbose: print(f"Final Babai Success: {succ}")
        if succ:
            if verbose: print(f"t_gs_reduced: {t_gs_reduced}")
            nbab_succ+=1
        # - - - end extra check - - -

        if not succ:
            slicer = RandomizedSlicer(g6k)
            slicer.set_nthreads(2)

            if verbose: print("target:", [float(tt) for tt in t_gs_reduced])
            if verbose: print("dbsize", g6k.db_size())

            nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
            nrand = ceil(nrand_param*(1./nrand_)**sieve_dim)
            print(f"nrand:{nrand}")
            slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)

            blocks = 2 # should be the same as in siever
            blocks = min(3, max(1, blocks))
            blocks = min(int(sieve_dim / 28), blocks)
            sp = SieverParams()
            N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
            buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
            buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
            buckets = max(buckets, 2**(blocks-1))


            slicer.set_proj_error_bound(norm_slack*(e_@e_))
            slicer.set_max_slicer_interations(slicer_interations)
            slicer.set_Nt(1)
            slicer.set_saturation_scalar(1.05)
            filename = ("cdbt_dim_n"+str(n)+"_beta"+str(betamax)+"_sdim"+str(sieve_dim)+"_"+str(ctr_experiment)+"_").encode('utf-8')

            then = time.perf_counter()
            slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], True, True)
            endtime = time.perf_counter()-then
            if verbose: print(f"slicer w. nthreads: {nthreads} done in {endtime}")
            runtimes.append( endtime )

            iterator = slicer.itervalues_cdb_t()
            out_gs_reduced = None
            for tmp, _ in iterator:
                out_gs_reduced = np.array(tmp)  #cdb[0]
                break
            assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"

            out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
            bab_01 = np.array( G.babai( np.array(t)-out ) )

            # - - - Check - - - -
            if verbose: print(f"e_llr: {e_llr}")
            if verbose: print(f"out_gs_reduced-e_llr[-sieve_dim:]: {np.concatenate( [out_gs_reduced] ) - e_llr[-sieve_dim:]}")
            if verbose: print(f"|e_|: {(e_@e_)**0.5} vs. {G.get_r(n-sieve_dim, n-sieve_dim)**0.5/gh_sub}")
            es_.append((e_@e_)**0.5)

            succ = all(c==bab_01)
            if verbose: print(f"{c==bab_01}")
            if verbose: print(f"Success: {(succ)}")
            if succ:
                nsli_succ+=1
            if verbose: print(f"both succeeded: {succ and succbab}", flush=True)

        if verbose: print(f"es_: {sorted(es_)}")
        if verbose: print(f"MEAN: {np.mean(runtimes)}")
        if verbose: print(runtimes)
    print(f"nbab_succ, nsli_succ: {nbab_succ,nsli_succ+nbab_succ} out of {nexp}")
