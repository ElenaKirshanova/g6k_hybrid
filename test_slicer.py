from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from hybrid_estimator.batchCVP import batchCVPP_cost
from utils import *
import sys

import numpy as np

import time

if __name__ == "__main__":

    slicer_interations = 250
    norm_slack = 1.01      #terminate slicer if norm_slack*||e_projected|| is found
    approx_factor = 0.83
    nrand_param = 5
    nthreads = 5
    nexp = 2

    FPLLL.set_precision(200)
    n, betamax, sieve_dim = 80, 53, 72
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
            bkz.BKZ(beta,tours=5)
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
    gh = min( [G.r()[0], gaussian_heuristic(G.r())] )
    gh_sub = gaussian_heuristic(G.r()[-sieve_dim:]) #min( [G.r()[-sieve_dim], gaussian_heuristic(G.r()[-sieve_dim:])] )
    print(f"gh: {gh**0.5}, gh_sub: {gh_sub**0.5}")


    print(f"dbsize: {len(g6k)}")

    nbab_succ, nsli_succ = 0, 0
    runtimes=[]

    es_ = []
    for _ in range(nexp):
        c = [ randrange(-33,34) for j in range(n) ]
        # e = np.array( [ randrange(-8,9) for j in range(n) ],dtype=np.int64 )
        e = np.array( random_on_sphere(n,approx_factor*gh**0.5) )
        e = np.round(e)

        print(f"gauss: {gh**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh_sub) ) #,scale_fact=gh_sub
        e_llr = np.array( from_canonical_scaled(G,e,scale_fact=gh_sub) ) #,scale_fact=gh_sub
        dist_sq_bnd = e_@e_,
        print(f"projected (e_@e_): {(e_@e_)} vs r/gh: {G.get_r(n-sieve_dim, n-sieve_dim)/gh}")
        print("projected target squared length:", (e_@e_))

        print(f"e_: {e_}")

        b = G.B.multiply_left( c )
        b_ = np.array(b,dtype=np.int64)
        t_ = e+b_
        t = [ int(tt) for tt in t_ ]

        # param_sieve = SieverParams()
        # param_sieve['threads'] = 4
        # g6k = Siever(G,param_sieve)
        # g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
        # print("Running bdgl2...")
        # g6k(alg="bdgl2")
        # g6k.M.update_gso() 
        #
        # print(f"dbsize: {len(g6k)}")

        #assert(False)

        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh_sub )
        #print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")
        #retrieve the projective sublattice
        B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh_sub), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
        t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
        t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer

        # t_gs_non_scaled = G.from_canonical(t)[-sieve_dim:]
        # shift_babai_c = G.babai((n-sieve_dim)*[0] + list(t_gs_non_scaled), start=n-sieve_dim,gso=True)
        # shift_babai = G.B.multiply_left( (n-sieve_dim)*[0] + list( shift_babai_c ),scale_fact=gh_sub )
        # t_gs_reduced = from_canonical_scaled( G,np.array(t)-shift_babai,offset=sieve_dim,scale_fact=gh_sub ) #this is the actual reduced target
        # t_gs_shift = from_canonical_scaled( G,shift_babai,offset=sieve_dim,scale_fact=gh_sub )


        # t_gs_reduced = t_gs
        # t_gs_shift = t_gs-t_gs_reduced
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
        print(f"Babai Success: {succbab}")
        # - - - end prelim check - - -
        # - - - extra check - - -
        bab_t = np.array( g6k.M.babai(t) )
        #print(f"Coeffs of b found: {(c==bab_t)}")
        succ = all(c==bab_t)
        print(f"Final Babai Success: {succ}")
        if succ:
            print(f"t_gs_reduced: {t_gs_reduced}")
            nbab_succ+=1
        # else:
        #     print(c==bab_t)
        # - - - end extra check - - -

        if not succ:
            # filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
            # g6k.dump_on_disk( filename )
            #then = perf_counter()

            #out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000)
            slicer = RandomizedSlicer(g6k)
            slicer.set_nthreads(2)

            print("target:", [float(tt) for tt in t_gs_reduced])
            print("dbsize", g6k.db_size())

            nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
            nrand = ceil(nrand_param*(1./nrand_)**sieve_dim) #min( 250, target_list_size / len(target_candidates ) )
            # nrand = 6000
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

            #print("blocks: ", blocks, " buckets: ", buckets )

            slicer.set_proj_error_bound(norm_slack*(e_@e_))
            # slicer.set_lifted_error_bound(1.01*(e_@e_))
            slicer.set_max_slicer_interations(slicer_interations)

            then = time.perf_counter()
            slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], True)
            endtime = time.perf_counter()-then
            print(f"slicer w. nthreads: {nthreads} done in {endtime}")
            runtimes.append( endtime )

            iterator = slicer.itervalues_cdb_t()
            out_gs_reduced = None
            for tmp in iterator:
                out_gs_reduced = np.array(tmp)  #cdb[0]
                break
            assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"
            # out_gs = out_gs_reduced + t_gs_shift
            # out = to_canonical_scaled( G,out_gs,offset=sieve_dim,scale_fact=gh_sub )
            # N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
            # N.update_gso()
            # bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
            # tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
            # tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
            # bab_0 = N.babai(tmp)
            # bab_01=np.array( bab_0+bab_1 )

            out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
            bab_01 = np.array( G.babai( np.array(t)-out ) )

            # - - - Check - - - -
            # print(f"e_: {e_}")
            print(f"e_llr: {e_llr}")
            print(f"out_gs_reduced-e_llr[-sieve_dim:]: {np.concatenate( [out_gs_reduced] ) - e_llr[-sieve_dim:]}")
            print(f"|e_|: {(e_@e_)**0.5} vs. {G.get_r(n-sieve_dim, n-sieve_dim)**0.5/gh_sub}")
            es_.append((e_@e_)**0.5)

            succ = all(c==bab_01)
            print(f"{c==bab_01}")
            print(f"Success: {(succ)}")
            if succ:
                nsli_succ+=1
            print(f"both succeded: {succ and succbab}", flush=True)
        print(f"nbab_succ, nsli_succ: {nbab_succ,nsli_succ+nbab_succ} out of {nexp}")
        print(f"es_: {sorted(es_)}")
        print(f"MEAN: {np.mean(runtimes)}")
        print(runtimes)
