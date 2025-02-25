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

def find_vect_in_list(v,l,tolerance=1.0e-6):
    assert len(v) == len(l[0]), f"Shapes do not allign! {len(v)} vs. {len(l[0])}"
    mindiff = float("inf")
    # print(f"debug v: {v}")
    for i in range(len(l)):
        # print(f"debug ti: {l[i]}")
        tmp = np.abs( np.array(v)-np.array(l[i]) )
        # print(f"tmp: {tmp}")
        mindiff = min( mindiff, max(tmp) )
        if (mindiff<tolerance):
            # print(f"mindiff: {mindiff}")
            return i
    print(f"FAIL mindiff: {mindiff}")
    return None

def test_batch(params):
    slicer_interations = params[ "slicer_interations" ]
    norm_slack = params[ "norm_slack" ]      #terminate slicer if norm_slack*||e_projected|| is found
    approx_factor = params[ "approx_factor" ]
    n_targets = params[ "n_targets" ]
    nrand_param = params[ "nrand_param" ]
    saturation_scalar = params["saturation_scalar"]
    nthreads = params[ "nthreads" ]
    nexp = params[ "nexp" ]
    n, betamax, sieve_dim = params["n"], params["betamax"], params["sieve_dim"] 
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
    for expnum in range(nexp):
        N = GSO.Mat( G.B[:n-sieve_dim], float_type=ft )
        N.update_gso()

        unique_targets = []
        unique_answers = []
        unique_t_gs_reduced = []
        succbab_list = []
        succsli_list = []
        max_proj_dist_sq = 0.
        for cntr in range(n_targets):
            c =np.array( [ randrange(-33,34) for j in range(n) ] )
            unique_answers.append( c )
            e = np.array( random_on_sphere(n,approx_factor*gh**0.5) )
            e = np.round(e)

            print(f"gauss: {gh**0.5} vs r_00: {G.get_r(0,0)**0.5} vs ||err||: {(e@e)**0.5}")

            e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh_sub) ) #,scale_fact=gh_sub
            # e_llr = np.array( from_canonical_scaled(G,e,scale_fact=gh_sub) ) #,scale_fact=gh_sub
            dist_sq_bnd = e_@e_
            max_proj_dist_sq = max(max_proj_dist_sq,dist_sq_bnd)
            print(f"projected (e_@e_): {(dist_sq_bnd)} | max: {max_proj_dist_sq} @ #{cntr} out of {n_targets}")
            es_.append((e_@e_)**0.5)

            b = G.B.multiply_left( c )
            b_ = np.array(b,dtype=np.int64)
            t_ = e+b_
            t = np.array( [ int(tt) for tt in t_ ] )
            unique_targets.append( t )

            t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh_sub )
            #retrieve the projective sublattice
            B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh_sub), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
            t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
            unique_t_gs_reduced.append( np.array(t_gs_reduced) )
            # t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer
            # - - - prelim check - - -
            out = to_canonical_scaled( G,t_gs_reduced,offset=sieve_dim,scale_fact=gh_sub )


            bab_1 = G.babai(t-np.array(out),start=n-sieve_dim) #last sieve_dim coordinates of s
            tmp = t - np.array( G.B[-sieve_dim:].multiply_left(bab_1) )
            tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=n-sieve_dim ) ) #project onto span(B[-sieve_dim:])
            bab_0 = N.babai(tmp)

            bab_01=np.array( bab_0+bab_1 )
            succbab = all(c==bab_01)
            print(f"Babai Success: {succbab}")
            # - - - end prelim check - - -
            succbab_list.append(succbab)
            if succbab:
                succsli_list.append(True)
                nbab_succ += 1
                nsli_succ += 1
                unique_targets.pop(-1)
                unique_answers.pop(-1)
                unique_t_gs_reduced.pop(-1)

        if not all(succbab_list):
            # filename = f"bdgl2_n{n}_b{sieve_dim}.pkl"
            # g6k.dump_on_disk( filename )
            #then = perf_counter()

            #out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000)
            slicer = RandomizedSlicer(g6k)
            slicer.set_nthreads(nthreads)
            print("dbsize", g6k.db_size())

            nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
            nrand = ceil(nrand_param*(1./nrand_)**sieve_dim) #min( 250, target_list_size / len(target_candidates ) )
            print(f"nrand:{nrand}")

            for t_gs_reduced in unique_t_gs_reduced:
                slicer.grow_db_with_target([float(tt) for tt in t_gs_reduced], n_per_target=nrand)
            iterator = slicer.itervalues_cdb_t()
            # print(f"- - - - debug !!!! - - - ")
            # for tmp, tmp_0 in iterator:
            #     print(f"tmp0: {tmp_0}")

            blocks = 2 # should be the same as in siever
            blocks = min(3, max(1, blocks))
            blocks = min(int(sieve_dim / 28), blocks)
            sp = SieverParams()
            N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
            buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
            buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
            buckets = max(buckets, 2**(blocks-1))

            #print("blocks: ", blocks, " buckets: ", buckets )

            slicer.set_proj_error_bound(norm_slack*max_proj_dist_sq)
            slicer.set_max_slicer_interations(slicer_interations)
            slicer.set_Nt(len(unique_targets))
            slicer.set_saturation_scalar(saturation_scalar)

            then = time.perf_counter()
            slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], True)
            endtime = time.perf_counter()-then
            print(f"slicer w. nthreads: {nthreads} done in {endtime}")
            runtimes.append( endtime )

            iterator = slicer.itervalues_cdb_t()
            out_gs_reduced = None
            target_index_list=[]
            for tmp, tmp_0 in iterator:
                # print(tmp)
                # print(f"tmp0: {tmp_0}")
                out_gs_reduced = np.array(tmp)  #cdb[0]
                corr_t_gs = np.array(list(tmp_0)) 
                assert not( out_gs_reduced is None ), "itervalues_cdb_t is empty"
                target_index = find_vect_in_list(corr_t_gs,unique_t_gs_reduced)

                if not target_index in target_index_list:
                    t = unique_targets[target_index]
                    c = unique_answers[target_index]

                    out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
                    bab_01 = np.array( G.babai( np.array(t)-out ) )

                    # - - - Check - - - -
                    # print(f"e_: {e_}")
                    # print(f"e_llr: {e_llr}")
                    # print(f"out_gs_reduced-e_llr[-sieve_dim:]: {np.concatenate( [out_gs_reduced] ) - e_llr[-sieve_dim:]}")
                    # print(f"|e_|: {(e_@e_)**0.5} vs. {G.get_r(n-sieve_dim, n-sieve_dim)**0.5/gh_sub}")

                    succ = all(c==bab_01)
                    print(f"{c==bab_01}")
                    print(f"Success: {(succ)}")
                    if succ:
                        target_index_list.append(target_index) #if we succseed, this target is dealt with
                        nsli_succ+=1
                if len(target_index_list) >= len(unique_targets) or (out_gs_reduced@out_gs_reduced)>norm_slack*max_proj_dist_sq:
                    break
            print(f"nbab_succ, nsli_succ: {nbab_succ,nsli_succ+nbab_succ} out of {(1+expnum)*n_targets}")
            print(f"es_: {sorted(es_)}")
            print(f"MEAN: {np.mean(runtimes)}")
            print(runtimes)
    return nbab_succ, nsli_succ

if __name__ == "__main__":

    slicer_interations = 250
    norm_slack = 1.01      #terminate slicer if norm_slack*||e_projected|| is found
    approx_factor = 0.41
    n_targets = 20
    saturation_scalar = 1.01
    nrand_param = 5 #5
    nthreads = 5
    nexp = 5

    FPLLL.set_precision(200)
    n, betamax, sieve_dim = 128, 53, 70

    params = {
        "slicer_interations" : slicer_interations,
        "norm_slack" : norm_slack,
        "approx_factor" : approx_factor,
        "n_targets" : n_targets,
        "saturation_scalar": saturation_scalar,
        "nrand_param" : nrand_param,
        "nthreads" : nthreads,
        "nexp" : nexp,
        "n" : n,
        "betamax" : betamax,
        "sieve_dim" : sieve_dim,
    }

    test_batch(params)
    
    