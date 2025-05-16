
"""
BKZ-beta reduces nlats lattices. Performs ntests CVP tests on each lattice a given dimension n.
Each test is performed for 11 approximation factors for each of 3 nrerands.

python cvpp_exp.py --n 70 --betamax 60 --ntests 50 --nlats 50 --nthreads 5 --nworkers 5
python cvpp_exp.py --n 80 --betamax 70 --ntests 50 --nlats 50 --nthreads 5 --nworkers 5
"""

from fpylll import FPLLL

FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever, SaturationError
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
import argparse

from global_consts import *


from lattice_reduction import LatticeReduction
from utils import * #random_on_sphere, reduce_to_fund_par_proj
from hybrid_estimator.batchCVP import batchCVPP_cost

def get_parser():
    parser = argparse.ArgumentParser(
        description="CVPP experiments."
    )
    parser.add_argument(
    "--nthreads", default=1, type=int, help="Threads per slicer."
    )
    parser.add_argument(
    "--nworkers", default=1, type=int, help="Number of workers for experiments."
    )
    parser.add_argument(
    "--ntests", default=1, type=int, help="Number of CVP instances per lattice."
    )
    parser.add_argument(
    "--nlats", default=1, type=int, help="TNumber of lattices."
    )
    parser.add_argument(
    "--n", default=60, type=int, help="Lattice dimension"
    )
    parser.add_argument(
    "--betamax", default=30, type=int, help="Lattice dimension"
    )
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    return parser

def gen_cvpp_g6k(n,betamax=None,k=None,bits=11.705,seed=0,threads=1,verbose=False):
    #TODO: consider if we may load an already reduced basis and extend the context
    betamax=n if betamax is None else betamax
    k = n//2 if k is None else k
    B = IntegerMatrix(n,n)
    B.randomize("qary", bits=bits, k = k)

    LR = LatticeReduction( B )
    then = perf_counter()
    for beta in range(5,betamax+1):
        LR.BKZ(beta)

    if verbose: print(f"BKZ-{betamax} done in {perf_counter()-then}", flush=True)

    int_type = LR.gso.B.int_type
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")
    G = GSO.Mat( LR.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
    param_sieve = SieverParams()
    param_sieve['threads'] = threads
    param_sieve['db_size_base'] = (4/3.)**0.5 #(4/3.)**0.5 ~ 1.1547
    param_sieve['db_size_factor'] = 3.2 #3.2
    param_sieve['saturation_ratio'] = 0.5
    param_sieve['saturation_radius'] = 1.32

    g6k = Siever(G,param_sieve)
    g6k.initialize_local(0,0,n)
    then=perf_counter()
    try:
        g6k(alg="bdgl2")
    except SaturationError:
        pass
    if verbose: print(f"bdgl2-{n} done in {perf_counter()-then}")
    g6k.M.update_gso()
    g6k.dump_on_disk(f"cvppg6k_n{n}_{seed}_test.pkl")

def run_exp(n,cntr,ntests,approx_facts,max_slicer_interations=300, nthreads=1, nrand_params=[1.], verbose=False):
    saturation_scalar = SATURATION_SCALAR
    g6k = Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl")
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k.params = param_sieve

    G = g6k.M
    B = G.B

    sieve_dim = n
    gh = gaussian_heuristic(G.r())
    lambda1 = min( [G.get_r(0, 0)**0.5, gh**0.5] )

    aggregated_data = []
    #retrieve the projective sublattice
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
    for nrand_param in nrand_params:
        D = {}
        Ds = []
        for approx_fact in approx_facts:
            nsucc_slic, nsucc_bab = 0, 0
            nsucc_slic_apprcvp = 0
            for tstnum in range(ntests):

                if verbose: print(f" - - - {approx_fact} #{tstnum} out of {ntests} - - - nrand: {nrand_param}", flush=True)
                c = [ randrange(-2,3) for j in range(n) ]
                e = np.array( random_on_sphere(n,approx_fact*lambda1) )
                b = np.array( B.multiply_left( c ) )
                t = b+e

                """
                Testing Babai.
                """
                ctmp = G.babai( t )
                tmp = B.multiply_left( ctmp )
                err = tmp-b
                succ_bab = (err@err)<10**-6
                if succ_bab:
                    nsucc_bab += 1
                    nsucc_slic += 1
                    nsucc_slic_apprcvp += 1

                """
                Testing Slicer.
                """
                if not succ_bab:
                    sieve_dim = n

                    try:
                        e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim,scale_fact=gh) )
                        gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )

                        t_gs = from_canonical_scaled( G,t,offset=sieve_dim,scale_fact=gh )
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

                        slicer.set_proj_error_bound(1.01*(e_@e_))
                        slicer.set_max_slicer_interations(max_slicer_interations)
                        slicer.set_Nt(1)
                        slicer.set_saturation_scalar(saturation_scalar)
                        slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], False) # last argument - verbosity

                        iterator = slicer.itervalues_cdb_t()
                        succ = False
                        attemptcntr = 0
                        for tmp, _ in iterator:
                            attemptcntr += 1
                            out_gs_reduced = np.array( tmp ) 
                            if (out_gs_reduced@out_gs_reduced)>1.01*(e_@e_):
                                break

                            # - - - Check - - - -
                            e_ = np.array(e_)
                            out_gs_reduced = np.array( out_gs_reduced )

                            out = to_canonical_scaled( G,np.concatenate( [(G.d-sieve_dim)*[0], out_gs_reduced] ), scale_fact=gh_sub )
                            bab_01 = np.array( G.babai( np.array(t)-out ) )

                            succ = all(c==bab_01)
                            if succ:
                                break

                        if succ:
                            nsucc_slic += 1


                    except Exception as excpt: #if slicer fails for some reason,
                        #then pray, this is not a devastating segfault
                        print(excpt)
                        raise excpt

            D[(n,approx_fact)] = (0, 1.0*nsucc_slic / ntests, 1.0*nsucc_bab / ntests, 1.0*nsucc_slic_apprcvp / ntests)
            Ds.append(D)
            if verbose: print( f"Experiments for approx_fact={approx_fact} done...", flush=True)
        if verbose: print( f"Experiments for nrand_param={nrand_param} done...", flush=True)
        aggregated_data.append([nrand_param, Ds]) 
    return aggregated_data

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    nthreads = args.nthreads
    nworkers = args.nworkers
    max_slicer_interations = 300
    ntests = args.ntests
    nlats = args.nlats
    n = args.n
    bits = 11.705
    betamax = args.betamax
    approx_facts = [ 0.9 + 0.02*i for i in range(6) ]
    nrand_params = [ 1.0,5.0,10.0 ]
    verbose = args.verbose
    

    to_be_computed = []
    g6ks = []
    load_succ = True
    for cntr in range(nlats):
        try:
            Siever.restore_from_file(f"cvppg6k_n{n}_{cntr}_test.pkl")
        except FileNotFoundError:
            load_succ = False
            to_be_computed.append( (cntr,n,betamax,None,bits) )

    tasks = []
    output = []
    pool = Pool( processes = nworkers )
    for cntr,n,betamax,k,bits in to_be_computed:
        tasks.append( pool.apply_async(
            gen_cvpp_g6k, (n, betamax, k, bits, cntr, nthreads, verbose)
            ) )
    for t in tasks:
         t.get()

    pool.close()
    aggregated_data = []

    tasks = []
    output = []
    pool = Pool( processes = nworkers )
    for cntr in range(nlats):
        tasks.append( pool.apply_async(
            run_exp, (n,cntr,ntests,approx_facts,max_slicer_interations, nthreads, nrand_params, verbose)
            ) )
        print(cntr)

    for t in tasks:
        aggregated_data += [ t.get() ]
    pool.close()

    filename = f"slicsucc_{n}.pkl"
    with open(filename,"wb") as file:
        pickle.dump("./lwe_instances/reduced_lattices/"+aggregated_data, file)
    print( f"saved in {filename}" )