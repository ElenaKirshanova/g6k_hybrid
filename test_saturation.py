from fpylll import FPLLL

FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever, SaturationError
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from g6k.algorithms.pump import pump
from g6k.utils.stats import dummy_tracer

from global_consts import *
import pickle, time


from LatticeReduction import LatticeReduction
from utils import * #random_on_sphere, reduce_to_fund_par_proj
from hybrid_estimator.batchCVP import batchCVPP_cost

def gen_lat(n,betamax=None,k=None,bits=11.705,seed=0,threads=1,verbose=False):
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
    return LR.basis
    

def run_exp(n,cntr,ntests,saturation_scalars,max_slicer_interations=300, nthreads=1, nrand_param=10., verbose=False):
    # saturation_scalar = SATURATION_SCALAR
    B = gen_lat( n, betamax=50, k=n//2+1, seed=cntr, threads=nthreads )

    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    param_sieve["saturation_radius"] = 4/3.
    param_sieve["saturation_ratio"] = 0.4
    G=GSO.Mat( B, float_type="dd",
          U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
          UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type) )
    g6k = Siever(G)
    
    g6k.params = param_sieve

    g6k.lll(0, n)
    g6k.initialize_local(0,0,n)
    g6k.update_gso(0, n)
    then = time.perf_counter()
    pump(g6k, dummy_tracer, 0, g6k.r, 0, saturation_error="ignore", verbose=False)
    while g6k.l!=0:
        g6k.extend_left()
    round_time = time.perf_counter() - then
    print(f"pump-{n} for sat={0.4} done in {round_time}")

    G = g6k.M
    B = G.B

    sieve_dim = n
    gh = gaussian_heuristic(G.r())
    lambda1 = min( [G.get_r(0, 0)**0.5, gh**0.5] )

    aggregated_data = []
    #retrieve the projective sublattice
    B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim,scale_fact=gh), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
    for saturation_scalar in saturation_scalars:
        param_sieve = SieverParams()
        param_sieve['threads'] = nthreads
        param_sieve["sieve"] = "bdgl2"
        param_sieve["saturation_radius"] = 4/3.
        param_sieve["saturation_ratio"] = saturation_scalar
        g6k.params = param_sieve

        then = time.perf_counter()
        try:
            g6k()
        except SaturationError:
            print(f"SaturationError@ {saturation_scalar}. ignoring...")
            pass
        round_time = time.perf_counter()-then
        if verbose: print(f"Sieve-{n} for sat={saturation_scalar} done in {round_time}")

        D = {}
        Ds = []
        for approx_fact in approx_facts:
            nsucc_slic, nsucc_bab = 0, 0
            for tstnum in range(ntests):

                if verbose: print(f" - - - {approx_fact} Lat #{cntr} | #{tstnum} out of {ntests} - - - sat: {saturation_scalar}", flush=True)
                c = [ randrange(-2,3) for j in range(n) ]
                e = np.array( random_on_sphere(n,approx_fact*lambda1) )
                b = np.array( B.multiply_left( c ) )
                t = b+e

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

                    blocks = 2
                    sp, buckets = init_slicer_params(sieve_dim,blocks)


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
                        out_gs_reduced = np.array( tmp )  #cdb[0]
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
                    #then prey, this is not a devastating segfault
                    print(excpt)
                    raise excpt

            D[(n,approx_fact)] = 1.0*nsucc_slic / ntests
            Ds.append(D)
            if verbose: print( f"Experiments for approx_fact={approx_fact} done...", flush=True)
        if verbose: print( f"Experiments for nrand_param={nrand_param} done...", flush=True)
        aggregated_data.append([saturation_scalar, Ds]) 
    return aggregated_data

if __name__=="__main__":

    ###
    # [n, beta_max]
    # [60, 53], [70, 60], [80, 70], [90, 80], [100, 85]
    ###
    nthreads = 1
    nworkers = 5
    max_slicer_interations = 300
    ntests = 20
    nlats = 10
    n = 60
    bits = 11.705
    betamax = 50
    approx_facts = [ 0.9 + 0.02*i for i in range(6) ]
    nrand_param = 10.
    sats = [ 0.5, 0.75, 0.99 ]
    verbose = True

    aggregated_data = []

    tasks = []
    output = []
    pool = Pool( processes = nworkers )
    for cntr in range(nlats):
        tasks.append( pool.apply_async(
            run_exp, (n,cntr,ntests,sats,max_slicer_interations, nthreads, nrand_param, verbose)
            ) )
        print(cntr)

    for t in tasks:
        aggregated_data += [ t.get() ]
    pool.close()

    for tmp in aggregated_data:
        print(f"nrand_parameter: {aggregated_data[0]}")
        print(aggregated_data[1])

    filename = f"slicsucc_sat_{n}.pkl"
    with open(filename,"wb") as file:
        pickle.dump(aggregated_data, file)
    print( f"saved in {filename}" )