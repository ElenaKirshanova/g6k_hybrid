from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys

import time, pickle
from random import shuffle

from hyb_att_on_kyber import alg_2_batched

from discretegauss import sample_dgauss
from hybrid_estimator.batchCVP import batchCVPP_cost

DGAUSS_SIGMA = 0.8
DTYPE = np.float64 #np.longdouble or np.float64

# def alg_2_batched_debug( g6k,target_candidates, dist_sq_bnd=1.0, nthreads=1, tracer_alg2=None, e=None ):
#     # raise NotImplementedError
#     sieve_dim = g6k.r-g6k.l #n_slicer_coord
#     print(f"in alg2 sieve_dim={sieve_dim}")
#
#     # dist_sq_bnd = 1.0 #TODO: implement
#     G = g6k.M
#     B = G.B
#     dim = G.d
#     Gsub = GSO.Mat( G.B[:dim-sieve_dim], float_type=G.float_type )
#     Gsub.update_gso()
#     # print(f"dim(Gsub): {Gsub.d}")
#
#     # - - - prepare Slicer for batch cvp - - -
#     slicer = RandomizedSlicer(g6k)
#     slicer.set_nthreads(1);
#     # - - - END prepare Slicer for batch cvp - - -
#     scaling_vec = np.array( [tmp**0.5 for tmp in G.r()[dim-sieve_dim:]] )
#     #WARNING: we do not store t_gs_reduced_list since t_gs_list =  t_gs - gs(shift_babai_c*B)
#     #this is a time-memory tradeoff. Since Slicer returns only an error vector, we don\'t
#     #know which of the target candidates it corresponds to. TODO: or should we?
#     target_list_size =  2 * g6k.db_size() #len(g6k)
#     nrand = 150 #min( 250, target_list_size / len(target_candidates ) )
#     print(f"len(target_candidates): {len(target_candidates)} nrand: {nrand}")
#     t_gs_list = []
#     t_gs_reduced_list = []
#     shift_babai_c_list = []
#     for target in target_candidates:
#         # print(end=".", flush=True)
#         t_gs = from_canonical_scaled( G,target,offset=sieve_dim )
#         # t_gs_non_scaled = G.from_canonical(target)[dim-sieve_dim:]
#         # shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, dimension=sieve_dim, gso=True) )
#         # print( f"shift_babai_c: {shift_babai_c}" )
#         # shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
#         # t_gs_reduced = from_canonical_scaled( G,np.array(target)-shift_babai,offset=sieve_dim ) #this is the actual reduced target
#         # assert len(t_gs_reduced) == sieve_dim
#         # assert all( abs( t_gs_reduced[dim-sieve_dim:] ) <0.501 ) #assert that the last Sieve dim coords are size reduced
#
#         B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
#         t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
#         t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer
#         shift_babai_c = G.babai((dim-sieve_dim)*[0] + list(t_gs_shift), start=dim-sieve_dim,gso=True)
#
#         t_gs_list.append(t_gs)
#         shift_babai_c_list.append(shift_babai_c)
#         t_gs_reduced_list.append(t_gs_reduced)
#
#         # print(target[dim-sieve_dim:])
#         # print(f"Doing grow_db")
#         then_gdbwt = perf_counter()
#         print(f"supposed ce.len: {t_gs_reduced@t_gs_reduced}")
#         slicer.grow_db_with_target(t_gs_reduced, n_per_target=nrand)
#         # slicer.grow_db_with_target((dim-sieve_dim)*[0] + [float(tt) for tt in t_gs_reduced[dim-sieve_dim:]], n_per_target=nrand) #add a candidate to the Slicer
#         gdbwt_t = perf_counter() - then_gdbwt #TODO: collect this stat
#         # print(f"grow_db done in {gdbwt_t}",flush=True)
#     #run slicer
#     print(f"running slicer")
#     blocks = 2 # should be the same as in siever
#     blocks = min(3, max(1, blocks))
#     blocks = min(int(sieve_dim / 28), blocks)
#     sp = g6k.params
#     N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
#     buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
#     buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
#     buckets = max(buckets, 2**(blocks-1))
#     slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], ((dist_sq_bnd)))
#     print(f"t_gs_reduced: {t_gs_reduced}")
#     iterator = slicer.itervalues_t()
#     for tmp in iterator:
#         out_gs_reduced = np.array(tmp)  #db_t[0] is expected to contain the error vector
#         cur_nrm_sq = out_gs_reduced@out_gs_reduced
#         break
#
#     print(f"out_gs_reduced-t_gs_reduced: {out_gs_reduced-t_gs_reduced}")
#     print(f"out_gs_reduced: {out_gs_reduced}")
#     index = 0
#     #Now we deduce which target candidate the error vector corresponds to.
#     #The idea is that if t_gs is an answer then t_gs_reduced - out_gs_reduced is in the projective lat
#     #and is (close to) zero.
#     min_norm_err_sq = float("inf")
#     index_best = None
#     b_best = None
#     for index in range(len(shift_babai_c_list)):
#         """
#         t_gs_reduced = t_gs_reduced_list[index] #we could do this to t_gs, but this one is shorter
#         shift_babai_c_reduced =  shift_babai_c_list[index]
#
#         #We guess what was the shift corresponding to the answer.
#         shift_babai_reduced = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c_reduced ) )
#         shift_babai_reduced_gs = from_canonical_scaled( G,shift_babai_reduced, offset=sieve_dim )
#         guess_gs = np.array(t_gs_reduced - out_gs_reduced) #a supposed BDD solution for t_gs_reduced
#         print(len(guess_gs),len(shift_babai_reduced_gs))
#         guess_gs = guess_gs + shift_babai_reduced_gs
#
#         t_gs = t_gs_list[index]
#         diff_gs = t_gs - guess_gs #an actual error vector we observe == actual error (+ some lattice vector for bad candidates)
#         diff_gs_nrm_sq = diff_gs@diff_gs #its norm. Ideally, == norm of error
#         """
#         print(f"LEN: {len(target_candidates)}")
#
#         t = np.array( target_candidates[index] )
#         t_1 = np.array( G.from_canonical( t,start=0 ) )
#         for i in range(dim-sieve_dim):
#             t_1[i] = 0.
#         t_1 = np.array( G.to_canonical( t_1,start=0 ) )
#         t_0 = np.array( G.from_canonical( t,start=0 ) )
#         for i in range(dim-sieve_dim, dim):
#             t_0[i] = 0.
#         t_0 = np.array( G.to_canonical( t_0,start=0 ) )
#         #we substitute the obtaied error from the target and call babai to
#         #account for an fp error
#
#         # out_reduced = to_canonical_scaled( G, np.concatenate([ (dim-sieve_dim)*[0] , out_gs_reduced ]), offset=dim )
#         out_reduced = np.array( to_canonical_scaled( G, out_gs_reduced, offset=sieve_dim ) )
#         t_1 = t_1 - out_reduced
#         bab_1 = G.babai(t_1,start=dim-sieve_dim, dimension=sieve_dim)
#
#         tmp = G.B[-sieve_dim:].multiply_left( bab_1 )
#         tmp = np.array( G.from_canonical(tmp,start=0) )
#         for i in range(dim-sieve_dim,dim):
#             tmp[i] = 0.
#         tmp = G.to_canonical( tmp, start=0 )
#         t_0 = t_0 - tmp
#         bab_0 = G.babai(t_0,start=0, dimension=n-sieve_dim)
#         bab_01 = np.concatenate( [bab_0,bab_1] )
#         solution_candidate = np.array( G.B.multiply_left( bab_01 ) )
#
#         diff = t - solution_candidate
#         diff_nrm_sq = diff@diff
#
#         if diff_nrm_sq < min_norm_err_sq:
#             min_norm_err_sq = diff_nrm_sq
#             best_index = index
#             best_solution_candidate = solution_candidate
#             best_bab_01 = bab_01
#     print(f"min_norm_err_sq: {min_norm_err_sq}")
#
#
#     print(f"alg2 terminates")
#     return best_bab_01

def alg_2_batched_debug( g6k,target_candidates, dist_sq_bnd=1.0, nthreads=1, tracer_alg2=None ):
    # raise NotImplementedError
    sieve_dim = g6k.r-g6k.l #n_slicer_coord
    print(f"in alg2 sieve_dim={sieve_dim}", flush=True)

    # dist_sq_bnd = 1.0 #TODO: implement
    G = g6k.M
    B = G.B
    dim = G.d
    Gsub = GSO.Mat( G.B[:dim-sieve_dim], float_type=G.float_type )
    Gsub.update_gso()

    # - - - prepare Slicer for batch cvp - - -
    slicer = RandomizedSlicer(g6k)
    slicer.set_nthreads(nthreads);
    # - - - END prepare Slicer for batch cvp - - -
    #WARNING: we do not store t_gs_reduced_list since t_gs_list =  t_gs - gs(shift_babai_c*B)
    #this is a time-memory tradeoff. Since Slicer returns only an error vector, we don\'t
    #know which of the target candidates it corresponds to. TODO: or should we?
    target_list_size =  2 * g6k.db_size() #len(g6k)
    nrand_, _ = batchCVPP_cost(sieve_dim,100,len(g6k)**(1./sieve_dim),1)
    nrand = ceil(2*(1./nrand_)**sieve_dim) #min( 250, target_list_size / len(target_candidates ) )
    # nrand = ceil( 0.75*len(g6k) ) #TODO: remove this in a such way that alg3 does not break
    print(f"len(target_candidates): {len(target_candidates)} nrand: {nrand}")
    t_gs_list = []
    t_gs_reduced_list = []
    shift_babai_c_list = []
    for target in target_candidates:
        t_gs = from_canonical_scaled( G,target,offset=sieve_dim )

        t_gs_non_scaled = G.from_canonical(target)[dim-sieve_dim:]
        shift_babai_c =  list( G.babai( list(t_gs_non_scaled), start=dim-sieve_dim, gso=True) )
        # print( f"shift_babai_c: {shift_babai_c}" )
        shift_babai = G.B.multiply_left( (dim-sieve_dim)*[0] + list( shift_babai_c ) )
        t_gs_reduced = from_canonical_scaled( G,np.array(target, dtype=DTYPE)-shift_babai,offset=sieve_dim ) #this is the actual reduced target


        # assert len(t_gs_reduced) == sieve_dim
        # assert all( abs( t_gs_reduced[dim-sieve_dim:] ) <0.501 ) #assert that the last Sieve dim coords are size reduced

        # B_gs = [ np.array( from_canonical_scaled(G, G.B[i], offset=sieve_dim), dtype=np.float64 ) for i in range(G.d - sieve_dim, G.d) ]
        # t_gs_reduced = reduce_to_fund_par_proj(B_gs,(t_gs),sieve_dim) #reduce the target w.r.t. B_gs
        # t_gs_shift = t_gs-t_gs_reduced #find the shift to be applied after the slicer
        # shift_babai_c = G.babai((dim-sieve_dim)*[0] + list(t_gs_shift), start=dim-sieve_dim,gso=True)

        t_gs_list.append(t_gs)
        shift_babai_c_list.append(shift_babai_c)
        t_gs_reduced_list.append(t_gs_reduced)

        then_gdbwt = perf_counter()
        slicer.grow_db_with_target(t_gs_reduced, n_per_target=nrand) #add a candidate to the Slicer
        gdbwt_t = perf_counter() - then_gdbwt #TODO: collect this stat
        # print(f"Doing grow_db")
        # for _ in range(nrand):
        #     delta = [ sample_dgauss(DGAUSS_SIGMA) for tmp in range( sieve_dim ) ]
        #     delta = np.array( g6k.M.B[-sieve_dim:].multiply_left( delta ), dtype=DTYPE )
        #     delta_gs =  from_canonical_scaled( G,delta,offset=sieve_dim )
        #     slicer.grow_db_with_target(t_gs_reduced+delta_gs, n_per_target=1)

        # print(f"grow_db done in {gdbwt_t}",flush=True)
    #run slicer
    print(f"running slicer")
    blocks = 2 # should be the same as in siever
    blocks = min(3, max(1, blocks))
    blocks = min(int(sieve_dim / 28), blocks)
    sp = g6k.params
    N = sp["db_size_factor"] * sp["db_size_base"] ** sieve_dim
    buckets = sp["bdgl_bucket_size_factor"]* 2.**((blocks-1.)/(blocks+1.)) * sp["bdgl_multi_hash"]**((2.*blocks)/(blocks+1.)) * (N ** (blocks/(1.0+blocks)))
    buckets = min(buckets, sp["bdgl_multi_hash"] * N / sp["bdgl_min_bucket_size"])
    buckets = max(buckets, 2**(blocks-1))

    slicer.bdgl_like_sieve(buckets, blocks, sp["bdgl_multi_hash"], (1.01*(dist_sq_bnd)))

    print(f"t_gs_reduced: {t_gs_reduced}")
    print(f"t_gs_reduced norm: {t_gs_reduced@t_gs_reduced}")
    iterator = slicer.itervalues_t()
    for tmp in iterator:
        out_gs_reduced = np.array(tmp, dtype=DTYPE)  #db_t[0] is expected to contain the error vector
        cur_nrm_sq = out_gs_reduced@out_gs_reduced
        break
    # print(f"cur_nrm ={cur_nrm_sq**0.5}")

    iterator = slicer.itervalues_t()
    nrms = []
    for tmp in iterator:
        tmp = np.array(tmp, dtype=DTYPE)  #db_t[0] is expected to contain the error vector
        tmp_nrm_sq = ( tmp@tmp )**0.5
        nrms.append( tmp_nrm_sq )
    # print(f"Targets nrms post: {[float(tt) for tt in nrms]}")
    setnrms = set(nrms)
    # print(setnrms)
    print(f"{len(setnrms)} out of {len(nrms)} targets are unique", flush=True)

    print(f"out_gs_reduced-t_gs_reduced: {out_gs_reduced-t_gs_reduced}")
    print(f"out_gs_reduced: {out_gs_reduced}")
    print(f"out_gs_reduced norm: {(out_gs_reduced@out_gs_reduced)**0.5} vs {dist_sq_bnd**0.5}")
    index = 0
    #Now we deduce which target candidate the error vector corresponds to.
    #The idea is that if t_gs is an answer then t_gs_reduced - out_gs_reduced is in the projective lat
    #and is (close to) zero.
    min_norm_err_sq = float("inf")
    index_best = None
    b_best = None
    for index in range(len(shift_babai_c_list)):
        # print(f"LEN: {len(target_candidates)}")

        t = np.array( target_candidates[index], dtype=DTYPE )
        t_1 = np.array( G.from_canonical( t,start=0 ), dtype=DTYPE )
        for i in range(dim-sieve_dim):
            t_1[i] = 0.
        t_1 = np.array( G.to_canonical( t_1,start=0 ), dtype=DTYPE )
        t_0 = np.array( G.from_canonical( t,start=0 ), dtype=DTYPE )
        for i in range(dim-sieve_dim, dim):
            t_0[i] = 0.
        t_0 = np.array( G.to_canonical( t_0,start=0 ), dtype=DTYPE )
        #we substitute the obtaied error from the target and call babai to
        #account for an fp error

        out_reduced = np.array( to_canonical_scaled( G, out_gs_reduced, offset=sieve_dim ), dtype=DTYPE )
        t_1 = t_1 - out_reduced
        bab_1 = G.babai(t_1,start=dim-sieve_dim, dimension=sieve_dim)

        tmp = G.B[-sieve_dim:].multiply_left( bab_1 )
        tmp = np.array( G.from_canonical(tmp,start=0), dtype=DTYPE )
        for i in range(dim-sieve_dim,dim):
            tmp[i] = 0.
        tmp = G.to_canonical( tmp, start=0 )
        t_0 = t_0 - tmp
        bab_0 = G.babai(t_0,start=0, dimension=dim-sieve_dim)
        bab_01 = np.concatenate( [bab_0,bab_1] )
        solution_candidate = np.array( G.B.multiply_left( bab_01 ), dtype=DTYPE )

        diff = t - solution_candidate
        diff_nrm_sq = diff@diff

        if diff_nrm_sq < min_norm_err_sq:
            min_norm_err_sq = diff_nrm_sq
            best_index = index
            best_solution_candidate = solution_candidate
            best_bab_01 = bab_01

    print(f"min_norm_err_sq: {min_norm_err_sq}")


    print(f"alg2 terminates")
    return best_bab_01

if __name__=="__main__":
    # n, betamax, sieve_dim = 140, 45, 45 #n=170 is liikely to fail
    nexp = 120
    n, betamax, sieve_dim = 90, 75, 55 #n=170 is liikely to fail
    print(f"n, betamax, sieve_dim: {(n, betamax, sieve_dim)}")

    bits=11.705
    ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

    loadsucc = False

    try:
        with open(f"qary_{n}_{betamax}_{bits:.4f}.pkl", "rb") as file:
            B = pickle.load( file )
        loadsucc = True
    except FileNotFoundError:
        print(f"Nothing to load. Computing")
        pass

    if not loadsucc:
        B = IntegerMatrix(n,n)
        B.randomize("qary", k=n//2, bits=bits)
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

        with open(f"qary_{n}_{betamax}_{bits:.4f}.pkl", "wb") as file:
            pickle.dump(bkz.basis, file)
        B = bkz.gso.B

    int_type = B.int_type
    G = GSO.Mat(B , U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
    G.update_gso()
    lll = LLL.Reduction( G )
    lll()

    gh = gaussian_heuristic(G.r())**0.5
    param_sieve = SieverParams()
    param_sieve['threads'] = 5
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    print("Running bdgl2...")
    g6k(alg="bdgl2")
    g6k.M.update_gso()

    print(f"dbsize: {len(g6k)}")
    time.sleep(0.2)

    nsli_succ = 0
    af_fail = []
    af_succ = []
    for gamma_fact in [0.48+0.05*i for i in range(5)]:
        for cntrtmp in range(nexp):
            print(f" - - - processing {cntrtmp+1} of {nexp} - - -", flush=True)
            c = [ randrange(-30,31) for j in range(n) ]
            e = np.array( random_on_sphere(n,(gamma_fact)*gh), dtype=DTYPE )
            b = G.B.multiply_left( c )
            b_ = np.array(b,dtype=np.int64)
            t_ = e+b_
            t = [ float(tt) for tt in t_ ]
            e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) , dtype=DTYPE )
            # egs_ = np.array( G.from_canonical(e)[n-sieve_dim:], dtype=np.float64 )
            # egs_ = np.array( G.to_canonical(egs_,start=n-sieve_dim), dtype=np.float64 )
            print(f"sqrt ee_: {(e_@e_)**0.5}")
            gh_sub = gaussian_heuristic( G.r()[-sieve_dim:] )
            print(f"sqrt rii: {(G.r()[-sieve_dim] / gh_sub)**0.5 }")
            # print(f"r: {[rr**0.5 for rr in G.r()]}")

            target_candidates = [t]
            for _ in range(0):
                e2 = np.array( random_on_sphere(n,0.1053*gh), dtype=DTYPE ) #np.array( [ randrange(0,1) for j in range(n) ],dtype=np.int64 )
                tcand_ = e2 + b #e2 + e + b_
                tcand = [ int(tt) for tt in t_ ]
                target_candidates.append( tcand )
            shuffle(target_candidates)

            #alg_2_batched( g6k,target_candidates,H11, nthreads=1, tracer_alg2=None )
            # bab_01 = np.array( alg_2_batched( g6k,target_candidates,dist_sq_bnd=1.001*e_@e_  ) )
            # bab_01 = np.array( alg_2_batched_debug( g6k,target_candidates,dist_sq_bnd=1.001*e_@e_,e=e  ) )
            bab_01 = np.array( alg_2_batched_debug( g6k,target_candidates,dist_sq_bnd=1.001*e_@e_  ) )
            print(f"e_: {e_}")
            print(f"c: {c}")
            print(f"bab01:{bab_01}")
            alg_2_batch_succ = (bab_01==c)
            print(f"alg_2_batch succsess: {alg_2_batch_succ}")
            alg_2_batch_succ = all( alg_2_batch_succ )
            if alg_2_batch_succ:
                nsli_succ+=1
                af_succ.append((e_@e_)**0.5)
            else:
                af_fail.append((e_@e_)**0.5)

            tmp = np.array( G.babai(t) )
            print(f"babai succsess: {(tmp==c)}")
    print(f"nsli_succ: {nsli_succ}")
    print(af_succ)
    print(af_fail)
