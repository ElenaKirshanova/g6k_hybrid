"""
This file requires LatticeReduction.py file in the root of repo.
"""

import sys
import numpy as np
from LatticeReduction import LatticeReduction
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from math import sqrt, ceil, floor
from copy import deepcopy
from random import shuffle, randrange, uniform

from utils import *
from lwe_gen import *

def gen_lats(n, n_lats=1, k=None, q=None, n_guess_coord=0, seed=None):
    L = []
    for _ in range(n_lats):
        k = 1 if k is None else k
        q = 3329 if q is None else q
        seed = randrange(2**32) if seed is None else seed
        dim = 2*n*k

        polys = []
        for i in range(k*k):
          polys.append( uniform_vec(n,0,q) )
        A = module(polys, k, k)

        B = IntegerMatrix.identity(dim,int_type="long")
        for i in range(n):
            B[i,i]=q
        for i in range(n):
            for j in range(n):
                B[n+i,j]=int( A[i,j] )

        ft = "ld" if B.nrows<193 else "dd"
        int_type = B.int_type

        U = IntegerMatrix(dim-n_guess_coord,dim-n_guess_coord,int_type=int_type)
        B0 = B[:dim-n_guess_coord]
        B1 = B[dim-n_guess_coord:]
        try:
            G = GSO.Mat(B0, float_type=ft)
        except: #if "dd" is not available
            FPLLL.set_precision(208)
            ft = "mpfr"
            G = GSO.Mat(B0, float_type=ft)
        G.update_gso()
        C = IntegerMatrix.from_matrix([b for b in B0]+[b for b in B1], int_type=int_type)
        try:
            G = GSO.Mat(C, U=IntegerMatrix.identity(dim,int_type=int_type), UinvT=IntegerMatrix.identity(dim,int_type=int_type),float_type=ft)
        except: #if "dd" is not available
            FPLLL.set_precision(208)
            G = GSO.Mat(C, U=IntegerMatrix.identity(dim,int_type=int_type), UinvT=IntegerMatrix.identity(dim,int_type=int_type),float_type="mpfr")
        G.update_gso()
        L.append([G,A])
    return L

def alg2_nobatch(G,t,n_sli_coord,dist_sq_bnd):
    """
    Returns heuristically closest to t vector of a lattice defined by G. The vector is given as the
    coordinats w.r.t. G.B.
    param G: GSO.Mat object of dimension dim - n_guess_coord
    param t: np.array or sage vector object (if invoked in sage)
    param n_sli_coord: slicer dimension
    param dist_sq_bnd: BDD distance's radius if > 0, else no bound on distance
    """
    ft = G.float_type

    param_sieve = SieverParams()
    param_sieve['threads'] = 5
    param_sieve['default_sieve'] = "bgj1"
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(G.d-n_sli_coord,G.d-n_sli_coord,G.d)

    then = perf_counter()
    g6k()
    g6k.M.update_gso()
    print(f"Sieving done in: {perf_counter()-then}. dbsize: {len(g6k)}")

    t_gs = from_canonical_scaled( G,t,offset=n_sli_coord )
    print(f"t_gs: {t_gs} | norm: {(t_gs@t_gs)}")
    then = perf_counter()

    print(f"dim,len(t_gs): {G.d,len(t_gs)}")

    debug_directives = 768 + 105
    then = perf_counter()
    out_gs = g6k.randomized_iterative_slice([float(tt) for tt in t_gs],samples=1000, debug_directives=debug_directives)
    print(f"Slicer done in: {perf_counter()-then}.")

    #- - - recovering the ansver
    out = to_canonical_scaled( G,out_gs,offset=n_sli_coord )

    N = GSO.Mat( G.B[:G.d-n_sli_coord], float_type=ft )
    N.update_gso()
    # steps 4-5 in Alg2
    bab_1 = G.babai(t-np.array(out),start=G.d-n_sli_coord) #last n_sli_coord coordinates of s
    tmp = t - np.array( G.B[-n_sli_coord:].multiply_left(bab_1) )
    #TODO: babai can work with canonical repr. but I'm not sure if the scaled or unscaled one
    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=G.d-n_sli_coord ) ) #project onto span(B[-n_sli_coord:])
    bab_0 = N.babai(tmp)
    # end steps 4-5 in Alg2

    bab_01=np.array( bab_0+bab_1 )
    return bab_01

def alg3_nobatch(G,t,n_guess_coord,n_sli_coord,guess_coords,bkzbeta,dist_sq_bnd=0):
    """
    Returns heuristically closest to t vector of a lattice defined by G.
    param G: GSO.Mat object of dimension dim - n_guess_coord
    param t: np.array or sage vector object (if invoked in sage)
    param n_guess_coord: number of "guessed" coordinates
    param n_sli_coord: slicer dimension
    param guess_coords: the n_guess_coord values of the "guessed" coordinates
    param bkzbeta: bkz blocksize
    """
    dim = G.d
    dim_ = dim-n_guess_coord
    ft, int_type = G.float_type, G.int_type
    H11 = IntegerMatrix.from_matrix(G.B[:dim-n_guess_coord], int_type="long")
    LR = LatticeReduction( H11 )

    bkzbetapre = min(50,bkzbeta-5)
    for beta in range(5,bkzbeta+1):
        """
        Notice if G[:2*n-n_guess_coord] is BKZ-(beta-1) reduced,
        we just BKZ-beta reduce it.
        """
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}")

    then = perf_counter()
    LR.BKZ(beta)
    print(f"BKZ-{beta} done in {perf_counter()-then}")
    H11 = LR.basis
    Bbkz = IntegerMatrix.from_matrix( list(H11) + list(G.B[G.d-n_guess_coord:]), int_type=int_type )
    # print(Bbkz)
    G = GSO.Mat( Bbkz,float_type=ft,U=IntegerMatrix.identity(dim,int_type=int_type), UinvT=IntegerMatrix.identity(dim,int_type=int_type) )
    G.update_gso()
    # t1 = np.concatenate( [t[:dim-n_guess_coord] , n_guess_coord*[0]] ) #step 2
    t1 = np.array( G.from_canonical( ( list(G.to_canonical(t))[:dim_] + n_guess_coord*[0] ) ) )
    t2 = np.array( G.from_canonical( dim_*[0] + list( G.to_canonical(t) )[dim_:] ) )
    # print(r)
    # t2 = np.concatenate( [(dim-n_guess_coord)*[0] , t[dim-n_guess_coord:]] ) #step 2

    G_ = GSO.Mat( H11,float_type=ft,U=IntegerMatrix.identity(dim_,int_type=int_type), UinvT=IntegerMatrix.identity(dim_,int_type=int_type) )
    G_.update_gso()
    lll = LLL.Reduction( G_ )
    lll()
    # v2 = t2 - np.concatenate( [ dim_*[0] , guess_coords ] )
    # v2_ = list( G.to_canonical(v2) )[:dim_]
    # v2_ = np.array( list(v2_) + n_guess_coord*[0] ) #projection onto span H11 = Q^{dim_}
    # v2_ = G.from_canonical( v2_ )
    # t1_ = (t1 - v2_) #line 7  G.to_canonical(v2_)?
    # print(f"v2_: {v2_}")

    v2 = t2 - np.concatenate( [ dim_*[0] , guess_coords ], dtype=np.float64 )
    print(f"vs: {v2}")
    t1_ = t - np.array(G.B[dim_:].multiply_left(v2[dim_:])) - np.concatenate( [ dim_*[0] , guess_coords ], dtype=np.float64 )
    # t1_ = G.from_canonical( np.concatenate([G.to_canonical(t1_)[:dim_],n_guess_coord*[0]]) )
    t1_ = np.concatenate( [ t1_[:dim_], n_guess_coord*[0] ] )

    print(f"t: {t}")
    print(f"t1: {[float(tt) for tt in list(t1_)]}")


    bab = alg2_nobatch(G_,t1_,n_sli_coord,dist_sq_bnd)
    v1 = np.array( G_.B.multiply_left( bab ) )
    print(v1, v2)
    v = v1+np.array(G.B[dim_:].multiply_left(v2[dim_:]))
    diff = t-v
    # cur_sq_norm = diff@diff
    # if cur_sq_norm <= dist_sq_bnd:
    #     print(f"BKZ-{beta} succseeded")
    #     break
    # v = G.B.multiply_left( G.babai(v) )

    return v #v1+np.array(G.B[dim_:].multiply_left(v2[dim_:]))

def alg3_nobatch_find_beta(G,T,n_guess_coord,GUESS_COORDS,bkzbetapre,bkzbeta,DIST_SQ_BND=0,ANSWER=None, tracer=None):
    """
    Returns heuristically closest to t vector of a lattice defined by G.
    param G: GSO.Mat object of dimension dim - n_guess_coord
    param T: list of( np.array or sage vector object (if invoked in sage) )
    param n_guess_coord: number of "guessed" coordinates
    param n_sli_coord: slicer dimension
    param guess_coords: the n_guess_coord values of the "guessed" coordinates
    param bkzbeta: bkz blocksize
    """
    dim = G.d
    dim_ = dim-n_guess_coord
    ft, int_type = G.float_type, G.int_type
    H11 = IntegerMatrix.from_matrix(G.B[:dim-n_guess_coord], int_type="long")
    LR = LatticeReduction( H11 )

    # bkzbetapre = min(50,bkzbeta-20)
    for beta in range(5,bkzbetapre+1):
        """
        Notice if G[:2*n-n_guess_coord] is BKZ-(beta-1) reduced,
        we just BKZ-beta reduce it.
        """
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}")
    for beta in range(bkzbetapre,bkzbeta+1):
        then = perf_counter()
        LR.BKZ(beta)
        print(f"BKZ-{beta} done in {perf_counter()-then}")
        H11 = LR.basis
        Bbkz = IntegerMatrix.from_matrix( list(H11) + list(G.B[G.d-n_guess_coord:]), int_type=int_type )
        # print(Bbkz)
        G = GSO.Mat( Bbkz,float_type=ft,U=IntegerMatrix.identity(dim,int_type=int_type), UinvT=IntegerMatrix.identity(dim,int_type=int_type) )
        G.update_gso()
        G_ = GSO.Mat( H11,float_type=ft,U=IntegerMatrix.identity(dim_,int_type=int_type), UinvT=IntegerMatrix.identity(dim_,int_type=int_type) )
        G_.update_gso()

        succ_num = 0
        V = []
        index = 0
        while index < len(T):
            t = T[index]
            answer = ANSWER[index]
            guess_coords = GUESS_COORDS[index]
            dist_sq_bnd = DIST_SQ_BND[index]
            n_sli_coord = beta

            # t1 = np.concatenate( [t[:dim-n_guess_coord] , n_guess_coord*[0]] ) #step 2
            t1 = np.array( G.from_canonical( ( list(G.to_canonical(t))[:dim_] + n_guess_coord*[0] ) ) )
            t2 = np.array( G.from_canonical( dim_*[0] + list( G.to_canonical(t) )[dim_:] ) )
            # print(r)
            # t2 = np.concatenate( [(dim-n_guess_coord)*[0] , t[dim-n_guess_coord:]] ) #step 2



            # v2 = t2 - np.concatenate( [ dim_*[0] , guess_coords ] )
            # v2_ = list( G.to_canonical(v2) )[:dim_]
            # v2_ = np.array( list(v2_) + n_guess_coord*[0] ) #projection onto span H11 = Q^{dim_}
            # v2_ = G.from_canonical( v2_ )
            # t1_ = (t1 - v2_) #line 7  G.to_canonical(v2_)?
            # print(f"v2_: {v2_}")

            v2 = t2 - np.concatenate( [ dim_*[0] , guess_coords ], dtype=np.float64 )
            print(f"vs: {v2}")
            t1_ = t - np.array(G.B[dim_:].multiply_left(v2[dim_:])) - np.concatenate( [ dim_*[0] , guess_coords ], dtype=np.float64 )
            # t1_ = G.from_canonical( np.concatenate([G.to_canonical(t1_)[:dim_],n_guess_coord*[0]]) )
            t1_ = np.concatenate( [ t1_[:dim_], n_guess_coord*[0] ] )

            print(f"t: {t}")
            print(f"t1: {[float(tt) for tt in list(t1_)]}")


            bab = alg2_nobatch(G_,t1_,n_sli_coord,dist_sq_bnd)
            v1 = np.array( G_.B.multiply_left( bab ) )
            # print(v1, v2)
            v = v1+np.array(G.B[dim_:].multiply_left(v2[dim_:]))
            diff = t-v
            cur_sq_norm = diff@diff
            dbg = (v==answer)
            print(tracer)
            succ=all(dbg)
            if succ:
                if not (tracer is None):
                    tracer.append( {(G.d//2): [n_guess_coord,beta,succ]} )
                    succ=True
                print(f"BKZ-{beta} succseeded")
                V.append(v)
                succ_num+=1
                T.pop(index)
                ANSWER.pop( index )
                GUESS_COORDS.pop( index )
                DIST_SQ_BND.pop( index )
            else:
                if beta>=bkzbeta:
                    if not (tracer is None):
                        tracer.append( {(G.d//2): [n_guess_coord,beta,succ]} )
                        succ=True
                index+=1
            if len(T)<=0:
                break

        # v = G.B.multiply_left( G.babai(v) )
        if len(T)==0:
            print(f"All {succ_num} targets found")
            break
        else:
            print(f"{succ_num} targets found {len(T)} left ")
    return v #v1+np.array(G.B[dim_:].multiply_left(v2[dim_:]))

if __name__=="__main__":
    n, k, bkzbetapre, bkzbeta, n_guess_coord = 150, 1, 40, 67, 15 #(120, 1, 35, 15, 35) and (130, 1, 51, 15, 60) should work
    eta, q = 3, 3329
    dim = 2*n*k
    int_type="long"
    n_lats, n_exp = 2, 20
    Lats = gen_lats(n, n_lats=n_lats, k=k, q=q, n_guess_coord=n_guess_coord, seed=None)
    tracer = []
    for L in Lats:
        Gcom,A = L


        # c = [ randrange(-3,4) for j in range(dim) ]
        # e = np.array( [ uniform(-2,3) for j in range(dim) ],dtype=np.int64 )
        # b = np.array( G.B.multiply_left( c ) )
        # b_ = np.array( np.array(b,dtype=np.int64) )
        # t_ = e+b_
        # t = [ int(tt) for tt in t_ ]
        # print(f"lent {len(t)}")
        # x, err = b, e
        T, GC, X, DIST_SQ_BND = [], [], [], []
        for _ in range(n_exp):
            G = gsomat_copy(Gcom)
            B = G.B
            s = binomial_vec(dim//2, eta)
            e = binomial_vec(dim//2, eta)
            b = (s.dot(A) + e) % q
            nrows,ncols = A.shape
            t = np.concatenate([b,[0]*nrows]) #BDD target
            t = [ int(tt) for tt in t ]
            # x = np.concatenate([b-e,s]) #BBD solution
            err = np.concatenate([e,-s])
            x = t - err
            # print( B.multiply_left(G.babai(x))==x )
            T.append(t)
            X.append(x)
            guess_coords = err[dim-n_guess_coord:]
            dist_sq_bnd = 1.01*(e@e)
            GC.append(guess_coords)
            DIST_SQ_BND.append(dist_sq_bnd)



        print(f"lt: {len(t)}")
        ans =  np.array( alg3_nobatch_find_beta(G,T,n_guess_coord,GC,bkzbetapre,bkzbeta,DIST_SQ_BND,ANSWER=X,tracer=tracer) )
        # print(f"x: {[int(xx) for xx in x]}")
        # print(f"ans: {[int(aa) for aa in ans]}")
        # print(f"diff:")
        # ans = np.array(ans)
        # print(x-ans)
        # # print([ round(tmp) for tmp in x-ans ])
        # print(f"Succ: {ans==x}")
        print(tracer)
    print(f"Done..")
    print(tracer)

    K, B, S = [], [], []
    for t in tracer:
        kappa, beta, succ = t[n]
        K.append( kappa )
        B.append( beta )
        S.append( succ )
    print("\n - - - beta - - -")
    print(f"avg: {np.average( B )}, med:{np.median( B )}, std:{np.std( B )}")
    # - - - Testing alg3_nobatch - - -
    # G, A = gen_lats(n, n_lats=1, k=k, q=q, n_guess_coord=n_guess_coord, seed=None)[0]
    # B = G.B
    #
    #
    # # c = [ randrange(-3,4) for j in range(dim) ]
    # # e = np.array( [ uniform(-2,3) for j in range(dim) ],dtype=np.int64 )
    # # b = np.array( G.B.multiply_left( c ) )
    # # b_ = np.array( np.array(b,dtype=np.int64) )
    # # t_ = e+b_
    # # t = [ int(tt) for tt in t_ ]
    # # print(f"lent {len(t)}")
    # # x, err = b, e
    #
    # s = binomial_vec(dim//2, eta)
    # e = binomial_vec(dim//2, eta)
    # b = (s.dot(A) + e) % q
    # nrows,ncols = A.shape
    # t = np.concatenate([b,[0]*nrows]) #BDD target
    # t = [ int(tt) for tt in t ]
    # # x = np.concatenate([b-e,s]) #BBD solution
    # err = np.concatenate([e,-s])
    # x = t - err
    # # print( B.multiply_left(G.babai(x))==x )
    #
    # lll = LLL.Reduction(G)
    # lll(kappa_end=dim-n_guess_coord)
    #
    # guess_coords = err[dim-n_guess_coord:]
    # dist_sq_bnd = 1.01*(e@e)
    # print(f"lt: {len(t)}")
    # ans =  np.array( alg3_nobatch(G,t,n_guess_coord,n_sli_coord,guess_coords,bkzbeta,dist_sq_bnd) )
    # print(f"x: {[int(xx) for xx in x]}")
    # print(f"ans: {[int(aa) for aa in ans]}")
    # print(f"diff:")
    # ans = np.array(ans)
    # print(x-ans)
    # # print([ round(tmp) for tmp in x-ans ])
    # print(f"Succ: {ans==x}")


    # print( B.multiply_left(G.babai(ans))==ans )
    # print(G.babai(ans))
    # print(G.babai(x))

    # - - - testing alg2 - - -
        # B.randomize("qary", k=n//2, bits=11.705)
        # B = B
        # ft = "ld" if n<193 else "dd"
        # G = GSO.Mat(B, float_type=ft,U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type))
        # G.update_gso()
    #
    # c = [ randrange(-3,4) for j in range(n) ]
    # e = np.array( [ uniform(-2,3) for j in range(n) ],dtype=np.int64 )
    # b = G.B.multiply_left( c )
    # b_ = np.array(b,dtype=np.int64)
    # t_ = e+b_
    # t = [ int(tt) for tt in t_ ]
    #
    # print(f"ans: {b}")
    # print(f"ans_coords: {c}")
    # print(f"target: {t}")
    #
    # dist_sq_bnd = 1.01*(e@e)
    # bab = alg2_nobatch(G,t,n_sli_coord,dist_sq_bnd)
    # ans = G.B.multiply_left( bab )
    # print(f"Out: {list(int(ii) for ii in ans)}")
    # b = np.array(b)
    # print(f"Succsesses: {b==ans}")
    # print(f"Overall succsess: {all(b==ans)}")
    # - - - end testing alg2 - - -
