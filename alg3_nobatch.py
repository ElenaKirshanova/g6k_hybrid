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

        ft = "ld" if n<193 else "dd"
        int_type = B.int_type

        U = IntegerMatrix(dim-n_guess_coord,dim-n_guess_coord,int_type=int_type)
        B0 = B[:dim-n_guess_coord]
        B1 = B[dim-n_guess_coord:]
        try:
            G = GSO.Mat(B0, float_type=ft)
        except: #if "dd" is not available
            FPLLL.set_precision(208)
            G = GSO.Mat(B0, float_type="mpfr")
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
    print("dbg_")
    tmp = t - np.array( G.B[-n_sli_coord:].multiply_left(bab_1) )
    #TODO: babai can work with canonical repr. but I'm not sure if the scaled or unscaled one
    tmp = N.to_canonical( G.from_canonical( tmp, start=0, dimension=G.d-n_sli_coord ) ) #project onto span(B[-n_sli_coord:])
    bab_0 = N.babai(tmp)
    # end steps 4-5 in Alg2

    bab_01=np.array( bab_0+bab_1 )
    return bab_01

def alg3_nobatch(G,t,n_guess_coord,n_sli_coord,guess_coords,bkzbeta,dist_sq_bnd=0):
    """
    Returns heuristically closest to t vector of a lattice defined by G. The vector is given as the
    coordinats w.r.t. G.B.
    param G: GSO.Mat object of dimension dim - n_guess_coord
    param t: np.array or sage vector object (if invoked in sage)
    param n_guess_coord: number of "guessed" coordinates
    param n_sli_coord: slicer dimension
    param guess_coords: the n_guess_coord values of the "guessed" coordinates
    param bkzbeta: bkz blocksize
    """
    dim = G.d
    ft, int_type = G.float_type, G.int_type
    H11 = IntegerMatrix.from_matrix(G.B[:dim-n_guess_coord], int_type="long")
    LR = LatticeReduction( H11 )

    for beta in range(5,bkzbeta+1):
        """
        Notice if G[:2*n-n_guess_coord] is BKZ-(beta-1) reduced,
        we just BKZ-beta reduce it.
        """
        then = perf_counter()
        LR.BKZ(beta=beta) #,start=0, end=dim-n_sli_coord-n_guess_coord
        print(f"BKZ-{beta} done in {perf_counter()-then}")

    H11 = LR.basis
    Bbkz = IntegerMatrix.from_matrix( list(H11) + list(G.B[G.d-n_guess_coord:]), int_type=int_type )
    G = GSO.Mat( Bbkz,float_type=ft,U=IntegerMatrix.identity(dim,int_type=int_type), UinvT=IntegerMatrix.identity(dim,int_type=int_type) )
    G.update_gso()
    t1 = np.concatenate( [t[:dim-n_guess_coord] , n_guess_coord*[0]] ) #step 2
    t2 = np.concatenate( [(dim-n_guess_coord)*[0] , t[dim-n_guess_coord:]] ) #step 2

    dim_ = dim-n_guess_coord
    print(f"dim_: {dim_}")
    v2_ = t2 - np.concatenate( [ dim_*[0] , guess_coords ] )
    v2_ = np.concatenate( [ np.array( G.from_canonical( v2_, start=0, dimension=dim-n_guess_coord ) ), n_guess_coord*[0] ] )
    t1_ = (t1 - G.to_canonical(v2_)) #line 7

    G_ = GSO.Mat( H11,float_type=ft,U=IntegerMatrix.identity(dim_,int_type=int_type), UinvT=IntegerMatrix.identity(dim_,int_type=int_type) )
    G_.update_gso()
    bab = alg2_nobatch(G_,t,n_sli_coord,dist_sq_bnd)
    print("dbg_")
    print(H11.nrows,len(bab))
    v1_ = np.array( H11.multiply_left( bab ) )
    v = v1_+v2_

    return v
    # raise NotImplementedError("No Alg3 atm")

if __name__=="__main__":
    n, k, bkzbeta, n_guess_coord, n_sli_coord = 32, 1, 20, 10, 25
    dim = 2*n*k
    int_type="long"
    G, A = gen_lats(n, n_lats=1, k=k, q=3329, n_guess_coord=n_guess_coord, seed=None)[0]
    B = G.B


    c = [ randrange(-3,4) for j in range(n) ]
    e = np.array( [ uniform(-2,3) for j in range(dim) ],dtype=np.int64 )
    b = np.array( G.B.multiply_left( c ) )
    b_ = np.array( np.array(b,dtype=np.int64) )
    t_ = e+b_
    t = [ int(tt) for tt in t_ ]
    print(f"lent {len(t)}")

    lll = LLL.Reduction(G)
    lll(kappa_end=dim-n_guess_coord)

    guess_coords = e[dim-n_guess_coord:]
    dist_sq_bnd = 1.01*(e@e)
    ans =  alg3_nobatch(G,t,n_guess_coord,n_sli_coord,guess_coords,bkzbeta,dist_sq_bnd)
    print(f"ans: {ans}")
    print(f"diff:")
    print([ round(tmp) for tmp in np.array(b)-np.array(ans) ])

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
