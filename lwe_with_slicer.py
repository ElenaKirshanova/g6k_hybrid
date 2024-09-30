
import sys
import math
import copy

from fpylll import BKZ as fplll_bkz, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

import g6k.siever
from g6k.algorithms.bkz import pump_n_jump_bkz_tour
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from g6k.utils.util import load_lwe_challenge

from g6k.utils.lwe_estimation import gsa_params, primal_lattice_basis
from utils import *
from hybrid_estimator.batchCVP import batchCVPP_cost




if __name__=='__main__':
    fpylll_crossover = 55
    goal_margin = 1.5
    bkz_tours = 2

    alpha = 0.005
    n = 40
    A, c, q = load_lwe_challenge(n=n, alpha=alpha)

    ft = "ld" if 2*n<145 else ( "dd" if config.have_qd else "mpfr")


    try:
        min_cost_param = gsa_params(n=A.ncols, alpha=alpha, q=q,
                                    samples=A.nrows, decouple=True)
        print(min_cost_param)
        (b, s, m) = min_cost_param #bkz_dim, svp_dim, num_samples
    except TypeError:
        raise TypeError("No winning parameters.") #for alpha=0.045 it fails to find the params

    target_norm = goal_margin * (alpha*q)**2 * m
    print('alpha*q', alpha*q, 'q:', q)

    if m is None:
        m = A.nrows
    elif m > A.nrows:
        raise ValueError("Only m=%d samples available." % A.nrows)
    n = A.ncols

    B = IntegerMatrix(m+n, m)
    for i in range(m):
        for j in range(n):
            B[j, i] = A[i, j]
        B[i+n, i] = q

    #Bcopy = copy.deepcopy(B)

    print('c:', c[:m])

    dim = B.nrows
    U = IntegerMatrix.identity(dim)
    UinvT = IntegerMatrix.identity(dim)

    M_transf = GSO.Mat(B, float_type="d", U = U, UinvT = UinvT)

    L = LLL.Reduction(M_transf, delta=0.99, eta=0.501)
    L()
    assert(B[m:] == IntegerMatrix(n, m))

    B = B[:m]

    fh = open("lwe_input", "w")
    _ = fh.write(str(B))
    fh.close()

    """"
    #CHECKS
    #secret = [451, 410, 1055, 253, 1038, 456, 1300, 406, 1586, 134, 1260, 5, 454, 1247, 1069, 800, 444, 199, 1093, 135, 1175, 676, 1374, 69, 596, 871, 770, 261, 658, 1575, 657, 1370, 601, 1492, 913, 516, 138, 623, 35, 1550] # for n=50, alpha=0.005
    latv = A.transpose().multiply_left(secret)[:m]
    latv = [el for el in latv]
    print(latv)
    er = [0]*m
    for i in range(m):
        er[i] = (c[:m][i] - latv[i])%q
        if er[i]>floor(q/2): er[i] = -(q-er[i])
    print(er)
    er_ = np.array(er)
    print(er_@er_, target_norm)


    kv = np.array(latv)+er_-c[:m]
    for i,kv_ in enumerate(kv):
        assert(kv_%q == 0)
        kv[i] = kv[i]//q

    c_ = np.array(latv)+er_
    c_ = [el%q for el in c_]
    c__ = np.array(latv)+er_-np.array(kv)*q
    skv = np.concatenate((np.array(secret), -kv) )
    coeff = UinvT.transpose().multiply_left(skv) #B.multiply_left(coeff)+er_ = c
    """





















