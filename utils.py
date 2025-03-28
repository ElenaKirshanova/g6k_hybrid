import sys, os
import glob #for automated search in subfolders
import numpy as np
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from math import sqrt, ceil, floor, log, exp
from copy import deepcopy
from random import shuffle, randrange
from sample import centeredBinomial

from discretegauss import sample_dgauss

import pickle
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

from global_consts import DTYPE

save_folder = "./saved_lattices/"
inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"

def get_filename(which_file,params):
    """
    f"lwe_instance_ternary_{n}_{q}_{dist_param:.04f}_{seed}"
    f"kyb_preprimal_{n}_{q}_ternary_{dist_param:.04f}_{seed[0]}_{betapre}.pkl"
    f"report_pre_{n}_{q}_ternary_{dist_param:.04f}_{seed[0]}_{betapre}.pkl"
    f"exp{nks}_{q}_ternary_{dist_param:.04f}.pkl"
    f'g6kdump_{n}_{q}_ternary_{dist_param:.04f}_{seed[0]}_{kappa}_{g6k.n}.pkl'
    f"report_prehyb_{n}_{q}_ternary_{dist_param:.04f}_{k}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}.pkl"
    f"tha_{n}_{q}_ternary_{dist_param:.04f}_{n_guess_coord}_{n_slicer_coord}.pkl"

    f"lwe_instance_binomial_{n}_{q}_{dist_param}_{seed}"
    f"kyb_preprimal_{n}_{q}_binomial_{dist_param}_{seed[0]}_{betapre}.pkl"
    f"report_pre_{n}_{q}_binomial_{dist_param}_{seed[0]}_{betapre}.pkl"
    f"exp{nks}_{q}_binomial_{dist_param}.pkl"
    f'g6kdump_{n}_{q}_binomial_{dist_param}_{seed[0]}_{kappa}_{g6k.n}.pkl'
    f"report_prehyb_{n}_{q}_binomial_{dist_param}_{k}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}.pkl"
    f"tha_{n}_{q}_binomial_{dist_param}_{n_guess_coord}_{n_slicer_coord}.pkl"
    """
    dp = params["dist_param"]
    if params["dist"] == "ternary":
        dpstr = f"{dp:.04f}"
    elif params["dist"] == "binomial":
        dpstr = f"{dp}"
    else: raise ValueError("dist should be either \"ternary\" or \"binomial\" ")
    # params.update( {"dpstr": dpstr} )

    if "lwe_instance" == which_file:
        # raise NotImplementedError
        n, q, seed, dist = params["n"], params["q"], params["seed"], params["dist"]
        return f"lwe_instance_{dist}_{n}_{q}_{dpstr}_{seed[0]}.pkl"
    
    elif "kyb_preprimal" == which_file:
        # raise NotImplementedError
        n, q, seed, betapre, dist = params["n"], params["q"], params["seed"], params["betapre"], params["dist"]
        return f"kyb_preprimal_{n}_{q}_{dist}_{dpstr}_{seed[0]}_{betapre}.pkl"
    
    elif "report_pre" == which_file:
        # raise NotImplementedError
        n, q, seed, betapre, dist = params["n"], params["q"], params["seed"], params["betapre"], params["dist"]
        return f"kyb_preprimal_{n}_{q}_{dist}_{dpstr}_{seed[0]}_{betapre}.pkl"
    
    elif "exp" == which_file:
        raise NotImplementedError
    
    elif "g6kdump" == which_file:
        # raise NotImplementedError
        n, q, seed, dist, kappa, n_sli_coord, bkz_beta = params["n"], params["q"], params["seed"], params["dist"], params["kappa"], params["n_sli_coord"], params["bkz_beta"]
        return f'g6kdump_{n}_{q}_{dist}_{dpstr}_{seed[0]}_{kappa}_{n_sli_coord}_{bkz_beta}.pkl'
    
    elif "report_prehyb" == which_file:
        # raise NotImplementedError dist["
        n, q, dist, seed, kappa, sieve_dim_min, sieve_dim_max = dist["n"], dist["q"], dist["dist"], dist["seed"], dist["kappa"], dist["sieve_dim_min"], dist["sieve_dim_max"]
        return f"report_prehyb_{n}_{q}_{dist}_{dpstr}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}.pkl"
    
    elif "tha" == which_file:
        raise NotImplementedError

    return 0

def gsomat_copy(M):
    n,m,int_type,float_type = M.B.nrows,M.B.ncols,M.int_type,M.float_type

    B = []
    for i in range(n):
        B.append([])
        for j in range(m):
            B[-1].append(int(M.B[i][j]))
    B = IntegerMatrix.from_matrix( B,int_type=int_type )

    U = []
    for i in range(n):
        U.append([])
        for j in range(m):
            U[-1].append(int(M.U[i][j]))
    U = IntegerMatrix.from_matrix( U,int_type=int_type )

    UinvT = []
    for i in range(n):
        UinvT.append([])
        for j in range(m):
            UinvT[-1].append(int(M.UinvT[i][j]))
    UinvT = IntegerMatrix.from_matrix( UinvT,int_type=int_type )

    M = GSO.Mat( B, float_type=float_type, U=U, UinvT=UinvT )
    M.update_gso()
    return M

def to_canonical_scaled(M, t, offset=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    assert not( scale_fact is None ), "scale_fact is None "
    if len(t)==0:
        return np.array([])
    if offset is None:
        offset=M.d

    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r())
    r_ = np.array( [sqrt(scale_fact/tt) for tt in M.r()[-offset:]], dtype=DTYPE )
    tmp = t*r_
    return np.array( M.to_canonical(tmp, start=M.d-offset) )

def from_canonical_scaled(M, t, offset=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of last coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    assert not( scale_fact is None ), "scale_fact is None "
    if len(t)==0:
        return np.array([])
    if offset is None:
        offset=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r())
    t_ = np.array( M.from_canonical(t)[-offset:], dtype=DTYPE )
    r_ = np.array( [sqrt(tt/scale_fact) for tt in M.r()[-offset:]], dtype=DTYPE )

    return t_*r_

def to_canonical_scaled_start(M, t, dim=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of first coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if len(t)==0:
        return np.array([])
    if dim is None:
        dim=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r())
    r_ = np.array( [sqrt(scale_fact/tt) for tt in M.r()[:dim]], dtype=DTYPE )
    tmp = np.concatenate( [t*r_, (M.d-dim)*[0]] )

    return np.array( M.to_canonical(tmp,start=0) )

def from_canonical_scaled_start(M, t, dim=None, scale_fact=None):
    """
    param M: updated GSO.Mat object
    param t: target vector
    param offset: number of first coordinates the coordinates are computed for
                  or None if the dimension is maximal
    """
    if len(t)==0:
        return np.array([])
    if dim is None:
        dim=M.d
    if scale_fact is None:
        scale_fact = gaussian_heuristic(M.r())
    t_ = np.array( M.from_canonical(t)[:dim], dtype=DTYPE )
    r_ = np.array( [sqrt(tt/scale_fact) for tt in M.r()[:dim]], dtype=DTYPE )

    return t_*r_

def gen_and_pickle_lattice(n, k=None, bits=None, betamax=None, seed=None):
    isExist = os.path.exists(save_folder)
    if not isExist:
        try:
            os.makedirs(save_folder)
        except:
            pass    #still in docker if isExists==False, for some reason folder can exist and this will throw an exception.

def reduce_to_fund_par_proj(B_gs,t_gs,dim):
    t_gs_save = deepcopy( t_gs )
    c = [0 for i in range(dim)]
    # for i in range(dim):
    for j in range(dim-1,-1,-1):
        mu = round( t_gs[j] / B_gs[j][j] )
        t_gs -= B_gs[j] * mu
        c[j] -= mu
    for i in range(dim):
        t_gs_save += c[i] * B_gs[i]
    return t_gs_save


def load_lattices(n):
    # An iterator through n-dimensional lattices
    # Each instance requires an exponential amount of memory, so we
    # don't store it all simoultaniously.
    # lats = []
    for filename in glob.glob(f'{save_folder}siever_{n}*.pkl'):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f: # open in readonly mode
            g6k_obj = Siever.restore_from_file(filename)
            yield g6k_obj
            # lats.append(L)
        print(filename)

# https://math.stackexchange.com/questions/4705204/uniformly-sampling-from-a-high-dimensional-unit-sphere
def random_on_sphere(d,r):
    """
    d - dimension of vector
    r - radius of the sphere
    """
    u = np.random.normal(0,1,d)  # an array of d normally distributed random variables
    d=np.sum(u**2) **(0.5)
    return r*u/d

# Borrowed from https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def uniform_in_ball(num_points, dimension, radius=1):
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T

def test_vect_proj( G, n_slicer_coord, n_tests, dist ):
    # Gives norms of n_tests projected and scaled vectors ~Bin(eta). The projection is onto
    # the last n_slicer_coord dimensional projective lattice.
    # dist = centeredBinomial(eta)

    gh_sub = gaussian_heuristic( G.r()[-n_slicer_coord:] )
    lens = []
    for cntr in range(n_tests):
        v = dist.sample(G.d)
        v_ = from_canonical_scaled( G, v, offset=n_slicer_coord,scale_fact=gh_sub )
        lv_ = (v_@v_)**0.5
        lens.append(lv_)
    return(lens)

def dist_babai(G, t):
    #Given GSO object G, returns distance between t and G.babai(t).
    cv = G.babai( t )
    v = np.array( G.B.multiply_left( cv ) )
    dist = (t-v)
    dist = (dist@dist)**0.5
    return dist

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
