from experiments.lwe_gen import *

import sys,os
import time
from time import perf_counter
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from sample import *

from preprocessing import run_preprocessing
#def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)

max_nsampl = 10**7
inp_path = "lwe instances/saved_lattices/"
out_path = "lwe instances/reduced_lattices/"

def kyberGen(n, q = 3329, eta = 3, k=1):
    polys = []
    for i in range(k*k):
        polys.append( uniform_vec(n,0,q) )
    A = module(polys, k, k)

    return A,q

def se_gen(k,n,eta):
    s = binomial_vec(k*n, eta)
    e = binomial_vec(k*n, eta)
    return s, e

def generateLWEInstances(n, q = 3329, eta = 3, k=1, ntar=5):
    A,q = kyberGen(n,q = q, eta = eta, k=k)
    bse = []
    for _ in range(ntar):
        s, e = se_gen(k,n,eta)
        b = (s.dot(A) + e) % q
        bse.append( (b,s,e) )

    return A,q,bse

def gen_and_dump_lwe(n, q, eta, k, ntar, seed=0):
    A,q,bse= generateLWEInstances(n, q, eta, k, ntar)

    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "wb") as fl:
        pickle.dump({"A": A, "q": q, "eta": eta, "k": k, "bse": bse}, fl)

def load_lwe(n,q,eta,k,seed=0):
    print(f"- - - k={k} - - - load")
    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_

def prepare_kyber(n,q,eta,k,betamax,kappa,seed=[0,0]): #for debug purposes
    report = {
        "kyb": ( n,q,eta,k ),
        "beta": 0,
        "kappa": 0,
        "time": 0
    }
    # prepeare the lattice
    dim = n*k
    print( f"Launching hybrid on: {n,q,eta,k}" )
    print(f"betamax,kappa: {betamax,kappa}")
    try:
        A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
        filename = f"g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+i}"
        g6k = Siever.restore_from_file(  )
    except FileNotFoundError:
        gen_and_dump_lwe(n, q, eta, k, ntar, seed[0])
        A, q, eta,k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]
    print(f"lenbse: {len(bse)} seed={seed[1]}")
    b, s, e = bse[seed[1]]

    r,c = A.shape
    print(f"Shape: {A.shape}, n, k: {n,k}")
    t = np.concatenate([b,[0]*r]) #BDD target
    x = np.concatenate([b-e,s]) #BBD solution
    sol = np.concatenate([e,-s])

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    tarnrmsq = 1.01*(sol.dot(sol))

    H11 = deepcopy( B[B.nrows-kappa] ) #the part of basis to be reduced
    LR = LatticeReduction( H11 )
    for beta in range(5,betamax+1):
        then_round=time.perf_counter()
        LR.BKZ(beta,tours=5)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
    return { 'B': B, 'H11': H11, 'q': q, 'eta': eta, 'k': k, 'bse': bse, 'betamax': betamax }


def attacker(input_dict, n_guess_coord, sieve_dim_max, nsieves, nthreads=1, tracer_exp=None):
    """
    # param input_dict: dictionary
    param n_guess_coord: guessing stage dim (kappa)
    param dist_sq_bnd: distance bound
    param sieve_dim_max: 1 + max dim siever is prepared for
    param nsieves: length of range within which we attempt to find a solution
    Assumes H11 is BKZ-betamax reduced while B is original. Assumes all Sievers are precomputed
    and stored in out_path+f"/bkzdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+i}"
    for i in range(nsieves).
    """
    # B, H11, q, eta, k, bse, betamax = input_dict['B'], input_dict['H11'], input_dict['q'], input_dict['eta'], input_dict['k'], input_dict['bse'], input_dict['betamax']
    n, kappa, q, eta, k, seed = input_dict['n'], input_dict['kappa'], input_dict['q'], input_dict['eta'], input_dict['k'], input_dict['seed']
    A, q, eta, k, bse = load_lwe(n,q,eta,k,seed=seed[0])

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    g6k = Siever.restore_from_file( out_path + f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-1}' )
    H11 = g6k.M.B

    dim = B.nrows
    fl = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

    int_type = B.int_type
    G = GSO.Mat( B, U=IntegerMatrix.identity(dim-kappa,int_type=int_type), UinvT=IntegerMatrix.identity(dim-kappa,int_type=int_type), float_type=ft )
    G.update_gso()

    print(G.get_r(0,0)**0.5)
    print(f"t_gs: {t_gs} | norm2: {(t_gs@t_gs)}")

    #TODO: make dimension incremention + BKZ functionality
    """
    param_sieve = SieverParams()
    param_sieve['threads'] = 5
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
    g6k("bdgl2")
    g6k.M.update_gso()
    """

    for sieveid in range(nsieves):
        vec_index = 0
        filename_siever = out_path+f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_max-nsieves+i}'
        g6k = Siever.restore_from_file(filename_siever)
        g6k.params["nthreads"] = nthreads
        for b, s, e in bse:
            try:
                candidate = alg_3(g6k,B,H11,b,n_guess_coord, nthreads=nthreads, tracer_alg3=None)
                answer = (s.dot(A)) % q
                print( f"answer: {answer}" )
                print( f"candidate: {candidate}" )
                #TODO: automate the check
                if all(answer==candidate):
                    print(f"vec-{vec_index}: succsess!")
                else:
                    print(print(f"vec-{vec_index}: fail!"))
                vec_index += 1
            except NotImplementedError as expt:
                print("g6k load supposedly successfull.")
                pass

def alg_3(g6k,B,H11,t,n_guess_coord, nthreads=1, tracer_alg3=None):
    raise NotImplementedError
    # - - - prepare targets - - -
    then_start = perf_counter()
    # t_gs = from_canonical_scaled( G,t,offset=sieve_dim )

    t1, t2 = t[:-kappa], t[-kappa:]
    slicer = RandomizedSlicer(g6k)
    distrib = centeredBinomial(eta)
    #TODO: make/(check if is) practical
    nsampl = ceil( 2 ** ( D.entropy * kappa ) )
    print(f"nsampl: {nsampl}")
    nsampl = min(max_nsampl, nsampl)
    target_candidates = [t1] #first target is always the original one
    vtilde2s = [np.array((n-kappa)*[0] + t2)]
    for _ in range(nsampl): #Alg 3 steps 4-7
        etilde2 = np.array( (dim-kappa)*[0] + distrib.sample( kappa ) ) #= (0 | e2)
        vtilde2 = np.array((n-kappa)*[0] + t2)-etilde2
        t1_ = np.array( t1+kappa*[0] ) - np.array(H11.multiply_left(vtilde2))
        target_candidates.append( t1_ )

    """
    We return (if we succeed) (-s,e)[dim-kappa-betamax:n-kappa] to avoid fp errors.
    """
    #TODO: dist_sq_bnd might have changed at this point (or even in attacker)
    ctilde1 = alg_2_batched( g6k,target_candidates,H11,betamax, nthreads=nthreads, tracer_alg2=None )
    v1 = np.array( G_.B.multiply_left( ctilde1 ) )
    #keep a track of v2?
    argminv = None
    minv = 10**12
    for vtilde2 in vtilde2s:
        v = v1+v2
        vv = v@v
        if vv < minv:
            argminv = v
    return v

def alg_2_batched( g6k,target_candidates,H11,n_slicer_coord, nthreads=1, tracer_alg2=None ):
    raise NotImplementedError
    dist_sq_bnd = 1.0 #TODO: implement

if __name__=="__main__":
    # (dimension, predicted kappa, predicted beta)
    # params = [(140, 12, 48), (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    params = [(140, 12, 48)]#, (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    nworkers, nthreads = 1, 1

    lats_per_dim = 10
    inst_per_lat = 10 #how many instances per A, q
    q, eta = 3329, 3
    #def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)
    output = []
    pool = Pool(processes = nworkers )
    tasks = []
    for param in params:
        for latnum in range(lats_per_dim):
            for kappa in range(param[1]-1, param[1]+2,1):
                # seed is a list [current latnum, max num of targets
                #attacker(input_dict, n_guess_coord, sieve_dim_max, nsieves, tracer_exp=None)
                input_dict = {"n": param[0], "kappa": kappa, "q":q , "eta":eta , "k":1 , "seed":[latnum,10] }
                tasks.append( pool.apply_async(
                    attacker, (
                        input_dict, #input_dict
                        kappa, #n_guess_coord
                        param[2]+4, #sieve_dim_max
                        7,  #nsieves
                        nthreads, #nthreads
                        None, #tracer_exp
                        )
                ) )

for t in tasks:
    output.append( t.get() )
pool.close()

for o_ in output:
    print(o_)
sys.stdout.flush()
