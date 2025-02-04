import sys,os
import time
from time import perf_counter
from fpylll import *
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from utils import *
from LatticeReduction import LatticeReduction

try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

import pickle
from global_consts import *

inp_path = "lwe_instances/saved_lattices/"
out_path = "lwe_instances/reduced_lattices/"
#path = "saved_lattices/"
does_exist = os.path.exists(inp_path)
if not does_exist:
    sys.exit('cannot find path for input lattices')

does_exist = os.path.exists(out_path)
if not does_exist:
    try:
        os.makedirs(out_path)
    except:
        pass #TODO: why pass?


def load_lwe(n,q,eta,k,seed=0):
    #print(f"- - - k={k} - - - load")
    with open(inp_path + f"lwe_instance_{n}_{q}_{eta}_{k}_{seed}", "rb") as fl:
        D = pickle.load(fl)
    A_, q_, eta_, k_, bse_ = D["A"], D["q"], D["eta"], D["k"], D["bse"]
    return A_, q_, eta_, k_, bse_


def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=N_SIEVE_THREADS,dump_bkz=True):
    report = {
        "params": (n,q,eta,k,seed),
        "beta_bkz": beta_bkz,
        "sieve_dim_max": sieve_dim_max,
        "sieve_dim_min": sieve_dim_max-nsieves,
        "kappa": kappa,
        "bkz_runtime": 0,
        "bdgl_runtime": [0]*(nsieves+1),
    }
    dim = n*k
    A, q, eta, k, bse = load_lwe(n,q,eta,k,seed[0]) #D["A"], D["q"], D["bse"]

    B = [ [int(0) for i in range(2*k*n)] for j in range(2*k*n) ]
    for i in range( k*n ):
        B[i][i] = int( q )
    for i in range(k*n, 2*k*n):
        B[i][i] = 1
    for i in range(k*n, 2*k*n):
        for j in range(k*n):
            B[i][j] = int( A[i-k*n,j] )

    if sieve_dim_max<60:
        nthreads = 1
    elif sieve_dim_max<80:
        nthreads = 2


    H11 = B[:len(B)-kappa] #the part of basis to be reduced
    H11 = IntegerMatrix.from_matrix( [ h11[:len(B)-kappa] for h11 in H11  ] )
    H11r, H11c = H11.nrows, H11.ncols
    assert(H11r==H11c)

    LR = LatticeReduction( H11, threads_bkz=nthreads )
    bkz_start = time.perf_counter()
    for beta in range(5,beta_bkz+1):
        then_round=time.perf_counter()
        LR.BKZ(beta)
        round_time = time.perf_counter()-then_round
        print(f"BKZ-{beta} done in {round_time}")
        sys.stdout.flush()
    report["bkz_runtime"] = time.perf_counter() - bkz_start
    H11 = LR.basis

    # if dump_bkz:
    #     with open(out_path+f"/kyb_prehybrid_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}", "wb") as f:
    #         pickle.dump({"B": H11}, f)


    #---------run sieving------------
    int_type = H11.int_type
    FPLLL.set_precision(210)
    ft = "dd" if config.have_qd else "mpfr"
    G = GSO.Mat( H11, U=IntegerMatrix.identity(H11r,int_type=int_type), UinvT=IntegerMatrix.identity(H11r,int_type=int_type), float_type=ft )
    G.update_gso()
    param_sieve = SieverParams()
    param_sieve['threads'] = nthreads
    param_sieve['otf_lift'] = False
    g6k = Siever(G,param_sieve)
    g6k.initialize_local(H11r-sieve_dim_max, H11r-sieve_dim_max+nsieves ,H11r)

    sieve_start = time.perf_counter()
    g6k(alg="bdgl2")
    i = 0
    report["bdgl_runtime"][i] = time.perf_counter()-sieve_start
    print(f"siever-{seed[0]}-{kappa}-{sieve_dim_max-nsieves+i} finished in added time {time.perf_counter()-sieve_start}\n" )
    sys.stdout.flush()
    #NOTE: this dumps
    assert g6k.r - g6k.l == sieve_dim_max-nsieves+i, f"g6k context: {g6k.r - g6k.l} != {sieve_dim_max-nsieves+i}"
    g6k.dump_on_disk(out_path+f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{g6k.n}.pkl')
    for i in range(1,nsieves+1):
        g6k.extend_left(1)
        sieve_start = time.perf_counter()
        g6k(alg="bdgl2")
        report["bdgl_runtime"][i] = time.perf_counter()-sieve_start
        print(f"siever-{seed[0]}-{kappa}-{sieve_dim_max-nsieves+i} finished in added time {time.perf_counter()-sieve_start}\n" )
        sys.stdout.flush()
        #NOTE: this dumps
        assert g6k.r - g6k.l == sieve_dim_max-nsieves+i, f"g6k context: {g6k.r - g6k.l} != {sieve_dim_max-nsieves+i}"
        g6k.dump_on_disk(out_path+f'g6kdump_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{g6k.n}.pkl')


    print(report)
    sys.stdout.flush()
    return report

if __name__=="__main__":
    # (dimension, predicted kappa, predicted beta)
    # params = [(140, 12, 48), (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    #params = [(140, 12, 48)]#, (150, 13, 57), (160, 13, 67), (170, 13, 76), (180, 14, 84)]
    params = [(140, 6, 55),(150, 6, 65),(140, 6, 74)]
    nworkers, nthreads =  5, N_SIEVE_THREADS #20, 4

    beta_bkz_offset = 1 #bkz blocksize would surpass the predicted value by this offset
    sieve_dim_max_offset = 4 #the largest slicer will work on dim=prediceted beta + this offset
    kappa_offset = 2 #data for predicted kappa up to predicted kappa + kappa_offset - 1 will be saved

    # lats_per_dim = 10
    # inst_per_lat = 10 #how many instances per A, q
    lats_per_dim = 2
    inst_per_lat = 2 #how many instances per A, q
    q, eta = 3329, 3
    #def run_preprocessing(n,q,eta,k,seed,beta_bkz,sieve_dim_max,nsieves,kappa,nthreads=1)
    output = []
    pool = Pool(processes = nworkers )
    tasks = []
    for param in params:
        for latnum in range(lats_per_dim):
            for kappa in range(param[1], param[1]+kappa_offset,1):
                tasks.append( pool.apply_async(
                    run_preprocessing, (
                        param[0], #n
                        q, #q
                        eta, #eta
                        1, #k
                        [latnum,0], #seed, second value is irrelevant
                        param[2]+beta_bkz_offset, #beta_bkz
                        param[2]+sieve_dim_max_offset, #sieve_dim_max
                        1,  #nsieves
                        kappa, #kappa
                        nthreads #nthreads
                        )
                ) )

    for t in tasks:
        output.append( t.get() )
    pool.close()

    for o_ in output:
        print(o_)
        n,q,eta,k,seed = o_["params"]
        kappa = o_["kappa"]
        beta_bkz = o_["beta_bkz"]
        sieve_dim_max = o_["sieve_dim_max"]
        sieve_dim_min = o_["sieve_dim_min"]
        filename = out_path + f"report_prehyb_{n}_{q}_{eta}_{k}_{seed[0]}_{kappa}_{sieve_dim_min}_{sieve_dim_max}.pkl"

        with open(filename, "wb") as file:
            pickle.dump( o_,file )

    sys.stdout.flush()
