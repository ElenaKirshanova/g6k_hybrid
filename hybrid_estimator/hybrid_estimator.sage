#Examples:
"""
# binomial error with parameter 2 and ternary secret of Hamming weight 50 estimation with probabilistic modes from DBDD
# sage hybrid_estimator.sage -n 150 -q 3329 -e "binomial" --param_e 3 -s "ternary" --param_s 50 --proba

n     kappa    BKZ only    bit sec.       Hybrid    bit sec.
150   25       60          45.38          45        41.84
"""
"""
# gaussian error with st.dev 2.2 and ternary secret of HW 32
# sage hybrid_estimator.sage -n 2048 -q 1267650600228229401496703205376 -e "gaussian" --param_e 3.2 -s "ternary" --param_s 32 --verbose
"""
"""
# Kyber-512 instance
# sage hybrid_estimator.sage -n 512 -q 3329 -e "binomial" --param_e 3 -s "binomial" --param_s 3
n     kappa    BKZ only    bit sec.       Hybrid    bit sec.  
512   23       405         147.66         384       142.5  
"""


import argparse, sys

from batchCVP import batchCVPP_cost
from zgsa_fast import find_beta



#loading sage files
load("framework/instance_gen.sage")
load("framework/proba_utils.sage")


def entropy(D):
    H = sum( [ -p*log(p,2) for p in D.values() ] )
    return H

supported_distributions={"ternary","binomial","gaussian"}

def bkz_cost(beta, d, alg="BDGL16_real"):
    if alg == "BDGL16_real":
        return 0.387*beta+log(8*d,2)+16.4
    elif alg == "BDGL16_asym":
        return 0.292*beta+log(8*d,2)+16.4
    else:
        print("Unrecognized SVP algorithm")
        return 0

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-q", type=int)
    parser.add_argument("-e", type=str)
    parser.add_argument("--param_e", type=float)
    parser.add_argument("-s", type=str)
    parser.add_argument("--param_s",type=float)
    parser.add_argument("--proba", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    n = args.n
    q = args.q
    probabilistic_bkz = args.proba
    verbose = args.verbose

    e_dist= args.e.strip()
    st_e = args.param_e
    match e_dist:
        case "ternary":
            std_e, D_e = build_ternary_law(n, int(st_e)) # Hamming weight st_e
        case "binomial":
            std_e, D_e = build_centered_binomial_law(int(st_e)) #eta = st_e
        case "gaussian":
            std_e, D_e = build_Gaussian_law(st_e, ceil(3*st_e))
        case _:
            sys.exit("the provided error distribution is not supported, use either of {\"ternary\",\"binomial\",\"gaussian\"}")

    s_dist = args.s.strip()
    st_s = args.param_s
    match s_dist:
        case "ternary":
            _, D_s = build_ternary_law(n, int(st_s)) # Hamming weight st_s
        case "binomial":
            _, D_s = build_centered_binomial_law(int(st_s)) #eta = st_s
        case "gaussian":
            _, D_s = build_Gaussian_law(st_s, ceil(3*st_s))
        case _:
            sys.exit("the provided secret distribution is not supported, use either of {\"ternary\",\"binomial\",\"gaussian\"}")

    return n, q, D_e, D_s, std_e, probabilistic_bkz, verbose

def estimate(n, q, D_e, D_s, std_e,probabilistic_bkz=False, kappa_max = 40):

    print("running the estimator with probabilistic_bkz = %r " %(probabilistic_bkz) )

    print('{:5s} {:8s} {:12s}{:14s} {:10s}{:10s}'.format('n', 'kappa', 'BKZ only', 'bit sec.', 'Hybrid', 'bit sec.'))

    H = entropy(D_s)
    costNonHybrid = 0

    for kappa in range(kappa_max):

        if probabilistic_bkz:
            A, b, dbdd = initialize_from_LWE_instance(DBDD_predict_diag, n-kappa, q, n, D_e, D_s, verbosity = 0)
            dbdd.integrate_q_vectors(q)

            beta, delta = dbdd.estimate_attack(probabilistic=probabilistic_bkz, silent=True)
            costBKZ = bkz_cost(beta, dbdd.dim(), alg="BDGL16_asym")
        else:
            beta = find_beta(2*n-kappa, n, q, std_e)
            costBKZ = bkz_cost(beta, 2*n-kappa, alg="BDGL16_asym")


        _, costCVP = batchCVPP_cost(beta, H*kappa, sqrt(4/3), 1)

        if kappa == 0:
            costNonHybrid = costBKZ
            betaNonHybrid = beta

        if costCVP > costBKZ:
            break

        costHybrid = costBKZ+1
        betaHybrid = beta

        if verbose:
            print('{:5s} {:8s} {:12s}{:14s} {:10s}{:10s}'.format(str(n), str(kappa), str(int(betaNonHybrid)),str(round(costNonHybrid,2)),str(int(betaHybrid)),str(round(costHybrid,2))))

    print('{:5s} {:8s} {:12s}{:14s} {:10s}{:10s}'.format(str(n), str(kappa), str(int(betaNonHybrid)),str(round(costNonHybrid,2)),str(int(betaHybrid)),str(round(costHybrid,2))))

if __name__=="__main__":
    n, q, D_e, D_s, std_e, probabilistic_bkz, verbose = parse()
    estimate( n, q, D_e, D_s, std_e, probabilistic_bkz=probabilistic_bkz)