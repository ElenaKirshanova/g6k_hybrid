"""
Examples:
sage hybrid_cost kyber208 -kmax=32
sage hybrid_cost kyber1024 -kmax=60
"""

from zgsa_nonsym import ZGSA, ZGSA_old
from batchCVP import batchCVPP_cost
from parser import HelpException, parse_all
from utils import st_dev_central_binomial, H, CB2, CB3
import matplotlib.pyplot as plt


#core-SVP
def svp_cost(beta, d, alg="BDGL16_real"):
    if alg == "BDGL16_real":
        return 0.387*beta+log(8*d,2)+16.4
    elif alg == "BDGL16_asym":
        return 0.292*beta+log(8*d,2)+16.4
    else:
        print("Unrecognized SVP algorithm")
        return 0

if __name__=="__main__":
    try:
        #n, q, kappa, st_dev_e, dist = parse_all()
        n = 160
        q = 3329
        dim = 2*n

        logvol = log(q)*n

        for beta in range(50, n, 10):
            r_log = ZGSA(dim, n, q, beta)
            #r_log_old =ZGSA_old(dim, n, q, beta)
            r_dbdd = [log(bkzgsa_gso_len(logvol, i, dim, beta)) for i in range(dim)]

            print(logvol.n(), sum(r_log), sum(r_dbdd))

            plt.plot(r_log, linestyle = 'dashed', color='green', linewidth=2)
            #plt.plot(r_log_old, linestyle = 'dotted', color='red')
            plt.plot(r_dbdd, linestyle = 'dotted', color='red')
            plt.legend([str(beta)])
            plt.show()


        beta = find_beta(dim, n, q, st_dev_e)

        min_rt = infinity
        minTbkz = 0
        minTcvp = 0
        minbeta = 0
        minkappa = 0
        for kappa_ in range(kappa+1):
            M_log = kappa_*H(CB3)+1 # number of CVP-targets
            beta = find_beta(dim-kappa_, n-kappa_, q, st_dev_e)
            if beta==infinity: continue
            Tbkz = svp_cost(beta,dim-kappa_)
            _, Tcvp = batchCVPP_cost(beta, M_log, sqrt(4/3.), 1)
            min_ = max(Tbkz, Tcvp)
            if min_<min_rt:
                min_rt = min_
                minTbkz = Tbkz
                minTcvp = Tcvp
                minbeta = beta
                minkappa = kappa_
                #print(RR(min_rt), RR(minTbkz), RR(minTcvp), minbeta, minkappa)

        print()
        print(f"n={n}, q={q}")
        print(f"Est. cost: {RR(min_rt):.4f}, Cost SVP: {RR(minTbkz):.4f}, Cost CVP: {RR(minTcvp):.4f}, beta: {minbeta}, guessing coords.: {minkappa}")
    except HelpException:
        pass
