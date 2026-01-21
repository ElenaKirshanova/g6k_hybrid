#Largely taken from the Leaky LWE estimator https://github.com/lducas/leaky-LWE-Estimator
#as well as the Lattice Estimator


from math import pi, exp, log

def GH_sv_factor_squared(k):
    return ((pi * k)**(1. / k) * k / (2. * pi * exp(1)))

def compute_delta(k):
    """Computes delta from the block size k. Interpolation from the following
    data table:
    Source : https://bitbucket.org/malb/lwe-estimator/
    src/9302d4204b4f4f8ceec521231c4ca62027596337/estima
    tor.py?at=master&fileviewer=file-view-default
    :k: integer
    estimator.py table:
    """

    small = {0: 1e20, 1: 1e20, 2: 1.021900, 3: 1.020807, 4: 1.019713, 5: 1.018620,
             6: 1.018128, 7: 1.017636, 8: 1.017144, 9: 1.016652, 10: 1.016160,
             11: 1.015898, 12: 1.015636, 13: 1.015374, 14: 1.015112, 15: 1.014850,
             16: 1.014720, 17: 1.014590, 18: 1.014460, 19: 1.014330, 20: 1.014200,
             21: 1.014044, 22: 1.013888, 23: 1.013732, 24: 1.013576, 25: 1.013420,
             26: 1.013383, 27: 1.013347, 28: 1.013310, 29: 1.013253, 30: 1.013197,
             31: 1.013140, 32: 1.013084, 33: 1.013027, 34: 1.012970, 35: 1.012914,
             36: 1.012857, 37: 1.012801, 38: 1.012744, 39: 1.012687, 40: 1.012631,
             41: 1.012574, 42: 1.012518, 43: 1.012461, 44: 1.012404, 45: 1.012348,
             46: 1.012291, 47: 1.012235, 48: 1.012178, 49: 1.012121, 50: 1.012065}

    if k != round(k):
        x = k - floor(k)
        d1 = compute_delta(floor(k))
        d2 = compute_delta(floor(k) + 1)
        return x * d2 + (1 - x) * d1

    k = int(k)
    if k < 50:
        return small[k]
    else:
        delta = GH_sv_factor_squared(k)**(1. / (2. * k - 2.))
        return delta


def bkzgsa_gso_len(logvol, i, d, beta=None, delta=None):
    if delta is None:
        delta = compute_delta(beta)

    return delta**(d - 1 - 2 * i) * exp(logvol / d)


def plot_gso(r, *args, **kwds):
    return line([(i, r_,) for i, r_ in enumerate(r)], *args, **kwds)

#Thm. 4.1 from https://eprint.iacr.org/2025/1910.pdf
def find_beta(d, n, q, st_dev_e):
    for beta in range(2, d//2, 1):
        gso_len = log(bkzgsa_gso_len(n*log(q), d-beta, d, beta=beta))
        lhs  = 0.5*log(beta)+log(st_dev_e)
        if lhs < gso_len:
            return beta
    return infinity