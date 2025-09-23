from random import choices
from math import comb, log2, ceil

class Distribution:

    def __init__(self, D):
        self.D = D
        self.population = []
        self.weights = []

        self.entropy = 0
        self.mean = 0
        self.secondMoment = 0
        self.variance = 0

        s = 0

        for key in D:
            p = D[key]

            self.population.append(key)
            self.weights.append(p)

            self.entropy += - p * log2(p)
            self.mean += key * p
            self.secondMoment += key**2 * p

            s += p

        self.variance = self.secondMoment - self.mean**2

        if not abs(s-1)<4.55e-16:  #fix for large eta
            raise ValueError(f"Probabilities don't sum to one. ")

    def sample(self, n):
        return [ choices(self.population, self.weights)[0] for _ in range(n) ]
    
def renormalize_nz(D):
    assert abs( sum(D.values())-1 ) <10**-12

    try:
        D.pop(0)
    except KeyError:
        pass
    scale = 1/sum(D.values())
    for key in D.keys():
        D[key] *= scale
    return D

def centeredBinomial(eta):
    n = 2*eta
    D = {}
    for i in range(-eta,eta+1):
        D[i] = comb(n, eta+i) / 2**n
    # print(D)
    return Distribution(D)

def ternaryDist(w):
    D = {
        -1: w,
        0 : 1-2*w,
        1 : w
    }
    return Distribution(D)
