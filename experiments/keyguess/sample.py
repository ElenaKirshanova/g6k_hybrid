from random import choices
from math import comb, log2, ceil

class Distribution:

    def __init__(self, D, name=None):

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

        if s!=1:
            raise ValueError("Probabilities don't sum to one.")

        if name is None:
            self.name = f"D({D})"
        else:
            self.name = name

    def sample(self, n):
        return [ choices(self.population, self.weights)[0] for _ in range(n) ]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

def centeredBinomial(eta):
    n = 2*eta
    D = {}
    for i in range(-eta,eta+1):
        D[i] = comb(n, eta+i) / 2**n
    D = Distribution(D, name=f"centeredBinomial({eta})")
    print(D)
    return D

if __name__=="__main__":
    kappa = 5
    dist = centeredBinomial(3)

    for _ in range( ceil( 2 ** ( dist.entropy * kappa ) ) ):
        e_2 = dist.sample(kappa)
