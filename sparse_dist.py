from sample import Distribution
from scipy.special import comb
from random import uniform,shuffle
from math import log

def ltabs(a,b):
    return abs(a-b)<10**-12

class sparse_distribution:
    # n-vector`s subvectors of len k with Hamming weight 0<=w_prime<=w and nonzero entries distributed according to D. 
    def __init__(self,n,k,w,D: Distribution):
        # assert k>= w
        self.PD = {}
        self.cumPD = {}
        cumP = 0
        for w_prime in range(w+1):
            tmp = comb(n-k,w-w_prime)*comb(k,w_prime) / comb(n,w)  
            self.PD[w_prime] = tmp
            cumP += tmp
            self.cumPD[w_prime] = cumP
        assert ltabs(cumP,1), "I messed up in the formula"

        pd = [ tmp for tmp in self.PD ]
        if 0 in pd:
            pd.pop(0)
        self.entropy = entropy(n,k,w,D)
        
        self.n = n
        self.k = k
        self.w = w
        assert not 0 in D.D.keys(), "0 entry in distribution contradicts the (promise on) Hamming weight"

        self.D = D

    def subsample(self): 
        r = uniform(0,1)
        w_prime=0
        #getting the number of nonzero coordinates among the last k ones
        while r>=self.cumPD[w_prime] and w_prime < self.w:
            w_prime+=1

        v = [0]
        while not all(v): #w_prime nonzero coordinates are ensured if all v[i] are nonzero
            v = self.D.sample(w_prime) 
        assert len(v) == w_prime, f"Bad len v : {len(v), w_prime}"
        v = v + (self.k-w_prime)*[0]
        assert len(v) == self.k, f"Very bad len v : {len(v), self.k}"
        shuffle(v)
        return v
    
    def sample(self,i):
        return self.subsample()
    

def perm_probas(n,k,w,D: Distribution):
    # assert k>= w
    NUM = []
    for w_prime in range(w+1):
        tmp = comb(n-k,w-w_prime)*comb(k,w_prime) / comb(n,w)
        NUM.append( tmp )
    return NUM

def entropy(n, k, w, D):
    
    NUM = perm_probas(n,k,w,D)
    print( NUM )
    NUM = [(tmp) for tmp in NUM]
    
    ent = 0
    for w_prime in range(w+1):
        ent += NUM[w_prime] * ( log(NUM[w_prime],2) - w_prime )

    return -ent