# Multinomial coefficient and the multiplicity of a vector, i.e. the amount of unsigned permutations

from scipy.special import comb

def multinomial(params):
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1],exact=True) * multinomial(params[:-1])

def multiplicity(dist):
    dist_m={}
    for key in dist:
        p=dist[key]
        dist_m[p]=dist_m.get(p,0)+1
    del(p)
    P=[]
    for key in dist_m:
        P+=[key]
    P.sort(reverse=True)
    m=[]
    for p in P:
        m+=[dist_m[p]]
    return P,m

def count_with_multiplicities(L,m):
    out=1
    for i in range(len(L)):
        out*=m[i]**L[i]
    return out*multinomial(L)