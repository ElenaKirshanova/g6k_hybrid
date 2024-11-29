load('Distribution_Classes.sage')

falcon_q=12289
falcon512_sigma=float(1.17*(falcon_q/(2*512))^(1/2))
falcon1024_sigma=float(1.17*(falcon_q/(2*1024))^(1/2))

falcon_denominator={}

def create_falcon_dist(sigma, denominator_goodness_ratio=2**(-10), nosamples=2**20, goodness_limit=2^-0.0001, give_b=False):
    # step 1: find an upper bound b_0 for the denominator of p_i (i.e. the sum of all exp(-j^2/(2*sigma^2)) ) that is in very close proximity
    
    b=0
    denominator_old=1
    denominator_new=1
    while denominator_new/denominator_old>denominator_goodness_ratio:
        denominator_old=denominator_new
        b+=1
        denominator_new=exp(-(b)^2/(2*sigma^2))
    denominator=sum([exp(-i^2/(2*sigma^2)) for i in range(-b,b+1)])+exp(-b^2/(2*sigma^2))
    if give_b:
        return b
    
    # step 2: find an upper bound b_1 such that sampling nosamples many coefficients according to our approximate FALCON distribution would return only return elements that are absolute smaller than b_1
    b_1=0
    p=1/denominator
    while p^nosamples<goodness_limit:
        b_1+=1
        p+=2*exp(-(b_1)^2/(2*sigma^2))/denominator
    
    # step 3: calculate actual denominator for truncated distribution
    true_denominator=1
    for i in range(1,b_1+1):
        true_denominator+=2*exp(-i^2/(2*sigma^2))
    
    # step 4: create distribution
    dist={}
    for i in range(-b_1,b_1+1):
        dist[i]=exp(-i^2/(2*sigma^2))/true_denominator
    return distribution(dist)


def distribution_falcon(i, params, denominator_goodness_ratio=2**(-10)):
    
    sigma=params[0]
    
    global falcon_denominator
    
    #Step 1: find denominator
    if sigma not in falcon_denominator or falcon_denominator[sigma][0]>denominator_goodness_ratio:
        b=create_falcon_dist(sigma, denominator_goodness_ratio=denominator_goodness_ratio, give_b=True)
        D=sum([exp(-i^2/(2*sigma^2)) for i in range(-b, b+1)])+2*exp(-b^2/(2*sigma^2))
        falcon_denominator[sigma]=(denominator_goodness_ratio,D,b)
    denominator=falcon_denominator[sigma][1]
    
    #Step 2: calculate probability of sampling i
    return exp(-(i^2)/(2*sigma^2))/denominator

falcon512dist=func_distribution("D(4.06)", distribution_falcon, [falcon512_sigma], ["sigma"])
falcon1024dist=func_distribution("D(2.87)", distribution_falcon, [falcon1024_sigma], ["sigma"])