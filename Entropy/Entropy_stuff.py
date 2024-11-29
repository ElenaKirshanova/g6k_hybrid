from scipy.stats import entropy

def dist_entropy(dist, incomplete=True, base=2):
    # scaled entropy. This function automatically completes the input set to 1; should this lead to errors due to rounding, use incomplete=False. Is designed to scale with the scaling factor of the binomial coefficient.
    A=[]
    for key in dist:
        A+=[dist[key]]
    return entropy(A, base=base)
