# the convolution function is helpful for finding the distribution family B(eta) by utilizing convolution(B(eta),B(zeta))=B(eta+zeta).

def convolution(dist1, dist2):
    out = {}
    for key_1 in dist1:
        for key_2 in dist2:
            key = key_1+key_2
            out[key] = out.get(key, 0) + dist1[key_1] * dist2[key_2]
    return out

def self_convolution(dist,i):
    out={0:1}
    ex=bin(i)[2:]
    for j in range(len(ex)):
        out=convolution(out,out)
        if ex[j]=='1':
            out=convolution(out,dist)
    return out