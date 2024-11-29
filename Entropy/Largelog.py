# To avoid errors due to log() having inputs too large, we define a version that iteratively reduces the input until it is sufficiently small to be evaluated by log()

from math import log, isfinite

def largelog(x, base=2):
    if x==0:
        return -float('inf')
    remember_x=x
    sign=1
    if remember_x<1:
        remember_x=1/remember_x
        sign=-1
    countdiv=0
    while not(isfinite(log(remember_x,2))):
        remember_x=remember_x/(2**1000)
        countdiv+=1
    return sign*(log(remember_x,base)+countdiv*1000)

def flog(x, base=2):
    return float(truelog(x, base=base))