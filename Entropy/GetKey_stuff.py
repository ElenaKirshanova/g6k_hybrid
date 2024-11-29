from numpy import array
from Entropy.Multinomial import multinomial

def find_Lex_MPerm(wlist,key):
    n=sum(wlist)
    for i in range(len(wlist)):
        if wlist[i]>0:
            min_i=i
            break
    if n==1:
        return [min_i]
    i=abs(min_i)
    count_old=0
    count=multinomial(list(array(wlist)-array(i*[0]+[1]+(len(wlist)-i-1)*[0])))
    while key>count:
        count_old=count
        i+=1
        count+=multinomial(list(array(wlist)-array(i*[0]+[1]+(len(wlist)-i-1)*[0])))
    return [i]+find_Lex_MPerm(list(array(wlist)-array(i*[0]+[1]+(len(wlist)-i-1)*[0])),key-count_old)

def find_Lex_MPerm_signed(wlist,key):
    n=sum(wlist)
    signed_n=n-wlist[0]
    if signed_n==0:
        return n*[0]
    for i in range(len(wlist)):
        if wlist[i]>0:
            min_i=i
            break
    unsigned_key=(key%(multinomial(wlist)))+1
    unsigned_out=find_Lex_MPerm(wlist,unsigned_key)
    sign_int=bin((key-unsigned_key+1)//(multinomial(wlist)))[2:].zfill(signed_n)
    pointer=0
    out=[]
    for i in unsigned_out:
        if i==0:
            out+=[0]
        else:
            out+=[i*(-1)**(int(sign_int[pointer]))]
            pointer+=1
    return out
