from fpylll import *
FPLLL.set_random_seed(0x1337)
from g6k.siever import Siever
from g6k.siever_params import SieverParams
from g6k.slicer import RandomizedSlicer
from utils import *
import sys

from hyb_att_on_kyber import alg_2_batched

n, betamax, sieve_dim = 89, 44, 63 #n=170 is liikely to fail
ft = "ld" if n<145 else ( "dd" if config.have_qd else "mpfr")

#B = IntegerMatrix(n,n)
#B.randomize("qary", k=40, q = 1601)


B = IntegerMatrix.from_file("lwe_input")
G = GSO.Mat(B, float_type=ft)
G.update_gso()



#c = [ randrange(-30,31) for j in range(n) ]
e = [-17, 10, -6, 0, -11, -14, 5, 4, 9, -3, 2, 2, 15, 1, 8, -1, -3, 3, -2, 16, -3, 5, 1, -4, -2, 1, 4, -6, 5, 5, 15, -13, 1, 2, 6, 11, -13, -12, 4, -8, -9, -3, 4, -10, 4, 12, -10, 25, -7, 8, 4, 0, 6, -9, 3, -4, 3, -4, -10, 4, 3, 7, 2, -5, 0, -2, 15, 18, 11, 2, -7, 5, -11, 1, 1, 9, -4, -1, -21, 3, -6, -14, -7, -1, -13, 19, -3, 12, -8]
#np.array( random_on_sphere(n,floor(7*sqrt(n))) )
#b = G.B.multiply_left( c )
#b_ = np.array(b,dtype=np.int64)
t_ = [982, 1031, 497, 1541, 250, 1094, 1478, 430, 248, 990, 7, 85, 222, 1059, 996, 1191, 1315, 344, 1478, 1107, 522, 58, 1196, 792, 466, 611, 913, 1511, 529, 472, 328, 1544, 1287, 497, 12, 1218, 894, 737, 994, 1246, 1175, 920, 1217, 621, 484, 1273, 1188, 15, 172, 1051, 236, 1102, 156, 1348, 11, 948, 473, 731, 388, 135, 9, 1463, 1438, 286, 697, 659, 34, 716, 739, 1572, 504, 1474, 1452, 12, 1029, 1554, 1377, 284, 1586, 1308, 1184, 889, 1328, 837, 1483, 971, 847, 1572, 1162]#b_ = np.array(b,dtype=np.int64)
#t_ = e+b_
t = [ int(tt) for tt in t_ ]
#e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
#print(f"e_: {e_}")

if sieve_dim<30: print("Slicer is not implemented on dim < 30")
if sieve_dim<40: print("LSH won't work on dim < 40")

lll = LLL.Reduction(G)
lll()


bkz = LatticeReduction(B)
for beta in range(5,betamax+1):
    then_round=time.perf_counter()
    bkz.BKZ(beta,tours=5)
    round_time = time.perf_counter()-then_round
    print(f"BKZ-{beta} done in {round_time}")
    sys.stdout.flush()

int_type = bkz.gso.B.int_type
G = GSO.Mat( bkz.gso.B, U=IntegerMatrix.identity(n,int_type=int_type), UinvT=IntegerMatrix.identity(n,int_type=int_type), float_type=ft )
G.update_gso()
lll = LLL.Reduction( G )
lll()

gh = gaussian_heuristic(G.r())
param_sieve = SieverParams()
param_sieve['threads'] = 2
g6k = Siever(G,param_sieve)
g6k.initialize_local(n-sieve_dim,n-sieve_dim,n)
print("Running bdgl2...")
g6k(alg="bdgl2")
g6k.M.update_gso()

e_np = np.array(e)
print(gh, e_np@e_np)


""""
c = [ randrange(-30,31) for j in range(n) ]
e = np.array( random_on_sphere(n,0.49*gh) )
b = G.B.multiply_left( c )
b_ = np.array(b,dtype=np.int64)
t_ = e+b_
t = [ int(tt) for tt in t_ ]
e_ = np.array( from_canonical_scaled(G,e,offset=sieve_dim) )
print(f"e_: {e_}")
"""
target_candidates = [t]
#for _ in range(5):
#    e2 = np.array( [ randrange(0,1) for j in range(n) ],dtype=np.int64 )
#    tcand_ = e2 + t_
#    tcand = [ int(tt) for tt in t_ ]
#    target_candidates.append( tcand )

#alg_2_batched( g6k,target_candidates,H11, nthreads=1, tracer_alg2=None )
bab_01 = np.array( alg_2_batched( g6k,target_candidates, dist_sq_bnd=0.85  ) )
#print(f"c: {c}")
print(bab_01)
#print(f"alg_2_batch success: {(bab_01==c)}")
print(G.B.multiply_left(bab_01))
print(t)
er_found = np.array(G.B.multiply_left(bab_01))-np.array(t_)
print(er_found)
print(er_found@er_found)

