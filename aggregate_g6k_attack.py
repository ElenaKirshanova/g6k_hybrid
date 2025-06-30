import pickle
import numpy as np
filename = "exp_110.pkl"
with open(filename,"rb") as file:
    my_tracers = pickle.load(file)

exps = len(my_tracers)
T_succ = []
T_att = []
blocksizes, svp_dims= [], []
succ_cntr = 0
for my_tracer in my_tracers:
    print( my_tracer )
    bkz_invoked = my_tracer["bkz_invoked"]
    svp_calls = my_tracer["svp_calls"]
    succ = my_tracer["succ"]
    T_overall = my_tracer["T_overall"]
    T_BKZ = my_tracer["T_BKZ"]

    blocksizes.append( max(bkz_invoked.keys()) )
    svp_dims.append( svp_calls[-1][0] )

    if succ:
        succ_cntr+=1
        T_succ.append(T_overall)

    T_att.append( T_overall )

print(f"AVG succ time {np.mean(T_succ)}")
print(f"AVG time {np.mean(T_att)}")
print(f"AVG bkz dim {np.mean(blocksizes)}")
print(f"AVG svp  dim {np.mean(svp_dims)}")