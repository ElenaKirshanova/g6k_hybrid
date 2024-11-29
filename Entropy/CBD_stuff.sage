load('Distribution_Classes.sage')
from Entropy.Convolution_stuff import *



CBD_list={0:distribution({0:1}),1:distribution({-1:1/4,0:1/2,1:1/4})}

def CBD(eta):
    global CBD_list
    CBD_list[eta]=CBD_list.get(eta,distribution(self_convolution(CBD_list[1].d,eta)))
    return CBD_list[eta]

B1=CBD_list[1]
B2=CBD(2)
B3=CBD(3)
