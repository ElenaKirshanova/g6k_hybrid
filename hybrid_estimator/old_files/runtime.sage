var('a')
T1 = a*((a - 2*(a-1)/(1+sqrt(1-1./a)))**(-1))
T2 = 1./(a*(1+2*(1-1/a)**(3/2)-3*(1-1./a)))

print(T1(a=4/3.))
print(T2(a=4/3.))

g = Graphics()
g += plot(T2, a, [1.1, 2.9], color="blue")
g += plot(T1, a, [1.1, 2.9], color="red")
g.show()