#approx_sum_of_roots returns the sum of sqrt(x) for all x in the interval [a,b]. If b-a>=200, we approximate this sum with an integral approach instead

def true_sum_of_roots(a, b):
    if b<a:
        return 0
    s=0
    for i in range(a, b+1):
        s+=i**(1/2)
    return (s)

def integral_approx(a,b):
    return (int(2/3*(b+1)**(3/2))-int(2/3*(a)**(3/2))+int(2/3*(b)**(3/2))-int(2/3*(a-1)**(3/2)))/2

def approx_sum_of_roots(a,b):
    if b-a<200:
        return true_sum_of_roots(a, b)
    else:
        return integral_approx(a,b)