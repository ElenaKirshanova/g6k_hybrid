from random import randrange, choices
import numpy as np
import json

# def generateLWEInstance(n):
#   A,s,e,q = kyberGen(n)
  
#   b = (s.dot(A) + e) % q

#   return A,b,q,s,e

def generateLWEInstances(n,q,dist,dist_param,ntar):
    A = []
    for _ in range(n):
       a = np.array([ randrange(q) for _ in range(n) ])
       A.append(a)
    A = np.array(A)

    bse = []

    for _ in range(ntar):
        if dist=="binomial":
            s = binomial_vec(n, dist_param)
            e = binomial_vec(n, dist_param)
        elif dist=="ternary":
           s = ternary_vec(n, dist_param)
           e = ternary_vec(n, dist_param)
        else:
           raise NotImplementedError("Distribution %s not implemented." % dist)
        
        b = (s.dot(A) + e) % q

        bse.append((b,s,e))
    
    return A,q,bse


"""
  Returns an integer following the binomial distribution with parameter eta.
"""
def binomial_dist(eta):
  s = 0
  for i in range(eta):
    s += randrange(2)
  return s

"""
  Returns an n-dimensional vector,
  whose coordinates follow a centered binomial distribution
  with parameter eta.
"""
def binomial_vec(n, eta):
  v = np.array([0]*n)
  for i in range(n):
    v[i] = binomial_dist(2*eta) - eta
  return v

"""
    For 0 <= w <= 1/2, returns an n-dimensional vector where each coordinates is
        1 with probability w,
        -1 with probability w,
        0 with probabiltiy 1-2*w.
"""
def ternary_vec(n,w):
   population = [1,-1,0]
   weights = [w,w,1-2*w]
   return np.array(choices(population,weights, k=n))

"""
  Returns an n-dimensional vector,
  whose coordinates follow the uniform distrbution
  on [a,...,b-1].
"""
def uniform_vec(n, a, b):
  return np.array([randrange(a,b) for _ in range(n)])

"""
  Returns the rotation matrix of poly, i.e.,
  the matrix consisting of the coefficient vectors of
    poly, X * poly, X^2 * poly, ..., X^(deg(poly)-1) * poly
  modulo X^n-1.
  If cyclotomic = True, then reduction mod X^n+1 is applied, instead of X^n-1.
"""
def rotMatrix(poly, cyclotomic=False):
  n = len(poly)
  A = np.array( [[0]*n for _ in range(n)] )
  
  for i in range(n):
    for j in range(n):
      c = 1
      if cyclotomic and j < i:
        c = -1

      A[i][j] = c * poly[(j-i)%n]
      
  return A

"""
  Given a list of polynomials poly = [p_1, ...,p_n],
  this function returns an rows*cols matrix,
  consisting of the rotation matrices of p_1, ..., p_n.
"""
def module( polys, rows, cols ):
  if rows*cols != len(polys):
    raise ValueError("len(polys) has to equal rows*cols.")
  
  n = len(polys[0])
  for poly in polys:
    if len(poly) != n:
      raise ValueError("polys must not contain polynomials of varying degrees.")
  
  blocks = []
  
  for i in range(rows):
    row = []
    for j in range(cols):
      row.append( rotMatrix(polys[i*cols+j], cyclotomic=True) )
    blocks.append(row)
  
  return np.block( blocks )
  
"""
  Returns A,s,e, where
  A is a random (n x m)-matrix mod q and
  the coordinates of s and e follow a centered binomial distribution with parameter eta.
"""
def binomialLWEGen(n,m,q,eta):
  A = np.array( [[0]*m for _ in range(n)] )
  
  for i in range(n):
    for j in range(m):
      A[i][j] = randrange(q)
  
  s = binomial_vec(n, eta)
  e = binomial_vec(m, eta)
  
  return A,s,e