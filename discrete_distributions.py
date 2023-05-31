from math import factorial


def combination(n, r):
    return factorial(n) / (factorial(n-r)*factorial(r))
  
  
def bernoulli_dist_pmf(p, x):
    return (p ** x) * ((1-p) ** (1-x))


def bernoulli_mean(p):
    return p


def bernoulli_var(p, q):
    return p * q
  

def binomial_dist_pmf(n, r, p):
    return combination(n, r) * (p ** r) * ((1-p) ** (n-r))


def binomial_dist_cdf(n, r, p):
    ret = 0
    for i in range(0, r+1):
        ret += binomial_dist_pmf(n, i, p)
    return ret


def binomial_mean(n, p):
    return n * p


def binomial_var(n, p):
    return n * p * (1-p)
  

def multinomial_dist_pmf(n, x, y, z, p_x, p_y, p_z):
    return factorial(n)/(factorial(x)*factorial(y)*factorial(z)) * (p_x ** x) * (p_y ** y) * (p_z ** z)


def geometric_dist_pmf(p, x):
    return p * ((1-p) ** (x-1))


def geometric_dist_cdf(p, x):
    return 1 - (1 - p) ** x  
  
  
def negative_binomial_dist_pmf(x, r, p):
    return combination(x-1, r-1) * (p ** r) * ((1-p) ** (x-r))


def negative_binomial_dist_cdf(x, r, p):
    ret = 0
    for i in range(r, x + 1):
        ret += negative_binomial_dist_pmf(i, 3, p)
    return ret
  
  
def hypergeometric_dist_pmf(N, A, n, x):
    return combination(A, x) * combination(N - A,  n - x) / combination(N, n)


def hypergeometric_dist_cdf(N, A, n, x):
    ret = 0
    for i in range(0, x + 1):
        ret += combination(A, i) * combination(N - A,  n - i) / combination(N, n)
    return ret
  
  
  
  
  
  
  
  
  
  
  
  
