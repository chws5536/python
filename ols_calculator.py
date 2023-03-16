import numpy as np


def OLS(y, x):
  return np.linalg.inv(x.T @ x) @ (x.T @ y)


# prepare data
X0 = np.ones(1000).T
mu, sigma = 0, 4
X1 = np.random.normal(mu, sigma, (1000, 4))
X = np.column_stack((X0, X1))
n = X.shape[0]
k = X.shape[1]
Y = np.random.normal(mu, sigma, (1000, 1))

# run regression
beta = OLS(Y, X)
print(beta)

# variance under homoskedasticity
invxx = np.linalg.inv(X.T @ X)
r = Y - (X @ beta)
r2 = r ** 2
s2 = sum(r2) / (n-k)
se = np.sqrt(np.diagonal(s2 * invxx))
t_val = np.diagonal(beta / se)
print(se, t_val)

# variance under heteroskedasticity
identity = np.identity(r.shape[0])
D = r2 * identity
robust_se = np.sqrt(np.diagonal(invxx @ (X.T @ D @ X) @ invxx))
robust_t_val = np.diagonal(beta/robust_se)
print(robust_se, robust_t_val)

