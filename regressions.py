import numpy as np
import pandas as pd


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


def IVREG(Y, X, Z):
    return np.linalg.inv(Z.T @ X) @ Z.T @ Y
  
  
df = np.array(pd.read_csv('path_to_data.csv'))

# prepare data
Y = df[:,0]
X0 = np.ones(100).T
# the second column of the independent variables is endogenous
X1 = df[:,1:3]
# the instrument
X2 = df[:,3]

X = np.column_stack((X0,X1))
n = X.shape[0]
k = X.shape[1]
Z = np.column_stack((X[:,0:-1],X2))

iv_beta = IVREG(Y, X, Z)
print(iv_beta)

P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
invxpx = np.linalg.inv(X.T @ P @ X)

r = Y - (X @ iv_beta)
r2 = r ** 2
sum_r2 = sum(r2) / (n-k)
iv_se = np.sqrt(np.diagonal(sum_r2 * invxpx))
iv_t_val = iv_beta / iv_se
print(iv_se, iv_t_val)

identity = np.identity(r.shape[0])
D = r2 * identity
iv_rob_se = np.sqrt(np.diagonal(np.linalg.inv(Z.T @ X) @ (Z.T @ D @ Z) @ np.linalg.inv(X.T @ Z)))
iv_rob_t_val = iv_beta / iv_rob_se
print(iv_rob_se, iv_rob_t_val)



