import numpy as np


def OLS(y, x):
  x_t = np.transpose(x)
  return np.linalg.inv(x_t @ x) @ (x_t @ y)


# prepare data
X0 = np.transpose(np.ones(1000))
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
invxx = np.linalg.inv(np.transpose(X) @ X)
residual = Y - (X @ beta)
r2 = np.transpose(residual) @ residual
s2 = r2 / (n-k)
standard_error = np.sqrt(np.diagonal(s2 * invxx))
print(standard_error)

# variance under heteroskedasticity

# hypothesis testing
# t-statistic
t_value = np.diagonal(beta / standard_error)
print(t_value)
