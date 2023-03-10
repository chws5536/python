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
betas = OLS(Y, X)
print(betas)



# variance under homoskedasticity
# variance under heteroskedasticity

# hypothesis testing
# t-statistic
# p-value
