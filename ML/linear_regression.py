import numpy as np

# simple linear regression
x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,2.,3.,4.,5.])
n = x.shape[0]

weight = 2.9
bias = 0.5
learning_rate = 0.01

for i in range(100+1):
    hypothesis = (weight * x) + bias
    cost = sum((hypothesis - y) ** 2) / n

    # take the derivative of the cost function wrt w and b
    weight_gradient = 2 * sum((hypothesis - y) * x) / n
    bias_gradient = 2 * sum(hypothesis - y) / n

    # update weight and bias
    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient
    if i % 10 == 0:
        print(i, weight, bias, cost)
        
        
# multivariable linear regression
x1 = np.array([73., 93., 89., 96., 73.])
x2 = np.array([80., 88., 91., 98., 66.])
x3 = np.array([75., 93., 90., 100., 70.])
y = np.array([152., 185., 180., 196., 142.])

x = np.column_stack((x1, x2, x3))
n = x.shape[0]
k = x.shape[1]

weight = np.array([10., 10., 10.])
bias = 10.
learning_rate = 1.e-6

for i in range(1000 + 1):
    hypothesis = (x @ weight) + bias
    cost = sum((hypothesis - y) ** 2) / n

    # take the derivative of the cost function wrt w and b
    weight_gradient = 2 * (x.T @ (hypothesis - y)) / n

    bias_gradient = 2 * sum(hypothesis - y) / n

    # update weight and bias, gradient descent
    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    if i % 50 == 0:
        print(i, weight, bias, cost)

        
        
