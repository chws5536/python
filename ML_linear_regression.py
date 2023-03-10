import numpy as np

# simple linear regression

#create data
x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,2.,3.,4.,5.])
n = x.shape[0]

weight = 2.9
bias = 0.5
learning_rate = 0.01

# cost minimization
for i in range(100+1):
    hypothesis = (weight * x) + bias
    # print(hypothesis)
    cost = (np.sum((hypothesis - y) ** 2)) / n
    # print(cost)

    # take the derivative of the cost function wrt w and b
    weight_gradient = 2 * sum((hypothesis - y) * x) / n
    # print(weight_grad)
    bias_gradient = 2 * sum(hypothesis - y) / n
    # print(bias_grad)

    # update weight and bias
    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient
    if i % 10 == 0:
        print(i, weight, bias, cost)
        

# multivariable linear regression

#create data
x1 = np.array([73., 93., 89., 96., 73.])
x2 = np.array([80., 88., 91., 98., 66.])
x3 = np.array([75., 93., 90., 100., 70.])
y = np.array([152., 185., 180., 196., 142.])

x = np.column_stack((x1, x2, x3))
n = x.shape[0]
k = x.shape[1]

weights = np.array([10., 10., 10.])
bias = 10.
learning_rate = 1.e-6

weight_gradients = np.zeros(k)
for i in range(1000+1):
    hypothesis = (x @ weights) + bias
    cost = (np.sum((hypothesis - y) ** 2)) / n

    # take the derivative of the cost function wrt w and b
    weight_grad_list = np.zeros((k,))
    for j in range(k):
        weight_gradient = 2 * sum((hypothesis - y) * x[:, j]) / n
        weight_gradients[j] = weight_gradient
    # print(weight_gradient1, weight_gradient1, weight_gradient2)

    bias_gradient = 2 * sum(hypothesis - y) / n
    # print(bias_gradient)

    # update weight and bias, gradient descent
    weights -= learning_rate * weight_gradients
    bias -= learning_rate * bias_gradient
    # print(weights, bias)

    if i % 50 == 0:
        print(i, weight, bias, cost)
        
        
        
