import numpy as np

x = np.array([[1., 2.],
              [2., 3.],
              [3., 1.],
              [4., 3.],
              [5., 3.],
              [6., 2.]])
y = np.array([[0.], [0.], [0.], [1.], [1.], [1.]])

weight = np.array([[0.],
                  [0.]])
bias = np.array([0.])
# print(x.shape, weight.shape, bias.shape)

learning_rate = 1e-2
n = x.shape[0]
k = x.shape[1]

for i in range(1001):
    # sigmoid function
    hypothesis = 1 / (1 + np.exp((x @ weight) + bias))
    # print(hypothesis)

    # cost = -y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)   #(6,1)
    # mean of cost function
    cost = sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)) / n

    # calculate the derivative of the cost function wrt weight
    weight_gradient_list = np.zeros((2, 1))
    for j in range(k):
        weight_gradient = np.diagonal(np.multiply(x[:, j], (y * np.exp((x @ weight) + bias) + y - 1)) / (np.exp((x @ weight) + bias) + 1))
        weight_mean = sum(weight_gradient) / n
        weight_gradient_list[j, :] = weight_mean
        # print(weight_gradient_list)

    # calculate the derivative of the cost function wrt bias
    bias_gradient = (y * np.exp((x @ weight) + bias) + y - 1) / (1 + np.exp((x @ weight) + bias))
    bias_mean = sum(bias_gradient)/ n
    # print(bias_mean)
    #
    # update weight and bias, gradient descent
    # print(weight, bias)
    weight -= learning_rate * weight_gradient_list
    bias -= learning_rate * bias_mean
    # print(weight, bias)

    if i % 100 == 0:
        print(i, cost)

# test accuracy of weight and bias
x_test = np.array([[5., 2.]])
y_test = np.array([[1.]])
n = x_test.shape[0]

hypothesis = 1 / (1 + np.exp((x_test @ weight) + bias))
print(sum((hypothesis > 0.5) == y_test) / n)



