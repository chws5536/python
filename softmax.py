import numpy as np

x = np.array([[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]])

# one-hot-encoding
y = np.array([[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]])

weight = np.array([[ 0.7706481 ,  0.37335402, -0.05576323],
       [ 0.00358377, -0.5898363 ,  1.5702795 ],
       [ 0.2460895 , -0.09918973,  1.4418385 ],
       [ 0.3200988 ,  0.526784  , -0.7703731 ]])

bias = np.array([-1.3080608 , -0.13253094,  0.5513761 ])

n = x.shape[0]
k_x = x.shape[1]
k_y = y.shape[1]

epochs = 1001
lr = 1e-2

for i in range(epochs):
    hypothesis = x @ weight + bias
    # print(hypothesis)
    vector = np.sum(np.exp(hypothesis), axis=1)
    softmax = np.exp(hypothesis) / vector[:, None]
    # print(softmax)
    cost = sum(-np.sum(y * np.log(softmax), axis=1)) / n
    # print(cost)

    #calculate the derivative of the cross entropy loss function with respect to weight and bias

    # update weight and bias, gradient descent
    weight -= learning_rate * weight_gradient_list
    bias -= learning_rate * bias_mean

    if i % 100 == 1:
        print(i, cost)
        
#test
# x_test, y_test, n
hypothesis = x_test @ weight + bias
vector = np.sum(np.exp(hypothesis), axis=1)
softmax = np.exp(hypothesis) / vector[:, None]

