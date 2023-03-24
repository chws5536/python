import numpy as np


def det2x2(input):
    return input[0][0] * input[1][1] - input[1][0] * input[0][1]


def is_singular2x2(input):
    det = det2x2(input)
    ans = det == 0
    return det, ans
  

def det3x3(input):
  return input[0][0] * input[1][1] * input[2][2] + input[0][1] * input[1][2] * input[2][0] + input[0][2] * input[1][0] * input[2][1] \
        - input[0][2] * input[1][1] * input[2][0] - input[0][1] * input[1][0] * input[2][2] - input[0][0] * input[1][2] * input[2][1]

  
def is_singular3x3(input):
    det = det3x3(input)  
    ans = det == 0
    return det, ans
  
  
def det_4x4(input):
    n = input.shape[0]
    sum = 0
    for i in range(n):
        output = ((-1) ** i) * input[0, i] * det3x3(np.delete(np.delete(input, 0, 0), i, 1))
        sum += output
    return sum
  
  
def eigenvalues2x2(input):
    # calculate coefficients for a quadratic formula
    a = 1
    b = -(input[0][0] + input[1][1])
    c = (input[0][0] * input[1][1]) - (input[0][1] * input[1][0])

    # calculate eigenvalues by solving the quadratic equation
    r1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
    r2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
    r = np.array([r1, r2])
    return r
  
  
def cramers_rule3x3(A, b):
    n = A.shape[0]
    k = A.shape[1]
    ans = np.zeros((n, k))

    #calcualte determinant
    det = 1 / det3x3(A)

    # calculate co-factor matrix
    for i in range(n):
        for j in range(k):
            ans[i, j] = ((-1) ** (i + j)) * det2x2(np.delete(np.delete(A, i, 0), j, 1))
    return (det * ans).T @ b

    # compare answers
    # return np.linalg.inv(A) @ b
    
    
