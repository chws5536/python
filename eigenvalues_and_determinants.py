import numpy as np


def is_singular2x2(input):
    det = input[0][0] * input[1][1] - input[1][0] * input[0][1]
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
    flag = 1
    sum = 0
    for i in range(n):
        output = input[0, i] * det3x3(np.delete(np.delete(input, 0, 0), i, 1))
        if flag % 2 == 0:
            output = output * -1
        flag += 1
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
  
  
