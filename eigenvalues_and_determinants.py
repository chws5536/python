import numpy as np


def det2x2(A):
    return A[0][0] * A[1][1] - A[1][0] * A[0][1]


def is_singular2x2(A):
    det = det2x2(A)
    ans = det == 0
    return det, ans


def det3x3(A):
    return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1] \
           - A[0][2] * A[1][1] * A[2][0] - A[0][1] * A[1][0] * A[2][2] - A[0][0] * A[1][2] * A[2][1]


def is_singular3x3(A):
    det = det3x3(A)
    ans = det == 0
    return det, ans


def det_4x4(A):
    n = A.shape[0]
    sum = 0
    for i in range(n):
        output = ((-1) ** i) * A[0, i] * det3x3(np.delete(np.delete(A, 0, 0), i, 1))
        sum += output
    return sum


def det_nxn(A):
    det = 0
    n = A.shape[0]
    sum_ = 0
    if n > 3:
        for i in range(n):
            ret = ((-1) ** i) * A[0, i] * det_nxn(np.delete(np.delete(A, 0, 0), i, 1))
            sum_ += ret
        return sum_

    elif n == 3:
        det = A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] + A[0][2] * A[1][0] * A[2][1] \
              - A[0][2] * A[1][1] * A[2][0] - A[0][1] * A[1][0] * A[2][2] - A[0][0] * A[1][2] * A[2][1]

    elif n == 2:
        det = A[0][0] * A[1][1] - A[1][0] * A[0][1]
    else:
        det = A[0][0]

    return det


def is_singular_nxn(A):
    det = det_nxn(A)
    ans = det == 0
    return det, ans


def eigenvalues2x2(A):
    # calculate coefficients for a quadratic formula
    a = 1
    b = -(A[0][0] + A[1][1])
    c = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])

    # calculate eigenvalues by solving the quadratic equation
    r1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
    r2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
    r = np.array([r1, r2])
    return r


def cramers_rule3x3(A, b):
    n = A.shape[0]
    k = A.shape[1]
    ans = np.zeros((n, k))

    # calcualte determinant
    det = 1 / det3x3(A)

    # calculate co-factor matrix
    for i in range(n):
        for j in range(k):
            ans[i, j] = ((-1) ** (i + j)) * det2x2(np.delete(np.delete(A, i, 0), j, 1))
    return (det * ans).T @ b


def is_positive_definite(A):
    n = A.shape[0]
    det = det_nxn(A)
    if det <= 0:
        return 0
    for i in range(n, 1, -1):
        A = np.delete(np.delete(A, n - 1, 0), n - 1, 1)
        n = A.shape[0]
        if n == 3:
            det = det_nxn(A)
            if det <= 0:
                return 0
        elif n == 2:
            det = det_nxn(A)
            if det <= 0:
                return 0
        elif n == 1:
            if A[0][0] <= 0:
                return 0
    return 1


def is_positive_semidefinite(A):
    n = A.shape[0]
    for i in range(n - 1, -1, -1):
        if A[i][i] < 0:
            return 0
        A_sub = np.delete(np.delete(A, i, 0), i, 1)
        det = det_nxn(A_sub)
        if det < 0:
            return 0
        for j in range(n - 2, -1, -1):
            A_sub_sub = np.delete(np.delete(A_sub, j, 0), j, 1)
            det = det_nxn(A_sub_sub)
            if det < 0:
                return 0
    det = det_nxn(A)
    if det < 0:
        return 0
    return 1


def sign(n, det):
    odd = n % 2
    if odd:
        if det >= 0:
            return 0
    else:
        if det <= 0:
            return 0


def is_negative_definite(A):
    n = A.shape[0]
    det = det_nxn(A)
    if sign(n, det) == 0:
        return 0
    for i in range(n, 1, -1):
        A = np.delete(np.delete(A, n - 1, 0), n - 1, 1)
        n = A.shape[0]
        if n == 3:
            det = det_nxn(A)
            if sign(n, det) == 0:
                return 0
        elif n == 2:
            det = det_nxn(A)
            if sign(n, det) == 0:
                return 0
        elif n == 1:
            if sign(n, A[0][0]) == 0:
                return 0
    return 1


def semi_sign(n, det):
    odd = n % 2
    if odd:
        if det > 0:
            return 0
    else:
        if det < 0:
            return 0


def is_negative_semidefinite(A):
    n = A.shape[0]
    det = det_nxn(A)
    if semi_sign(n, det) == 0:
        return 0
    for i in range(n - 1, -1, -1):
        A_sub = np.delete(np.delete(A, i, 0), i, 1)
        n = A_sub.shape[0]
        det = det_nxn(A_sub)
        if semi_sign(n, det) == 0:
            return 0
        for j in range(n - 2, -1, -1):
            A_sub_sub = np.delete(np.delete(A_sub, j, 0), j, 1)
            n = A_sub_sub.shape[0]
            det = det_nxn(A_sub_sub)
            if semi_sign(n, det) == 0:
                return 0

    n = A.shape[0]
    for k in range(n - 1, -1, -1):
        if A[k][k] > 0:
            return 0
    return 1
