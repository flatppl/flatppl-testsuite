"""The derivative of det(A) is the cofactor matrix, including at singular A."""

import numpy as np


def grad_oracle(point):
    matrix = np.asarray(point["A"])
    return {"A": [
        [float((-1)**(i+j) * np.linalg.det(np.delete(np.delete(matrix, i, axis=0), j, axis=1)))
         for j in range(len(matrix))]
        for i in range(len(matrix))
    ]}
