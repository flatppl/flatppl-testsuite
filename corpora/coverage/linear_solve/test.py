"""§07 Linear algebra: linsolve(A,b) solves Ax=b. For this nonsymmetric
matrix det(A)=4 and inverse(A)=[[2,-1],[-2,3]]/4. The scalar projection
x1+3*x2 differs from both matrix rows and exposes a transposed solve."""
import math


def oracle(point: dict) -> float:
    b1, b2 = point["b"]
    x1 = (2 * b1 - b2) / 4
    x2 = (-2 * b1 + 3 * b2) / 4
    mu = x1 + 3 * x2
    return -0.5 * math.log(2 * math.pi) - math.log(2) - 0.5 * ((9 - mu) / 2) ** 2
