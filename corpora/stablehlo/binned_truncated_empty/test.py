"""A disjoint zero-count bin has zero mass and zero parameter derivatives."""


def oracle(point):
    return 0.0


def grad_oracle(point):
    return {"mu": 0.0, "a": 0.0}
