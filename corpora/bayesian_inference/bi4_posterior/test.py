"""Independent oracle for post_bi4.

``theta1 ~ Normal(0,1)``, ``theta2 ~ Exponential(rate=1)``, ``a = 5*theta2``,
``b = abs(theta1)*theta2``, ``obs ~ iid(Normal(mu=a, sigma=b), 10)``, scored
at ``theta1=0.5, theta2=1.0`` against the fixed 10-point ``observed_data``. A
``bayesupdate`` posterior's log-density at a point is the prior log-density
plus the likelihood log-density -- i.e. the joint log-density of (theta,
data) -- so bi1-4, which build that same joint four different ways (an
explicit ``joint`` prior, a ``lawof(record(...))`` prior, a
``disintegrate``d joint, and a ``restrict``ed joint), all freeze to this one
value.
"""
from scipy.stats import expon, norm


def oracle() -> float:
    data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
    t1, t2 = 0.5, 1.0
    a = 5 * t2
    b = abs(t1) * t2
    return (
        norm.logpdf(t1, 0, 1)
        + expon.logpdf(t2, scale=1.0)
        + sum(norm.logpdf(x, a, b) for x in data)
    )
