"""Spec §03 "Note on infinities": `reals` admits `inf` and `-inf` as legal values,
but "the reference measure is the ordinary Lebesgue measure on the finite-real
part; the points ±∞ carry zero Lebesgue mass". So `inf` lies outside the support
of `LogNormal`, the density tends to 0 there, and the log-density is `-inf`. A
formula such as `(k - 1) log(x) - rate x` gives the IEEE artifact `inf - inf =
NaN` instead."""
import math


def oracle(point):
    return -math.inf
