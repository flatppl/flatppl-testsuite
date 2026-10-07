"""Closed-form exponential-tilt normalizers with independent live intervals."""
import math


def oracle(point):
    return sum(a*(x-lo) - math.log(math.expm1(a*(hi-lo))/a)
               for a, lo, hi, x in zip(point["rates"], point["lows"],
                                        point["highs"], point["points"]))


def grad_oracle(point):
    result = {key: [] for key in ("rates", "lows", "highs", "points")}
    for a, lo, hi, x in zip(point["rates"], point["lows"],
                            point["highs"], point["points"]):
        gap = hi-lo
        denominator = math.expm1(a*gap)
        result["rates"].append(x-lo + 1/a - gap*(denominator+1)/denominator)
        result["lows"].append(a/denominator)
        result["highs"].append(-a*(denominator+1)/denominator)
        result["points"].append(a)
    return result
