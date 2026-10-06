"""Independent Lorentz contractions and native complex Breit-Wigner amplitudes."""

import numpy as np

MPI = 0.13957039
MD = 1.86965
METRIC = np.array([1., -1., -1., -1.])


def _minkowski(a, b):
    return np.sum(METRIC*a*b)


def _kallen(x, y, z):
    return x*x + y*y + z*z - 2*x*y - 2*x*z - 2*y*z


def _angular(p1, p2, p3):
    r, d, b = p1+p2, p1-p2, p1+p2+2*p3
    s = _minkowski(r, r)
    dr, rb = _minkowski(d, r), _minkowski(r, b)
    z1 = _minkowski(d, b) - dr*rb/s
    dhd = _minkowski(d, d) - dr*dr/s
    bhb = _minkowski(b, b) - rb*rb/s
    return np.array([1., z1, z1*z1-dhd*bhb/3])


def oracle(point):
    m, c = point["coordinates"]
    q = np.sqrt(_kallen(MD*MD, m*m, MPI*MPI))/(2*m)
    k = np.sqrt(_kallen(m*m, MPI*MPI, MPI*MPI))/(2*m)
    e1 = (MD*MD-m*m-MPI*MPI)/(2*m)
    sine = np.sqrt(1-c*c)
    p1 = np.array([e1, 0., 0., -q])
    p2 = np.array([m/2, k*sine, 0., k*c])
    p3 = np.array([m/2, -k*sine, 0., -k*c])
    masses = np.array([.98, .775, 1.275])
    widths = np.array([.07, .149, .185])
    couplings = np.array([1., .8, .5])*np.exp(1j*np.array([0., .7, -1.2]))

    def amplitude(p_second, p_third):
        r = p1 + p_second
        s = _minkowski(r, r)
        # This sign matches the example's explicit Breit-Wigner expression.
        bw = 1/(s-masses*masses-1j*masses*widths)
        return np.sum(couplings*bw*_angular(p1, p_second, p_third))

    total = amplitude(p2, p3) + amplitude(p3, p2)
    return float(np.log(m*q*k*abs(total)**2))


def grad_oracle(point):
    x = np.asarray(point["coordinates"], dtype=float)
    h = 1e-5
    result = []
    for i in range(2):
        d = np.eye(2)[i]*h
        f = lambda offset: oracle({"coordinates": x+offset*d})
        result.append((f(-2)-8*f(-1)+8*f(1)-f(2))/(12*h))
    return {"coordinates": result}
