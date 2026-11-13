"""Independent oracles for the LHCb amplitude fixtures (`corpora/stablehlo/lhcb_*`).

The reference values come from the source JSON files (`references.json`, see
`README.md`). The functions below follow the text of spec §09 "Kinematics
functions" and "Resonance functions", not the FlatPPL lowering:

* `breakup_momentum` uses the Källén form sqrt(λ(m², ma², mb²)) / (2m);
* `blatt_weisskopf` uses the spherical Bessel functions behind §09's barrier
  polynomials, χ_ℓ(z) = z^(ℓ+1) (j_ℓ(√z)² + y_ℓ(√z)²);
* `resonance_breitwigner` is §09's 1 / (m² − σ − i m Γ(σ)).

A density is the model's own `rate`: its `phase_space_density` binding times
the intensity, inside `decay_space` (spec §06 `weighted`), and −∞ outside it.
"""
import json
from pathlib import Path

import numpy as np
from scipy import special

DIR = Path(__file__).resolve().parent
REFERENCES = json.loads((DIR / "references.json").read_text())


def kallen(x, y, z):
    return x * x + y * y + z * z - 2 * x * y - 2 * y * z - 2 * z * x


def breakup_momentum(m, ma, mb):
    return np.sqrt(kallen(m * m, ma * ma, mb * mb)) / (2 * m)


def blatt_weisskopf(l, p, d):
    z = (d * p) ** 2
    x = np.sqrt(z)
    chi = z ** (l + 1) * (special.spherical_jn(l, x) ** 2 + special.spherical_yn(l, x) ** 2)
    return np.sqrt(z ** l / chi)


def resonance_breitwigner(s, mass, width, ma, mb, l, d):
    p = breakup_momentum(np.sqrt(s), ma, mb)
    p0 = breakup_momentum(mass, ma, mb)
    running = width * mass / np.sqrt(s) * p / p0 * (blatt_weisskopf(l, p, d) / blatt_weisskopf(l, p0, d)) ** 2
    return 1 / (mass * mass - s - 1j * mass * running)


def lineshape(model, name, point):
    """The real or imaginary part of lineshape `name` at `point["s"]`: the JSON
    checksum at its own σ, and §09's Breit-Wigner at any other σ."""
    entry = REFERENCES[model]["lineshapes"][name]
    s = point["s"]
    if s == entry["x"]:
        re, im = entry["value"]
        value = complex(re, im)
    else:
        bw = entry["breit_wigner"]
        value = resonance_breitwigner(s, bw["mass"], bw["width"], bw["ma"], bw["mb"], bw["l"], bw["d"])
    return value.imag if point["imaginary"] else value.real


# Masses (GeV) as the models declare them.
MASSES = {
    "lb2pkg-lhcb-2765817": (5.62, 0.938, 0.493, 0.0),
    "lc2ppik-lhcb-2683025": (2.28646, 0.938272046, 0.13957018, 0.493677),
    "b2dxdk-lhcb-2794793-Cminus": (5.27934, 1.86483, 0.13957039, 1.86965, 0.493677),
    "b2dxdk-lhcb-2794793-Cplus": (5.27934, 1.86483, 0.13957039, 1.86965, 0.493677),
}
# The fixed mass of the narrow (12) line of the B+ models.
B_M12 = 2.01026


def three_body_phase_space(model, point, prefix=""):
    """`phase_space_density` of the Λb (`q_12_3 * q_12`) and Λc (`q_31_2 * q_31`) models."""
    m0, m1, m2, m3 = MASSES[model]
    if model.startswith("lb2pkg"):
        m12 = point[prefix + "m_12"]
        return breakup_momentum(m0, m12, m3) * breakup_momentum(m12, m1, m2)
    m31 = point[prefix + "m_31"]
    return breakup_momentum(m0, m31, m2) * breakup_momentum(m31, m3, m1)


def three_body_inside(model, point, prefix=""):
    m0, m1, m2, m3 = MASSES[model]
    lo, hi, mass = ((m1 + m2, m0 - m3, point[prefix + "m_12"]) if model.startswith("lb2pkg")
                    else (m3 + m1, m0 - m2, point[prefix + "m_31"]))
    angles = [v for k, v in point.items() if "cos_theta" in k]
    phis = [v for k, v in point.items() if k.startswith(prefix + "phi")]
    return (lo <= mass <= hi and all(-1 <= c <= 1 for c in angles)
            and all(-np.pi <= f <= np.pi for f in phis))


# ---- B+ -> D0 pi+ D- K+ kinematics, following the model's own frame functions


def _mass(p):
    return np.sqrt(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2)


def _rot_z(p, a):
    return np.array([p[0], p[1] * np.cos(a) - p[2] * np.sin(a), p[1] * np.sin(a) + p[2] * np.cos(a), p[3]])


def _rot_y(p, a):
    return np.array([p[0], p[3] * np.sin(a) + p[1] * np.cos(a), p[2], p[3] * np.cos(a) - p[1] * np.sin(a)])


def _boost_z(p, g, bg):
    return np.array([g * p[0] + bg * p[3], p[1], p[2], g * p[3] + bg * p[0]])


def _angles(p):
    return np.arctan2(p[2], p[1]), np.arctan2(np.hypot(p[1], p[2]), p[3])


def _to_frame(p, P):
    phi, theta = _angles(P)
    m = _mass(P)
    return _boost_z(_rot_y(_rot_z(p, -phi), -theta), P[0] / m, -np.linalg.norm(P[1:]) / m)


def _from_frame(p, P):
    phi, theta = _angles(P)
    m = _mass(P)
    return _rot_z(_rot_y(_boost_z(p, P[0] / m, np.linalg.norm(P[1:]) / m), theta), phi)


def _momentum(q, c, phi, m):
    s = q * np.sqrt(max(1 - c * c, 0.0))
    return np.array([np.sqrt(q * q + m * m), s * np.cos(phi), s * np.sin(phi), q * c])


def _opposite(p, m):
    return np.array([np.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2 + m * m), -p[1], -p[2], -p[3]])


def decay_coordinates(model, momenta):
    """The decay coordinates of the topology [[[1, 2], 3], 4] whose four-momenta
    (the model's `momenta_fn`) are `momenta`, in the parent rest frame."""
    p1, p2, p3, _ = (np.asarray(p, dtype=float) for p in momenta)
    p123 = p1 + p2 + p3
    phi123, theta123 = _angles(p123)
    q1, q2 = _to_frame(p1, p123), _to_frame(p2, p123)
    p12 = q1 + q2
    phi12, theta12 = _angles(p12)
    r1 = _to_frame(q1, p12)
    phi1, theta1 = _angles(r1)
    return {
        "q_cos_theta_12_3_4": np.cos(theta123), "q_phi_12_3_4": phi123, "q_m_12_3": _mass(p123),
        "q_cos_theta_12_3": np.cos(theta12), "q_phi_12_3": phi12,
        "q_cos_theta_12": np.cos(theta1), "q_phi_12": phi1,
    }


def momenta(model, x):
    """The model's four-momenta at decay coordinates `x` (its `momenta_fn`)."""
    m0, m1, m2, m3, m4 = MASSES[model]
    q1234 = breakup_momentum(m0, x["q_m_12_3"], m4)
    e123 = _momentum(q1234, x["q_cos_theta_12_3_4"], x["q_phi_12_3_4"], x["q_m_12_3"])
    e4 = _opposite(e123, m4)
    e12 = _momentum(breakup_momentum(x["q_m_12_3"], B_M12, m3), x["q_cos_theta_12_3"], x["q_phi_12_3"], B_M12)
    e3 = _opposite(e12, m3)
    e1 = _momentum(breakup_momentum(B_M12, m1, m2), x["q_cos_theta_12"], x["q_phi_12"], m1)
    e2 = _opposite(e1, m2)
    return [_from_frame(_from_frame(e1, e12), e123), _from_frame(_from_frame(e2, e12), e123),
            _from_frame(e3, e123), e4]


def four_body_phase_space(model, x):
    """`phase_space_density = q_12_3_4 * q_12_3 * q_12` of the B+ models."""
    m0, m1, m2, m3, m4 = MASSES[model]
    return (breakup_momentum(m0, x["q_m_12_3"], m4) * breakup_momentum(x["q_m_12_3"], B_M12, m3)
            * breakup_momentum(B_M12, m1, m2))
