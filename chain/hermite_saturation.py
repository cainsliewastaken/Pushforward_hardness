"""Layer statistics of saturation. Unit pre-activation z = z0 + a*s with z0 ~ N(0, sig0^2) (init variance, independent
of x) and s = f(x) a smooth spike feature, s ~ N(0,1) over the data. Derivative factor g(s) = E_{z0}[tanh'(z0 + a s)].
Hermite expansion g(s) = sum_m c_m(a) He_m(s)/sqrt(m!): energy at degree m is the roughness-class spectrum of the
derivative factor; degree m carries roughness ~ m * rho_f. Report, vs amplitude a: energy fraction by degree,
mean degree (roughness index proxy), and the fraction beyond degree 4 ("tail"), for sig0 in {0.5, 1.0}."""
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial
def coeffs(a, sig0, M=24, ns=4001, nz=401):
    s = np.linspace(-6, 6, ns); z0 = np.linspace(-5*sig0, 5*sig0, nz) if sig0 > 0 else np.array([0.])
    wz = np.exp(-z0**2/(2*max(sig0, 1e-9)**2)); wz /= wz.sum()
    g = np.array([(wz*(1-np.tanh(z0 + a*si)**2)).sum() for si in s])
    ws = np.exp(-s**2/2)/np.sqrt(2*np.pi)*(s[1]-s[0])
    c = np.array([np.sum(ws*g*hermeval(s, [0]*m+[1]))/np.sqrt(factorial(m)) for m in range(M)])
    return c
print(f"{'sig0':>5s} {'a':>5s} | {'E0':>5s} {'E1':>5s} {'E2':>5s} {'E3':>5s} {'E4':>5s} {'E>4':>6s} | {'mean deg':>8s} (excluding degree 0)")
for sig0 in [0.5, 1.0, 1.5]:
    for a in [0.1, 0.25, 0.5, 1, 2, 4, 8]:
        c = coeffs(a, sig0); e = c**2; e1 = e[1:]/e[1:].sum(); md = (np.arange(1, len(c))*e1).sum()
        print(f"{sig0:5.2f} {a:5.2f} | " + ' '.join(f'{e[m]/e.sum():5.2f}' for m in range(5)) + f" {e[5:].sum()/e.sum():6.3f} | {md:8.2f}")
print("\nmeasured for comparison (section 21): roughness index of the backprop factor 0.025 (init) -> 0.15-0.67 (end) as the")
print("hidden amplitude product goes 0.1 -> 30; index of activations stays 0.06; smooth-feature index 0.06, so degree m ~ roughness 0.06 m")
