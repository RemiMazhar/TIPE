import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# gradient of f(a,b,c) = sum  (axi^2 + bxi + c - yi)^2
def poly2_gradient_batch(coeffs, xs, ys):
    a,b,c = coeffs
    x2s = xs*xs
    diffs = a * x2s + b * xs + c - ys

    # (axi^2 + bxi + c - yi) * [xi^2   xi   1].T
    gradient_by_pts = np.stack((x2s, xs, np.ones_like(xs)), axis=0) * 2 * diffs # gradient for each (xi,yi)

    return np.sum(gradient_by_pts, axis=-1)# / len(xs)

def reg_poly2_experimental(xs, ys, c0=np.zeros(3)):
    # c0: initial guess

    ka = 2 * np.sum(xs**4)
    kb = 2 * np.sum(xs**2)
    kc = 2 * len(xs)
    #k = np.array([ka, kb, kc])
    k = np.sqrt(ka**2 + kb**2 + kc**2)
    step = 1 / k

    coeffs = c0

    for i in range(20000):
        grad = poly2_gradient_batch(coeffs, xs, ys)
        coeffs -= grad * step
        if np.dot(grad, grad) <= step * 1e-2: return coeffs
        
    return coeffs

def poly2(x, a, b, c):
    return a * x * x  + b * x + c

def fit_parabola_scipy(xs, ys):
    popt, pcov = curve_fit(poly2, xs, ys)
    return popt

def affine(x, a, b):
    return a * x + b * x

def fit_line_scipy(xs, ys):
    popt, pcov = curve_fit(affine, xs, ys)
    return popt

if __name__=='__main__':
    n = 50
    xs = np.random.uniform(0, 0.6, n)
    ys = -10 * xs**2 + 2 * xs + 18

    noise = np.random.normal(0,0.01,n)
    ys += noise

    coeffs = reg_poly2_experimental(xs, ys)
    a,b,c = coeffs

    predxs = np.linspace(min(xs), max(xs), 100)
    predys = a * predxs**2 + b * predxs + c
    print(a, b, c)

    plt.scatter(xs, ys)
    plt.plot(predxs, predys)
    plt.show()