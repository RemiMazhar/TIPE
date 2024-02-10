import numpy as np
import sympy as sp
import itertools
from geometry import Camera
import time
from locate_camera import get_camera
from polynom_evaluation import compute_monoms, evaluate_monom_table, precalculate_monom_table, compute_monom_bruteforce
import json

RATIO = -0.0016

def get_dir(pos_ecran):
    # prend un point sur l'écran donne un vecteur unitaire dans la direction qui correspond
    x_ecran, y_ecran = pos_ecran
    x = RATIO * x_ecran
    y = y_ecran / x_ecran * x
    z = 1
    dir = np.array([x, y, z])
    return dir / np.linalg.norm(dir)

def sylvester(coeffs1, coeffs2):
    # coeffs1 les coefficients du polynome 1 (les coefficients peuvent être des polynômes)
    # coeffs2 les coefficients du polynome 2 (les coefficients peuvent être des polynômes)
    # coefficients par ordre de degré décroissant
    # calcule le déterminant de sylvester des polynomes 1 et 2
    deg1 = len(coeffs1) - 1
    deg2 = len(coeffs2) - 1
    mat = sp.matrices.zeros(deg1 + deg2)
    for i in range(deg2):
        for j in range(deg1+1):
            mat[i, i+j] = coeffs1[j]
    for i in range(deg1):
        for j in range(deg2+1):
            mat[deg2 + i, i+j] = coeffs2[j]
    t = time.time()
    res = mat.det()
    print('determinant calculation time:', time.time() - t)
    return res

def find_solutions(coeffsg, param_values):
    cos12, cos13, cos23, d12, d13, d23 = param_values
    r = np.roots(coeffsg) # racines de g (y compris complexes ou négatives, il les trouve numériquement)
    r = r.real[abs(r.imag)<1e-5] # on vire les complexes
    r = r[r.real>=0] # on vire les negatives
    x1s = np.sqrt(r.real)

    solutions = []

    for x1 in x1s:
        # pour chaque valeur de x1, on solve f12(x2)=0 et f13(x3)=0 séparément
        # puis on prend les solutions qui respectent tout
        r = np.roots([1, -2 * x1 * cos12, x1**2 - d12**2])
        r = r.real[abs(r.imag)<1e-5]
        x2s = r[r.real>=0]

        r = np.roots([1, -2 * x1 * cos13, x1**2 - d13**2])
        r = r.real[abs(r.imag)<1e-5]
        x3s = r[r.real>=0]

        for x2, x3 in itertools.product(x2s, x3s):
            if abs(x2**2 + x3**2 -2*x2*x3*cos23 - d23**2) < 1e-5:
                solutions.append((x1, x2, x3))
    return solutions

def solve_distances(data):
    # u1, u2, u3: points sur l'écran
    # p1, p2, p3: points en vrai
    (u1, p1), (u2, p2), (u3, p3) = data

    cos12 = np.dot(get_dir(u1), get_dir(u2))
    cos13 = np.dot(get_dir(u1), get_dir(u3))
    cos23 = np.dot(get_dir(u2), get_dir(u3))

    d12 = np.linalg.norm(p1 - p2)
    d13 = np.linalg.norm(p1 - p3)
    d23 = np.linalg.norm(p2 - p3)

    # xi distance entre la caméra et pi
    x1, x2, x3, x = sp.symbols('x1 x2 x3 x') # x = x1**2

    t = time.time()
    # les 3 équations polynomiales du papier
    f12 = (x1**2 + x2**2 - 2 * cos12 * x1 * x2 - d12**2).as_poly()
    f13 = (x1**2 + x3**2 - 2 * cos13 * x1 * x3 - d13**2).as_poly()
    f23 = (x2**2 + x3**2 - 2 * cos23 * x2 * x3 - d23**2).as_poly()

    # élimine x3
    coeffs1 = sp.Poly(f13, x3).all_coeffs() #extract_polynomial_coeffs(f13, x1, x3)
    coeffs2 = sp.Poly(f23, x3).all_coeffs() #extract_polynomial_coeffs(f23, x2, x3)
    h = sylvester(coeffs1, coeffs2) # h comme dans le papier (polynome en x1 et x2)

    # élimine x2
    coeffs1 = sp.Poly(f12, x2).all_coeffs() #extract_polynomial_coeffs(f12, x1, x2)
    coeffs2 = sp.Poly(h, x2).all_coeffs() #extract_polynomial_coeffs(h, x1, x2)
    g = sylvester(coeffs1, coeffs2) # g polynome de degré 8 avec seulement des termes pairs
    g = sp.Poly(g, x1)
    coeffsg = g.all_coeffs()[::2]
    
    print('total time to compute the coefficients of g:', time.time() - t)

    return find_solutions(coeffsg, [cos12, cos13, cos23, d12, d13, d23])

def make_triad(p1, p2, p3):
    x = p1 - p2
    x /= np.linalg.norm(x)
    y = (p3 - p1) - np.dot(p3 - p1, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1)

def PnP(data, precalc=[]):
    # notations: pwi = point numéro i dans le référentiel du monde, pci = dans le référentiel de la caméra
    # on cherche R rotation de la caméra vers le monde
    # ie R(pci) = pwi
    (u1, pw1), (u2, pw2), (u3, pw3) = data
    dir1 = get_dir(u1)
    dir2 = get_dir(u2)
    dir3 = get_dir(u3)
    triad_world = make_triad(pw1, pw2, pw3)
    solutions = []
    if precalc:
        distances = solve_distances_precalc(data, precalc)
    else:
        distances = solve_distances(data)
    for d1,d2,d3 in distances:
        pc1 = dir1 * d1
        pc2 = dir2 * d2
        pc3 = dir3 * d3
        triad_camera = make_triad(pc1, pc2, pc3)
        # R * triad_camera = triad_world
        # R = triad_world * triad_camera.T
        R = np.dot(triad_world, triad_camera.T)
        T = pw1 - np.dot(R, pc1)
        solutions.append(Camera(T, R, RATIO))
    return solutions

def precalculator():
    x1, x2, x3, x = sp.symbols('x1 x2 x3 x') # x = x1**2
    cos12, cos13, cos23 = sp.symbols('cos12 cos13 cos23')
    d12, d13, d23 = sp.symbols('d12 d13 d23')
    f12 = (x1**2 + x2**2 - 2 * cos12 * x1 * x2 - d12**2)
    f13 = (x1**2 + x3**2 - 2 * cos13 * x1 * x3 - d13**2)
    f23 = (x2**2 + x3**2 - 2 * cos23 * x2 * x3 - d23**2)
    h = sp.polys.polytools.resultant(sp.Poly(f13, x3), sp.Poly(f23, x3))
    g = sp.polys.polytools.resultant(sp.Poly(h, x2), sp.Poly(f12, x2))
    coeffs = [sp.Poly(c, gens=(cos12, cos13, cos23, d12, d13, d23)) for c in sp.Poly(g, x1).all_coeffs()[::2]]
    #res = [prepare_horner(c, (cos12, cos13, cos23, d12, d13, d23)) for c in coeffs]
    precalculated_coeffs = [c.terms() for c in coeffs]
    monoms_to_calculate, monom_table = precalculate_monom_table()
    return precalculated_coeffs, monoms_to_calculate, monom_table

def solve_distances_precalc(data, precalculated):
    t = time.time()
    (u1, p1), (u2, p2), (u3, p3) = data
    
    cos12 = np.dot(get_dir(u1), get_dir(u2))
    cos13 = np.dot(get_dir(u1), get_dir(u3))
    cos23 = np.dot(get_dir(u2), get_dir(u3))

    d12 = np.linalg.norm(p1 - p2)
    d13 = np.linalg.norm(p1 - p3)
    d23 = np.linalg.norm(p2 - p3)

    precalculated_coeffs, monoms_to_calculate, monom_table = precalculated
    monom_values = compute_monoms(monoms_to_calculate, [cos12, cos13, cos23, d12, d13, d23], monom_table)

    coeffsg = [0] * len(precalculated_coeffs)
    
    for i, terms in enumerate(precalculated_coeffs):
        coeffsg[i] = evaluate_monom_table(terms, monom_values)

    print('total time to compute the coefficients of g:', time.time() - t)

    return find_solutions(coeffsg, [cos12, cos13, cos23, d12, d13, d23])

if __name__=='__main__':
    data = [((315-300, 400-562), np.array([0,0,0], dtype=float)),
            ((50-300, 400-636), np.array([-15,0,15], dtype=float)),
            ((96-300, 400-496), np.array([-30,0,-30], dtype=float))]
    print('precalculating...')
    precalculated = precalculator()
    print('finished precalculating')

    print('\n---old, numerical method---')
    t = time.time()
    cam = get_camera(data)
    print(time.time() - t, 's')
    print('camera position:', cam.pos)

    print('\n---new, meth method, just calculate everything with sympy---')
    t = time.time()
    cameras = PnP(data)
    print(time.time() - t, 's')
    print('candidate camera positions:')
    for camera in cameras:
        print(camera.pos)

    print('\n---new, meth method, but precalculate stuff---')
    t = time.time()
    cameras = PnP(data, precalculated)
    print(time.time() - t, 's')
    print('candidate camera positions:')
    for camera in cameras:
        print(camera.pos)