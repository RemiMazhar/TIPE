import sympy as sp
import numpy as np
import itertools

PRIMITIVES = [tuple(x) for x in np.eye(6, dtype=int).tolist()]

def powers(x, n):
    res = [1]*(n+1)
    val = 1
    for i in range(n+1):
        res[i] = val
        val *= x
    return res

def evaluate_pow_lookup(terms, pow_lookup):
    res = 0
    for monom, coeff in terms:
        monom_val = 1
        for k in range(6):
            monom_val *= pow_lookup[k][monom[k]]
        res += coeff * monom_val
    return res

def prepare_horner(P, vars):
    # exprime un polynome multivariable de maniere recursive, comme une liste de coeff dont chacun est un polynome multivariable
    # P: un polynome multivariable de sympy
    # vars: les symboles générateurs de P
    if not vars: return P
    #renvoie dans le mauvais sens pcq c le bon sens pour horner
    return [prepare_horner(c, vars[1:]) for c in sp.Poly(P, vars[0]).all_coeffs()]

def evaluate_horner(P, vals):
    # P: un polynome multivariable préparé avec prepare_horner
    # vals: les valeurs des générateurs
    if not vals: return P
    othervals = vals[1:]
    tot = 0
    for coeff in P:
        c = evaluate_horner(coeff, othervals)
        tot = tot * vals[0] + c
    return tot

def find_divisors(monom):
    vals = [list(range(k+1)) for k in monom]
    return itertools.product(*vals)

def add_monoms(m1, m2):
    return tuple(map(sum, zip(m1, m2)))

def compute_monom_bruteforce(m, vals):
    res = 1
    for i in range(6):
        res *= vals[i] ** m[i]
    return res

def make_table(to_calculate, divisors, dep):
    table = dict()
    computations = dict()
    for x in to_calculate:
        table[x] = tuple()
        computations[x] = 1000
    for x in dep:
        table[x] = tuple()
        computations[x] = 0

    calculated = dep
    for _ in range(10):
        for m1, m2 in itertools.combinations_with_replacement(calculated, 2):
            s = add_monoms(m1, m2)
            if s in divisors:
                calculated.add(s)
                #if s not in table or not table[s]:
                if s not in table or computations[s] >= computations[m1] + computations[m2] + 1:
                    table[s] = (m1, m2)
                    computations[s] = computations[m1] + computations[m2] + 1
    
    return table

def compute1monom(monom_table, values, m):
    if m in values and values[m] != -1: return # cas de base: monome déjà calculé

    if not monom_table[m]: # autre cas de base: monome incalculable à partir d'autre monomes, on le calcule à partir des primitives
        values[m] = 1
        for i in range(6):
            values[m] *= values[PRIMITIVES[i]] ** m[i]
        return
    
    m1, m2 = monom_table[m]
    compute1monom(monom_table, values, m1)
    compute1monom(monom_table, values, m2)
    values[m] = values[m1] * values[m2]

def compute_monoms(to_calculate, prim_vals, monom_table):
    usefulvalues = dict()
    allvalues = dict()
    for i in range(6):
        allvalues[PRIMITIVES[i]] = prim_vals[i]
    for m in to_calculate:
        compute1monom(monom_table, allvalues, m)
        usefulvalues[m] = allvalues[m]
    return usefulvalues

def precalculate_monom_table():
    to_calculate = [tuple(x) for x in np.load('monoms.npy').tolist()]
    divisors = set()
    for m in to_calculate:
        divisors.update(find_divisors(m))

    return to_calculate, make_table(to_calculate, divisors, set(PRIMITIVES))

def evaluate_monom_table(terms, monom_values):
    res = 0
    for monom, coeff in terms:
        res += coeff * monom_values[monom]
    return res

#for d in find_divisors((8,)*6):
#    score = compute_score(to_calculate, divisors, set([(0,)*6, d]))
#    if score >= 1:
#        print(d, score)