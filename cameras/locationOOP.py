import numpy as np
import cv2
import matplotlib.pyplot as plt
from geometry import Camera, Droite, Sphere, Plan

RATIO = -0.0016582359671898303 # ratio pour le téléphone de rémi

"""-------------------------descente de gradient-------------------------"""

def descente(geometries, coeffs=[], dl=0.01, threshold=0.001):
    # geometries: liste d'objets géométriques (droites, sphères...) dont on peut calculer le gradient de la distance^2
    # trouve le point qui minimise la distance^2 moyenne, par descente de gradient
    # coeffs: poids de chaque objet dans la moyenne qu'on cherche à minimiser

    if len(coeffs) < len(geometries): coeffs = [1]*len(geometries)

    M = np.array([0.,0.,0.])
    grad = threshold + 1
    while np.dot(grad, grad) > threshold:
        grad = sum(g.gradient(M)*c for g,c in zip(geometries, coeffs))
        M -= grad * dl
    return M

"""-------------------------affichage-------------------------"""

def plot_dtes(dtes):
    # dtes: liste de tuples (A, d) correspondant à la droite passant par A de vecteur directeur d
    # graphe les droites projetées dans le plan y=0
    proj = np.array([
        [1,0,0],
        [0,0,1]
    ])
    for D in dtes:
        p1 = np.dot(proj, D.A)
        p1 = np.dot(proj, D.A - D.d * 100)
        plt.plot([p1[0], p1[0]], [p1[1], p1[1]])
    ax = plt.gca()
    ax.set_xlim(-50, 50)
    ax.set_ylim(50, -50)
    ax.set_aspect('equal')
    plt.show()

"""-------------------------le truc principal------------------------"""

def locate(data_points, trust=(1,0)):
    # data_points: liste de tuples (img, cam)
    # img une photo de la balle (toutes les photo de la liste ont été prises simultanément)
    # cam la caméra avec laquelle elle a été prise
    # trust: (a, b) respectivement coefficients de confiance dans la direction et dans la distance estimées

    geometries = []
    coeffs = []
    # calcul des vecteurs directeurs dans B0
    for img, cam in data_points:
        ball_geoms = cam.get_all(img)

        dte = ball_geoms['droite']
        plan = ball_geoms['plan']

        geometries.append(dte)
        coeffs.append(trust[0])

        geometries.append(plan)
        coeffs.append(trust[1])
        
    # plot_droites(droites)

    return descente(geometries, coeffs)



"""-------------------------test-------------------------"""

if __name__ == '__main__':
    Oc1 = np.array([0, 14, 50], dtype=float)
    Bc1ToB0 = np.identity(3, dtype=float)
    cam1 = Camera(Oc1, Bc1ToB0, RATIO)

    Oc2 = np.array([50, 14, 0], dtype=float)
    Bc2ToB0 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=float)
    cam2 = Camera(Oc2, Bc2ToB0, RATIO)

    names = ['0_2_0', '0_15_0', '0_30_0', '-15_2_15', '-15_15_15', '-15_30_15', '-30_2_-30', '-30_15_-30', '-30_30_-30']

    for name in names:
        img1 = cv2.imread('data/0_14_50/' + name + '.jpg')
        img2 = cv2.imread('data/50_14_0/' + name + '.jpg')
        pos_reel = np.array([int(s) for s in name.split('_')])
        print('position réelle:', pos_reel)
        pos_est = locate([(img1, cam1), (img2, cam2)])
        print('position estimée:', pos_est)
        print('distance:', np.linalg.norm(pos_reel - pos_est), 'cm')
        print()