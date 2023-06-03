import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from geometry import Camera

RATIO = -0.0016

""""-------------------trouve la position de la caméra à partir de points dont on connait les coordonnées sur l'écran et dans l'espace-------------------"""
# retourne le vecteur unitaire de la caméra vers un point de coordonnées pos_ecran sur l'écran
# cam_mat: matrice de rotation de la caméra
def get_dte(cam_mat, pos_ecran, ratio):
    x_ecran, y_ecran = pos_ecran
    x = ratio * x_ecran
    y = y_ecran / x_ecran * x
    z = 1
    dir = np.array([x, y, z])
    dir = np.dot(cam_mat, dir / np.linalg.norm(dir))
    return dir

# distance^2 entre une droite de vecteur directeur dir passant par cam_pos avec un point M
def dst2(cam_pos, dir, M):
    AM = M - cam_pos
    if np.dot(AM, dir) > 0: return np.dot(AM, AM) # demi-droite
    HM = AM - np.dot(AM, dir) * dir
    return np.dot(HM, HM)

# data: tuples (pos_ec, M) où pos_ec=(x,y) position sur l'écran et M=np.array([x,y,z]) position réelle associée
def get_error(params, data):
    xc, yc, zc, xr, yr, zr = params
    # xc yc zc: position de la caméra
    # xr yr zr: vecteur rotation de la caméra
    cam_pos = np.array([xc, yc, zc])
    rot = R.from_rotvec(np.array([xr, yr, zr]))
    error = 0
    for pos_ec, M in data:
        dir = get_dte(rot.as_matrix(), pos_ec, RATIO)
        error += dst2(cam_pos, dir, M)
    return error 

def get_camera(data):
    to_min = lambda params: get_error(params, data)
    res = minimize(to_min, [0,0,0,0,0,0])
    #res = minimize(to_min, [5,17,96,0,0,0])

    print('success: ', res.success)
    print(res.message)
    print(to_min(res.x))

    xc, yc, zc, xr, yr, zr = res.x
    cam_pos = np.array([xc, yc, zc])
    rot = R.from_rotvec(np.array([xr, yr, zr])).as_matrix()
    return Camera(cam_pos, rot, RATIO)

"""-------------------interface graphique bordélique pour cliquer manuellement sur les points-------------------"""
class Calibrator:
    def __init__(self):
        self.screen_pos = []

    def callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            self.screen_pos.append([x, y])
            #print(x,y)
    
    # l'utilisateur donne lui même les positions réelles des points de référence
    def get_data_ask(self, img):
        cv2.imshow('pspsps', img)
        cv2.setMouseCallback('pspsps', self.callback)
        cv2.waitKey(0)
        real_pos = []

        for i in self.screen_pos:
            print(i)
            coords = [int(i) for i in input().split(' ')]
            real_pos.append(coords)
        
        return [(s, np.array(r)) for s, r in zip(self.screen_pos, real_pos)]

    # les positions rélles des points de référence sont passées en argument
    def get_data(self, img, real_pos):
        cv2.imshow('pspsps', img)
        cv2.setMouseCallback('pspsps', self.callback)
        cv2.waitKey(0)
        h, w = img.shape[:2]
        screen_pos_basemieux = [(x - w//2, h//2 - y) for x, y in self.screen_pos]
        self.screen_pos = []
        return [(s, np.array(r)) for s, r in zip(screen_pos_basemieux, real_pos)]


if __name__=='__main__':
    #data = [(s, np.array(r)) for s, r in zip(screen_pos, real_pos)]

    # example data
    #data = [((315-300, 400-562), np.array([0,0,0])),
    #        ((50-300, 400-636), np.array([-15,0,15])),
    #        ((96-300, 400-496), np.array([-30,0,-30]))]

    data = [((5, -164), np.array([0,0,0])),
            ((-134, -120), np.array([-15,0,15])),
            ((236, -101), np.array([-30,0,-30]))]

    to_min = lambda params: get_error(params, data)
    res = minimize(to_min, [0,0,0,0,0,0])
    xc, yc, zc, xr, yr, zr = res.x
    cam_pos = np.array([xc, yc, zc])
    rot = R.from_rotvec(np.array([xr, yr, zr])).as_matrix()
    print(np.round(cam_pos, 2))
    print(np.round(rot, 2))

    Bc2ToB0 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=float)

    v1 = np.array([xr, yr, zr])
    v2 = R.from_matrix(Bc2ToB0).as_rotvec()
    print('-'*100)
    print(v1)
    print(v2)
    print('norme mesuree:', np.linalg.norm(v1) * 180 / np.pi)
    print('norme relle:', np.linalg.norm(v2) * 180 / np.pi)
    print('difference norme:', abs(np.linalg.norm(v1) - np.linalg.norm(v2)) * 180 / np.pi)

    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    print('angle entre les 2:', angle * 180 / np.pi)