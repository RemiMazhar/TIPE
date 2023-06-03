import numpy as np
from object_detection import get_ball_position

# à chaque fois gradient = gradient de la distance^2 à l'objet

class Camera:
    def __init__(self, pos, orientation, ratio):
        self.ratio = ratio # (x_ref_reel / x_ref_ecran / z_ref_reel)
        self.pos = pos # position de la caméra
        self.orientation = orientation # matrice de changement de base tell que M * (v dans B_caméra) = (v dans B_universelle)

    # renvoie la direction et la distance estimée de la balle
    def get_ball(self, img):
        x_ecran, y_ecran, r_ecran = get_ball_position(img)

        x = self.ratio * x_ecran
        y = y_ecran / x_ecran * x
        z = 1
        dir = np.array([x, y, z])
        dir = np.dot(self.orientation, dir / np.linalg.norm(dir))
        
        # d = r_reel / (ratio * r_ecran)
        # enfin d ou z en fonction de si on approxime sinx=x
        dst = abs (2 / (self.ratio  * r_ecran))
 
        return dir, dst
    
    def get_droite(self, img):
        dir, dst = self.get_ball(img)
        return Droite(self.pos, dir)
    
    def get_sphere(self, img):
        dir, dst = self.get_ball(img)
        return Sphere(self.pos, dst)
    
    def get_plan(self, img):
        dir, dst = self.get_ball(img)
        n = np.dot(self.orientation, np.array([0,0,-1]))
        A = self.pos + n * dst
        return Plan(A, n)
    
    def get_all(self, img):
        dir, dst = self.get_ball(img)
        n = np.dot(self.orientation, np.array([0,0,-1]))
        A = self.pos + n * dst
        return {
            'droite' : Droite(self.pos, dir),
            'sphere' : Sphere(self.pos, dst),
            'plan' : Plan(A, n)
        }


class Droite:
    def __init__(self, A, d):
        self.A = A
        self.d = d
        # droite passant par A de vecteur directeur d

    def distance(self, M):
        AM = M - self.A
        return np.linalg.norm(AM - np.dot(AM, self.d) * self.d)
    
    def gradient(self, M):
        AM = M - self.A
        return 2 * (AM - np.dot(AM, self.d) * self.d)
    
class Sphere:
    def __init__(self, O, r):
        self.O = O
        self.r = r
        # sphère de centre O et de rayon r
    
    def distance(self, M):
        OM = M - self.O
        return abs(np.linalg.norm(OM) - self.r)

    def gradient(self, M):
        OM = M - self.O
        return 2 * OM * (1 - self.r / np.linalg.norm(OM))
    
class Plan:
    def __init__(self, A, n):
        self.A = A
        self.n = n
        # plan passant par A de vecteur normal n
    
    def distance(self, M):
        AM = M - self.A
        return abs(np.dot(self.n, AM))

    def gradient(self, M):
        AM = M - self.A
        return 2 * self.n * np.dot(self.n, AM)