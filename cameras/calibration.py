from object_detection import get_ball_position
import cv2

"""script pour mesurer RATIO (constante qui dépend de la caméra)"""

# repère utilisé ici:
#   origine: caméra
#   x: vers la droite de l'écran
#   y: vers le haut de l'écran
#   z: direct

# notations:
# _ref_ (référence): lié à l'image de calibration, où on connait la position de la balle
# _mes_ (mesure): lié à une image quelconque, où on cherche à connaitre la position de la balle
# _reel: coordonnées en vrai en cm
# _ecran: coordonnées sur l'écran en pixels, origine au centre de l'écran, x vers la droite y vers le haut

# ensuite dans le plan z_ref:
#   x_mes_reel = (x_ref_reel / x_ref_ecran) * x_mes_ecran
#   y_mes_reel = y_mes_ecran / x_mes_ecran * x_mes_reel
#   z_mes_reel = z_ref
#
# dans le plan z=1:
#   x_mes_reel = (x_ref_reel / x_ref_ecran / z_ref_reel) * x_mes_ecran
#   y_mes_reel = y_mes_ecran / x_mes_ecran * x_mes_reel
#   z_mes_reel = 1
#
# => seule donnée de calibration nécessaire: RATIO = (x_ref_reel / x_ref_ecran / z_ref_reel)

# image de calibration
img = cv2.imread("14_03_2023/calibration2/-15_-14_-65.jpg")

# position de la balle sur l'écran
x_ref_ecran, y, radius = get_ball_position(img)

# position réelle de la balle
x_ref_reel = -15
z_ref_reel = -65

ratio = (x_ref_reel / x_ref_ecran / z_ref_reel)
print(ratio)

# result from 0_14_50/-15_15_15: ratio=-0.0016582359671898303

# results from -15_-14_-65 (camera): ratio=-0.0015344051297181388