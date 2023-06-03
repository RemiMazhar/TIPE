import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy      import optimize
from color_distrib_maker import MultiGaussian

"""-------------------------constantes-------------------------"""

# ball color
BC_RGB = [255, 127, 39]
BC_HSV = [22, 170, 240] # 0 < H < 179, 0 < S < 255, 0 < V < 255

#TRG_HSV = [60, 255, 255] # couleur des triangles vert
#TRP_HSV = [135, 255, 255] # couleur des triangles violets
#TRG_HSV = [97, 204, 153]
#TRP_HSV = [160, 125, 179]
TRG_HSV = [10, 114, 166]
TRP_HSV = [166, 114, 166]

# distribution statistique des couleurs de la balle (déterminée expérimentalement)
# prévu pour la version avec distance de mahalanobis, pas utilisé au final car trop lent
BALL_RGB_MU = np.array([38.62810263, 155.50055349, 239.8427591])
BALL_RGB_COV = np.array([
    [918.00122152, 102.7568878, 27.6382168 ],
    [102.7568878, 1948.52842493, 574.52119993],
    [27.6382168, 574.52119993,316.66225301]
])
BALL_RGB_DISTRIB = MultiGaussian(BALL_RGB_MU, BALL_RGB_COV)


THRESHOLD = 0.18 # 0.15


"""-------------------------détection de couleur-------------------------"""

# calcule une différence de hue (chiant pcq modulo 180)
def dst_h(h1, h2):
    return np.minimum(abs(h1-h2), abs(h1-h2-180))

# crée un masque blanc pour les pixels oranges noir sinon
def make_mask(img, color=BC_HSV):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h,s,v = hsvimg[:,:,0],hsvimg[:,:,1],hsvimg[:,:,2]

    threshold = THRESHOLD

    # on calcule séparément la différence de H, de S, et de V (normalisée entre 0 et 1)
    # puis on fait une moyenne quadratique pondérée des différences
    ch, cs, cv = 0.7, 0.1, 0.2
    dh = dst_h(h, color[0])/90 # H defini modulo 180 donc la diff est entre 0 et 90
    ds = (s - color[1])/255
    dv = (v - color[2])/255
    dst = np.sqrt((ch * dh**2 + cs * ds**2 + cv * dv**2))

    mask = gaussian_filter(dst, sigma=1) <= threshold # un peu flouté pcq c mieux

    # erode/dilate: pas très clair ce que ça fait exactement, mais ça rend le truc plus clean mais moins précis
    #mask = cv2.erode(mask.astype(float), None, iterations=1)
    #mask = cv2.dilate(mask.astype(float), None, iterations=1)

    return (dst * 255).astype(np.uint8), (mask * 255).astype(np.uint8)

# pareil avec distance de mahalanobis (trop lent)
def make_mask_maha(img, distrib=BALL_RGB_DISTRIB):
    threshold = 0.3
    l,h,_ = img.shape
    dst = np.zeros((l,h))
    for i in range(l):
        for j in range(h):
            dst[i,j] = distrib.maha_dst(img[i,j]) / 20
    mask = gaussian_filter(dst, sigma=1) <= threshold # un peu flouté pcq c mieux
    mask = cv2.erode(mask.astype(float), None, iterations=1)
    mask = cv2.dilate(mask.astype(float), None, iterations=1)
    return (dst * 255).astype(np.uint8), (mask * 255).astype(np.uint8)

"""-------------------------détection et sélection du contour de la balle-------------------------"""

# retourne le best fit circle d'un contour
def best_circle(contour):
    #return cv2.minEnclosingCircle(contour)
    ps = cv2.convexHull(contour)[:,0]
    (x0, y0), r0 = cv2.minEnclosingCircle(contour)
    mse = lambda l: error(l, ps)
    res = optimize.minimize(mse, [x0, y0, r0])
    x, y, r = res.x
    return (x, y), r

# distance carrée moyenne des points ps à un cercle de paramètres l=[xcentre, ycentre, r]
def error(l, ps):
    x, y, r = l
    ctr = np.array([x, y])
    return sum([(np.linalg.norm(p - ctr) - r)**2 for p in ps]) / ps.shape[0]

# choisit le contour qui ressemble le plus à une balle
# en fonction de la taille et de la ronditude
def choose_best_contour(contours):
    best_contour = contours[0]
    best_score = 0
    for c in contours:
        # test de ronditude: (aire de la zone détectée / aire du best fit circle)
        # score entre 0 et 1, 0=pas rond, 1=très rond
        area = cv2.contourArea(c)
        ((x, y), radius) = best_circle(c)
        round_score = area / (np.pi * radius**2)

        # test de grossitude: fonction croissante de R+ dans [0,1] 
        # appliquée à l'aire de la zone détectée
        fat_score = 1 - np.exp(-area / 25) # aire caractéristique de 25px

        # score total: moyenne pondérée des 2 avec des coeffs arbitraires
        score = 0.7 * round_score + 0.3 * fat_score

        if score > best_score:
            best_score = score
            best_contour = c
    return best_contour

# retourne la position et le rayon de la balle sur l'image (en pixels)
# centre de l'écran de coordonnées (0,0), x vers la droite y vers le haut
def get_ball_position(img):
    #img = cv2.resize(img, (600,800))

    dst, mask = make_mask(img)

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_LIST, 2)
    c = choose_best_contour(contours)
    ((x, y), radius) = best_circle(c)

    return x - img.shape[1] / 2, img.shape[0] / 2 - y, radius
    # explication de ces expressions: trust me bro

"""-------------------------détection de la table-------------------------"""

# choisit les 2 contours qui ressemblent le plus à des triangles
# en fonction de la taille et de la trianglitude
def best_2triangles(contours):
    triangles = []
    scores = []
    for c in contours:
        # test de trianglitude: (aire de la zone détectée / aire de son triangle circonscrit)
        # score entre 0 et 1, 0=pas triangle, 1=très triangle
        area = cv2.contourArea(c)
        
        triangle = cv2.minEnclosingTriangle(c)[1][:,0].astype(int)
        tarea = cv2.contourArea(triangle)
        if tarea == 0: continue
        triangle_score = area / tarea

        # test de grossitude: fonction croissante de R+ dans [0,1] 
        # appliquée à l'aire de la zone détectée
        fat_score = 1 - np.exp(-area / 25) # aire caractéristique de 25px

        # score total: moyenne pondérée des 2 avec des coeffs arbitraires
        score = 0.5 * triangle_score + 0.5 * fat_score

        triangles.append(triangle)
        scores.append(score)

    #sorted_triangles = [t for s,t in sorted(zip(scores, triangles))]
    best1 = max(scores)
    b1i = scores.index(best1)
    scores[b1i] = 0
    best2 = max(scores)
    b2i = scores.index(best2)
    return triangles[b1i], triangles[b2i]

def barycentre(triangle):
    return sum([p for p in triangle]) / 3

def det(v1, v2):
    a, c = v1
    b, d = v2
    # a b
    # c d
    return a * d - b * c

# base bien: origine au centre de l'écran, x vers la droite, y vers le haut
# base opencv: origine en haut a gauche, x vers la droite, y vers le bas

# chgt de base opencv vers base bien
def chgtBaseCv2Bien(p, img):
    x, y = p
    return np.array([x - img.shape[1] / 2, img.shape[0] / 2 - y])

# chgt de base bien vers base opencv
def chgtBaseBien2Cv(p, img):
    x, y = p
    return np.array([x + img.shape[1] / 2, img.shape[0] / 2 - y])

def detect_table(img):
    _, maskg = make_mask(img, TRG_HSV) # green
    _, maskp = make_mask(img, TRP_HSV) # purple

    contoursg, hierarchy = cv2.findContours(maskg, cv2.RETR_LIST, 2)
    contoursp, hierarchy = cv2.findContours(maskp, cv2.RETR_LIST, 2)
    trg1, trg2 = best_2triangles(contoursg)
    trp1, trp2 = best_2triangles(contoursp)

    g1 = barycentre(trg1) # barycentre d'un triangle vert
    g2 = barycentre(trg2) # barycentre de l'autre triangle vert
    p1 = barycentre(trp1) # barycentre d'un triangle purple
    p2 = barycentre(trp2) # barycentre de l'autre triangle purple

    # chgt de base pour utiliser une base directe (la base utilisée par opencv est indirecte)
    g1 = chgtBaseCv2Bien(g1, img)
    g2 = chgtBaseCv2Bien(g2, img)
    p1 = chgtBaseCv2Bien(p1, img)
    p2 = chgtBaseCv2Bien(p2, img)

    # vue de dessus
    # 4--------------3
    # |              |
    # 1--------------2
    # 12 vert et 34 purple

    # guess initial: g1=1, g2=2 p1=3 p2=4
    # puis modifications in place pour que ça devienne vrai

    # étape 1: impose 43 dans le même sens que 12
    if np.dot(g2 - g1, p1 - p2) < 0: # 12.43 >= 0: 12 dans le même sens que 43 donc ce qu'on veut
        p1, p2 = p2, p1 # échange 34

    # étape 2: impose 123 tourne dans le sens trigo
    if det(g2 - g1, p1 - g2) < 0: # det(12, 23) >= 0: 123 tourne dans le sens trigo donc ce qu'on veut
        g1, g2 = g2, g1 # échange 12
        p1, p2 = p2, p1 # échange 34
    return g1, g2, p1, p2

# affiche des trucs pour montrer le résultat de la détection
def display_table(img, save_name=''):
    dstg, maskg = make_mask(img, TRG_HSV) # green
    dstp, maskp = make_mask(img, TRP_HSV) # purple
    cv2.imwrite('tables/masks/g' + save_name, maskg)
    cv2.imwrite('tables/dst/g' + save_name, dstg)
    cv2.imwrite('tables/masks/p' + save_name, maskp)
    cv2.imwrite('tables/dst/p' + save_name, dstp)
    g1, g2, p1, p2 = detect_table(img)
    g1 = chgtBaseBien2Cv(g1, img).astype(int)
    g2 = chgtBaseBien2Cv(g2, img).astype(int)
    p1 = chgtBaseBien2Cv(p1, img).astype(int)
    p2 = chgtBaseBien2Cv(p2, img).astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(img, tuple(g1), 5, (255, 255, 255))
    cv2.circle(img, tuple(g2), 5, (255, 255, 255))
    cv2.circle(img, tuple(p1), 5, (255, 255, 255))
    cv2.circle(img, tuple(p2), 5, (255, 255, 255))
    cv2.putText(img, '1', tuple(g1), font, 1, (0, 0, 255), 3)
    cv2.putText(img, '2', tuple(g2), font, 1, (0, 0, 255), 3)
    cv2.putText(img, '3', tuple(p1), font, 1, (0, 0, 255), 3)
    cv2.putText(img, '4', tuple(p2), font, 1, (0, 0, 255), 3)
    #cv2.imshow('pspps', img)
    #cv2.waitKey(0)
    print(save_name)
    if save_name:
        cv2.imwrite('tables/' + save_name, img)

"""-------------------------test pour entourer la balle sur toutes les images-------------------------"""

# ok déso pour le nom
# enregistre l'image "différence avec le orange", le masque, et l'image avec la balle entourée
def makedathing(path, name):
    img = cv2.imread(path + name + '.jpg')
    #img = cv2.resize(img, (600,800))

    dst, mask = make_mask(img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, 2)
    if not contours: print("ya rien")

    c = choose_best_contour(contours)

    ((x, y), radius) = best_circle(c)

    cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    cv2.imwrite(path + 'processed/' + name + '.jpg', img)
    cv2.imwrite(path + 'masks/' + name + '.jpg', mask)
    cv2.imwrite(path + 'dst/' + name + '.jpg', dst)

if __name__ == '__main__':
    for i in range(10):
        img = cv2.imread('tables/' + str(i) + '.jpg')
        display_table(img, 'detected_' + str(i) + '.jpg')
    #img = cv2.imread('faketable2.png')
    #display_table(img)
    #img = cv2.imread('faketable3.png')
    #display_table(img)
    #img = cv2.imread('impossibletable.png')
    #display_table(img)

"""
if __name__=='__main__':
    #path = 'data/0_14_50/'
    #names = ['0_2_0', '0_15_0', '0_30_0', '-15_2_15', '-15_15_15', '-15_30_15', '-30_2_-30', '-30_15_-30', '-30_30_-30']
    path = '14_03_2023/v2frames/'
    names = [str(i) for i in range(75)]
    for name in names:
        print(name)
        makedathing(path, name)
    #makedathing('C:/Users/42/Documents/TIPE/cameras/', 'trump')
"""