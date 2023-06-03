from object_detection import get_ball_position, makedathing
import cv2
import matplotlib.pyplot as plt
import numpy as np
from geometry import Camera
from locationOOP import locate
from scipy.ndimage import gaussian_filter1d
from locate_camera import Calibrator, get_camera
from fitparabola import reg_poly2_experimental

"""---------cette partie fait sens---------"""
# position en fonction du temps à partir de n vidéos sachant d'où elles ont été prises
def get_trajectory(data, nFrames=20, fps=120):
    # data: liste de tuples (video, cam)
    pts = np.zeros((nFrames, 3))
    ts = np.arange(nFrames) / fps
    for i in range(nFrames):
        print('frame ' + str(i) + '     ', end='\r')
        imgs_data = [(vid.read()[1], cam) for vid, cam in data]
        pts[i] = locate(imgs_data)
    return ts, pts

"""---------celle-ci est moins claire mais je crois que c'est juste pour afficher la balle entourée sur chaque frame de la vidéo---------"""
def makedavideo(vid, path, nb_frames):
    out = cv2.VideoWriter(path + 'processed.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (450, 800))
    for i in range(nb_frames):
        frame = vid.read()[1]
        cv2.imwrite(path + str(i) + '.jpg', frame)
        makedathing(path, str(i))
        out.write(cv2.imread(path + 'processed/' + str(i) + '.jpg'))
    out.release()


"""---------et là des expériences à la con pour afficher les trajectoires/calculer g---------"""
def draw_trajectory(pts):
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    plt.subplot(131)
    plt.plot(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.gca()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    plt.subplot(132)
    plt.plot(zs, ys)
    plt.xlabel('z')
    plt.ylabel('y')
    ax = plt.gca()
    ax.set_xlim(50, -50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    plt.subplot(133)
    plt.plot(xs, zs)
    plt.xlabel('x')
    plt.ylabel('z')
    ax = plt.gca()
    ax.set_xlim(-50, 50)
    ax.set_ylim(50, -50)
    ax.set_aspect('equal')

    plt.show()

def draw_3dft(ts, pts):
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    plt.subplot(131)
    plt.plot(ts, xs)
    plt.xlabel('t')
    plt.ylabel('x')
    ax = plt.gca()
    ax.set_ylim(-50, 50)
    plt.subplot(132)
    plt.plot(ts, ys)
    plt.xlabel('t')
    plt.ylabel('y')
    ax = plt.gca()
    ax.set_ylim(-50, 50)
    plt.subplot(133)
    plt.plot(ts, zs)
    plt.xlabel('t')
    plt.ylabel('z')
    ax = plt.gca()
    ax.set_ylim(50, -50)

    plt.show()

def smoothify(ts, traj):
    traj[:, 0] = gaussian_filter1d(traj[:, 0], 1)
    traj[:, 1] = gaussian_filter1d(traj[:, 1], 1)
    traj[:, 2] = gaussian_filter1d(traj[:, 2], 1)
    return ts, traj

def deriv(ts, pts):
    der = np.zeros_like(pts)
    der[0] = (pts[1] - pts[0]) / ts[1]
    der[-1] = (pts[-1] - pts[-2]) / (ts[-1] - ts[-2])
    for i in range(1, ts.shape[0] - 1):
        dt = ts[i + 1] - ts[i - 1]
        dx = pts[i + 1] - pts[i - 1]
        der[i] = dx / dt
    return der

def calculate_g(ts, pts):
    ys = pts[:, 1]
    ys = gaussian_filter1d(ys, 0.5)
    plt.plot(ts, ys)
    plt.show()
    dys = deriv(ts, ys)
    dys = gaussian_filter1d(dys, 1)
    plt.plot(ts, dys)
    plt.show()
    ddys = deriv(ts, dys)
    ddys = gaussian_filter1d(ddys, 1)
    plt.plot(ts, ddys)
    plt.show()
    m = np.median(ddys)
    print(m)
    useful = [d for d in ddys if 0.5 < abs(d / m) < 2]
    plt.plot(useful)
    plt.show()
    print(np.average(useful))

def calculate_g_better(ts, pts):
    a,b,c = reg_poly2_experimental(ts, pts[:,1]) # y = at^2 + bt + c
    g = -2 * a
    vy0 = b
    z0 = c
    print('g=%.3f,  v_y0=%.3f, z0=%.3f' % (g, vy0, z0))
    ys_predict = a * ts**2 + b * ts + c
    plt.scatter(ts, pts[:,1])
    plt.plot(ts, ys_predict)
    plt.show()

# honnêtement flemme d'essayer de comprendre à partir d'ici
if __name__ == '__main__':
    v1 = cv2.VideoCapture('21_03_2023/v1_processed.mp4')
    v2 = cv2.VideoCapture('21_03_2023/v2_processed.mp4')
    #makedavideo(v1, '21_03_2023/v1frames/', int(v1.get(cv2.CAP_PROP_FRAME_COUNT)))
    #makedavideo(v2, '21_03_2023/v2frames/', int(v2.get(cv2.CAP_PROP_FRAME_COUNT)))

    cali = Calibrator()
    real_pos = [
        [0,0,50],
        [0,0,0],
        [-50,0,0],
        [-30,0,-30]
    ]

    """
    [-20.  58. -90.]
    [[-0.99680463 -0.06004365  0.05268104]
    [-0.01929757  0.82099977  0.5706023 ]
    [-0.07751217  0.5677624  -0.81953507]]
    """
    img1 = cv2.imread('21_03_2023/v1frames/0.jpg')
    for i in range(10):
        camdata1 = cali.get_data(img1, real_pos)
        cam1 = get_camera(camdata1)
        print(camdata1)
        print(np.round(cam1.pos))
        print(np.round(cam1.orientation, 2))

    """
    [107.  94.  13.]
    [[ 0.0665528  -0.389916S08  0.91844225]
    [-0.00122925  0.92045031  0.39085766]
    [-0.99778215 -0.02714166  0.06077924]]
    """
    img2 = cv2.imread('21_03_2023/v2frames/0.jpg')
    camdata2 = cali.get_data(img2, real_pos)
    cam2 = get_camera(camdata2)
    print(np.round(cam2.pos))
    print(np.round(cam2.orientation, 2))

    data = [(v1, cam1), (v2, cam2)]
    nframes = min(int(v1.get(cv2.CAP_PROP_FRAME_COUNT)), int(v2.get(cv2.CAP_PROP_FRAME_COUNT)))
    ts, traj = get_trajectory([(v1, cam1), (v2, cam2)], nframes)
    draw_trajectory(traj)
    draw_3dft(ts, traj)
    calculate_g_better(ts, traj)

    """
    RATIO = -0.0016

    Oc1 = np.array([0, 14, 50], dtype=float)
    Bc1ToB0 = np.identity(3, dtype=float)
    cam2 = Camera(Oc1, Bc1ToB0, RATIO)

    Oc2 = np.array([50, 14, 0], dtype=float)
    Bc2ToB0 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ], dtype=float)
    cam1 = Camera(Oc2, Bc2ToB0, RATIO)

    v1 = cv2.VideoCapture('C:/Users/42/Documents/TIPE/cameras/14_03_2023/v1.mp4')
    v2 = cv2.VideoCapture('C:/Users/42/Documents/TIPE/cameras/14_03_2023/v2.mp4')

    ts, traj = get_trajectory([(v1, cam1), (v2, cam2)], 75)
    #calculate_g(ts, traj)

    draw_trajectory(traj)
    draw_3dft(ts, traj)

    #path = 'C:/Users/42/Documents/TIPE/cameras/14_03_2023/v2frames/'
    #makedavideo(v2, path, 75)
    """