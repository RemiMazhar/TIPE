import numpy as np
from scipy.io import wavfile
import scipy.interpolate as inter
"""
---R1-----|>--C2---
        |         |
        C1        R2
        |         |
-------------------

où |> est un suiveur
"""

r1 = 330
r2 = 220
c1 = 1e-6
c2 = 0.47e-6

# but: w0 ~= 5027 (f0=800Hz)

K = 1 / (1 + r1 * c1 / (r2 * c2))
Q = np.sqrt(r1 * c1 / (r2 * c2))
w0 = 1/np.sqrt(r1 * c1 * r2 * c2)
wb = 1 / (r1 * c1)
wh = 1 / (r2 * c2)

# H = K / (1 + jQ(w/w0 - w0/w))
# U'' + w0/Q * U' + w0^2 * U = Kw0/Q * E'
# On pose V' = U
# V'' + w0/Q * V' + w0^2 * V = Kw0/Q * E

# équa diff du bandpass filter
# donne V(t+dt) et V'(t+dt) en fonction de E(t), V(t) et V''(t)
def bpED(E, v, dv ,dt):
    ddv = K * w0 / Q * E - w0**2 * v - w0 / Q * dv
    return (v + dv * dt, dv + ddv * dt)

def bpfilter(Es, dt):
    v = 0
    dv = 0
    us = np.zeros(Es.shape[0])
    for i, E in enumerate(Es):
        v, dv = bpED(E, v, dv, dt)
        us[i] = dv
    return us

if __name__ == '__main__':
    print('K={}, Q={}, w0={}s^-1, wb={}s^-1, wh={}s^-1'.format(K, Q, w0, wb, wh))
    #samplerate, data = wavfile.read('close800cali_usable.wav')
    samplerate, data = wavfile.read('measures/order/2023_05_29_18_06.wav')
    dt = 1 / samplerate
    ts = np.arange(data.shape[0]) * dt
    f = inter.interp1d(ts, data)
    ts2 = np.linspace(ts[0], ts[-1], int(40000 * ts[-1]))
    data2 = f(ts2)
    dt2 = 1 / 40000
    #data += 6000
    #wavfile.write('modified.wav', samplerate, data)
    Us = bpfilter(data2, dt2)
    Us = Us.astype(np.int16)
    wavfile.write('measures/upscaled.wav', 40000, data2.astype(np.int16))
    wavfile.write('measures/upscaledfiltered.wav', 40000, Us)


"""
R / (R + 1/jwC) = jwRC / (1 + jwRC)
1/jwC / (R + 1/jwC) = 1 / (1 + jwRC)
H = jwRC / (1 + jwR1C1)(1+jwR2C2)
= 1 / (1 + jwR1C1)(1 + 1/jwR2C2)
= 1 / (1 + R1C1/R2C2 + j(wR1C1 - 1/wR2C2))
K = 1 / (1 + R1C1/R2C2)
Q/w0 = R1C1
Qw0 = 1/R2C2

K = 1 / (1 + R1C1/R2C2)
Q = sqrt(R1C1/R2C2)
w0 = 1/sqrt(R1C1R2C2)

R1C1R2C2 = 1/w0^2
R1C1/R2C2 = Q^2
Q = 1
R1C1 = 1/w0
R2C2 = 1/w0
K = 1/2

"""