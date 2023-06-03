from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

"""
--->|--------
       |    | 
       C    R    
       |    |    
-------------

où >| est une diode

appelé entre autres peak detector, détecteur de crête, détecteur d'enveloppe
"""

c = 4.7e-6
r = 2000
# but: r * c >> T période du signal

def envelope(Es, dt):
    u = 0
    du = 0
    us = np.zeros(Es.shape[0])
    for i, E in enumerate(Es):
        if E - u > 0: # diode passante: charge instantanée
            u = E
            du = 0
        else: # diode bloquante: décharge
            # u' + rc * u = 0
            du = -u / (r * c)
            u += du * dt
        us[i] = u
    return us

if __name__ == '__main__':
    samplerate, data = wavfile.read('close800cali_usable.wav')
    data = data[:20000]
    dt = 1 / samplerate
    ts = np.arange(data.shape[0]) * dt
    us = envelope(data, dt)
    us = us.astype(np.int16)
    #wavfile.write('envelope.wav', samplerate, us)
    plt.plot(ts, data)
    plt.plot(ts, us)
    plt.show()

    #Us = bpfilter(data, dt)
    #Us = Us.astype(np.int16)
    #wavfile.write('filtered.wav', samplerate, Us)