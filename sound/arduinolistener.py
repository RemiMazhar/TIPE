import serial
import time
import numpy as np
from scipy.io import wavfile
from datetime import datetime

"""on lit des valeurs envoyées en binaire par l'arduino"""

arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

N = 25000

values = np.zeros(N, dtype = int)
n = 0
started = False
while n < N:
    data = arduino.read(2)
    if data:
        if not started:
            started = True
            t0 = time.time()
        values[n] = int.from_bytes(data, "little")
        n+=1
        if n % 1000 == 0: print(n)
samplerate = int(N / (time.time() - t0))
print(samplerate)
#values -= np.average(values).astype(int)
values *= 32
now = datetime.now()
wavfile.write('measures/order/' + now.strftime("%Y_%m_%d_%H_%M") + '.wav', samplerate, values.astype(np.int16))