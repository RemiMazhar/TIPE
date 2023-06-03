from bandpass import bpfilter
from peakdetector import envelope
from scipy.io import wavfile
import numpy as np

samplerate, data = wavfile.read('close800cali_usable.wav')
dt = 1 / samplerate
ts = np.arange(data.shape[0]) * dt

raw_envelope = envelope(data, dt).astype(np.int16)
filtered = bpfilter(data, dt).astype(np.int16)
filtered_envelope = envelope(filtered, dt).astype(np.int16)

wavfile.write('raw_envelope.wav', samplerate, raw_envelope)
wavfile.write('filtered_envelope.wav', samplerate, filtered_envelope)
wavfile.write('filtered_signal.wav', samplerate, filtered)