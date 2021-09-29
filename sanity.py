import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


y, samplerate = sf.read('data/cheetah/ENGINE_IDLE.wav')

freq_magnitudes = np.abs(np.fft.fft(y))
freq_values = np.fft.fftfreq(len(freq_magnitudes), 1/samplerate)

n = 150

plt.plot(freq_values[:n] * 60, freq_magnitudes[:n])
plt.show()
