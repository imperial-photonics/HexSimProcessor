import os

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import time

# import cProfile
from SIM_processing.hexSimProcessor import HexSimProcessor

plt.close('all')
isPlot = True
N = 1  # number of iterations
Nsize = 512

''' Initialize '''
# h=HexSimProcessor
h = HexSimProcessor()
h.debug = True
h.cleanup = False
h.magnification = 63
# h.magnification = 40
h.NA = 0.75
h.n = 1
h.pixelsize = 5.85
h.wavelength = 0.6
h.eta = 0.35
# h.eta = 0.70
h.alpha = 0.1
h.beta = 0.99
h.w = 0.3

h.N = (Nsize // 2) * 2

''' Read Image '''
img = tif.imread('/Users/maan/OneDrive - Imperial College London/Prochip/Polimi/63X_075.tif')
# img = tif.imread('/Users/maan/Documents/Office Projects/Prochip/HexSimProcessor/Tests/SIMdata_2019-11-05_15-21-42.tif')
print(img.shape)

if isPlot:
    plt.figure()
    plt.imshow(np.sum(img, 0), cmap=cm.gray, clim=(0.0, 0.7 * np.sum(img, 0).max()))

''' Calibration '''
start_time = time.time()
h.calibrate(img)
elapsed_time = time.time() - start_time
print(f'Calibration time: {elapsed_time:5f}s ')

''' Recontruction '''
''' FFTW '''
start_time = time.time()
imga = h.reconstruct_rfftw(img)
elapsed_time = time.time() - start_time
print(f'Reconstruction time: {elapsed_time:5f}s ')
if isPlot:
    plt.figure()
    plt.imshow(imga , cmap=cm.gray, clim=(0.0, 0.7 * imga.max()))

tif.imwrite('/Users/maan/temp/img.tif',sum(img,0))
tif.imwrite('/Users/maan/temp/imga.tif',imga)

'''Check phase shifts'''
h.debug = True
phase0, ampl0 = h.find_phase(h.kx[0], h.ky[0], img)
expected_phase = np.arange(7) * 2 * np.pi / 7
phase0 = np.unwrap(phase0 - expected_phase) + expected_phase - phase0[0]
plt.figure(20)
plt.plot(phase0, 'bx-')
plt.plot(expected_phase, 'b--')

phase1, ampl1 = h.find_phase(h.kx[1], h.ky[1], img)
expected_phase = np.arange(7) * 4 * np.pi / 7
phase1 = np.unwrap(phase1 - expected_phase) + expected_phase - phase1[0]
plt.figure(20)
plt.plot(phase1, 'gx-')
plt.plot(expected_phase, 'g--')

phase2, ampl2 = h.find_phase(h.kx[2], h.ky[2], img)
expected_phase = np.arange(7) * 6 * np.pi / 7
phase2 = np.unwrap(phase2 - expected_phase) + expected_phase - phase2[0]
plt.figure(20)
plt.plot(phase2, 'rx-')
plt.plot(expected_phase, 'r--')


plt.show()
