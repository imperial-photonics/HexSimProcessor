import os

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import time

# import cProfile
from SIM_processing.hexSimProcessor import HexSimProcessor
from SIM_processing.simProcessor import SimProcessor

plt.close('all')
isPlot = False
N = 10  # number of iterations
Nsize = 512

''' Read image stack'''
data_folder = Path(os.path.dirname(__file__))
filename = "./Raw_img_stack_512_inplane.tif"
filepath = os.path.join(data_folder, filename)

img2 = tif.imread(filepath)
if Nsize != 512:
    img2 = np.single(img2[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
else:
    img2 = np.single(img2)

''' Beads test '''
data_folder = Path(os.path.dirname(__file__))
filename = "SIMdata_2019-11-05_15-45-12.tif"
filepath = os.path.join(data_folder, filename)
img = np.single(tif.imread(filepath))

def hexTo2Beam(imgin, num):
    M = [[2.33333, 1.58031, -0.111709, -1.4686, -1.4686, -0.111709, 1.58031],
         [-0.666667, -1.64433, -1.13277, 0.482794, 1.98581, 2.24448, 1.06402],
         [-0.666667, 1.06402, 2.24448, 1.98581, 0.482794, -1.13277, -1.64433],
         [2.33333, -0.111709, -1.4686, 1.58031, 1.58031, -1.4686, -0.111709],
         [-0.666667, -1.13277, 1.98581, 1.06402, -1.64433, 0.482794, 2.24448],
         [-0.666667, 2.24448, 0.482794, -1.64433, 1.06402, 1.98581, -1.13277],
         [2.33333, -1.4686, 1.58031, -0.111709, -0.111709, 1.58031, -1.4686],
         [-0.666667, 0.482794, 1.06402, -1.13277, 2.24448, -1.64433, 1.98581],
         [-0.666667, 1.98581, -1.64433, 2.24448, -1.13277, 1.06402, 0.482794]]
    N = imgin.shape[1]
    imgout = np.zeros((3, N, N), np.single)
    for i in range(7):
        imgout[0, :, :] = imgout[0, :, :] + M[num * 3][i] * imgin[i, :, :]
        imgout[1, :, :] = imgout[1, :, :] + M[num * 3 + 1][i] * imgin[i, :, :]
        imgout[2, :, :] = imgout[2, :, :] + M[num * 3 + 2][i] * imgin[i, :, :]
    return imgout

h2 = SimProcessor()
h2.debug = False
h2.cleanup = True
h2.N = (Nsize // 2) * 2
h2.NA = 1.1
h2.magnification = 60
h2.wavelength = 0.525
h2.beta = 0.99
h2.alpha = 0.3
h2.n = 1.33
h2.debug = True
isPlot = True

img1x2 = hexTo2Beam(img2[140:147, :, :],0)
img1x2a = hexTo2Beam(img2[161:168, :, :],0)
h2.calibrate(img1x2)
h2.debug = False

img1x2o = h2.reconstruct_fftw(img1x2)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))
    # plt.imshow(img1x2o[256:768,256:768], cmap=cm.gray)

for i in range(3):
    img1x2o = h2.reconstructframe_fftw(img1x2a[i , :, :], i)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

img1x2o = h2.reconstruct_rfftw(img1x2)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

for i in range(3):
    img1x2o = h2.reconstructframe_rfftw(img1x2a[i , :, :], i)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

img1x2o = h2.reconstruct_ocv(img1x2)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

for i in range(3):
    img1x2o = h2.reconstructframe_ocv(img1x2a[i , :, :], i)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

img1x2o = h2.reconstruct_ocvU(img1x2).get()
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

for i in range(3):
    img1x2o = h2.reconstructframe_ocvU(img1x2a[i , :, :], i).get()
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

img1x2 = hexTo2Beam(img2[140:147, :, :],1)
h2.calibrate(img1x2)

img1x2o = h2.reconstruct_fftw(img1x2)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

img1x2 = hexTo2Beam(img2[140:147, :, :],2)
try:
    h2.calibrate(img1x2)
except AssertionError as error:
    print(error)

img1x2o = h2.reconstruct_fftw(img1x2)
if isPlot:
    plt.figure()
    plt.imshow(img1x2o, cmap=cm.hot, clim=(0.0, 0.7 * img1x2o.max()))

imgstack = np.zeros((120, Nsize, Nsize), dtype = np.single)

for i in range (40):
    imgstack[3*i:3*i+3, :, :] = hexTo2Beam(img2[7*i:7*i+7, :, :],0)

h2.debug = False

try:
    start_time = time.time()
    h2.calibrate(imgstack)
    elapsed_time = time.time() - start_time
    print(f'Calibrate time: {elapsed_time:5f}s ')
    start_time = time.time()
    imgo = h2.batchreconstruct(imgstack)
    elapsed_time = time.time() - start_time
    print(f'Batchreconstruct time: {elapsed_time:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgo[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgo[20, :, :].max()))
except AssertionError as error:
    print(error)

try:
    h2.calibrate_cupy(imgstack)
    start_time = time.time()
    h2.calibrate_cupy(imgstack)
    elapsed_time = time.time() - start_time
    print(f'Calibrate cupy time: {elapsed_time:5f}s ')
    start_time = time.time()
    imgo = h2.batchreconstruct_cupy(imgstack)
    elapsed_time = time.time() - start_time
    print(f'Batchreconstruct cupy time: {elapsed_time:5f}s ')
    start_time = time.time()
    imgo = h2.batchreconstruct_cupy(imgstack)
    elapsed_time = time.time() - start_time
    print(f'Batchreconstruct cupy time: {elapsed_time:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgo[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgo[20, :, :].max()))
except AssertionError as error:
    print(error)

from SIM_processing import simProcessor
simProcessor.opencv = True # turn off opencv processing and calculation of lookuptables

import cProfile
profile = cProfile.Profile()
profile.enable()
# h2.calibrate(imgstack)
h2.calibrate_cupy(imgstack) # To test cupy processing
profile.disable()
profile.dump_stats('2beamsim.prof')
# Use "snakeviz 2beamsim.prof" in terminal window for graphical view of results

try:
    import line_profiler
    lprofile = line_profiler.LineProfiler()
    wrapper = lprofile(h2._calibrate)
    # wrapper(imgstack, useCupy = False)
    wrapper(imgstack, useCupy = True) # To test cupy processing
    lprofile.disable()
    lprofile.print_stats(output_unit=1e-3)
except:
    print('no line_profiler')

plt.show()

