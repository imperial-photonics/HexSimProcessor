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
isPlot = False
N = 10  # number of iterations
Nsize = 512

''' Initialize '''
# h=HexSimProcessor
h = HexSimProcessor()
h.cleanup = True
h.N = (Nsize // 2) * 2
h.NA = 0.75
h.magnification = 40
h.wavelength = 0.560
h.beta = 0.99
h.n = 1.0
h.debug = False

''' Read Image '''
data_folder = Path(os.path.dirname(__file__))
filename  = "./SIMdata_2019-11-05_15-21-42.tif"
filepath = os.path.join(data_folder, filename)
# print(data_folder)
# quit()
# filename = str(data_folder / "SIMdata_2019-11-05_15-21-42/SIMdata_2019-11-05_15-21-42.tif")
# filename = "./SIMdata_2019-11-05_15-21-42.tif"
img1 = tif.imread(filepath)

if Nsize != 512:
    img1 = np.single(img1[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
else:
    img1 = np.single(img1)

if isPlot:
    plt.figure()
    plt.imshow(np.sum(img1, 0), cmap=cm.gray)

''' Calibration Cupy'''
try:
    h.calibrate_cupy(img1)
    start_time = time.time()
    h.calibrate_cupy(img1)
    elapsed_time = time.time() - start_time
    print(f'Fast Calibration time: {elapsed_time:5f}s ')
except AssertionError as error:
    print(error)

''' Calibration '''
start_time = time.time()
h.calibrate(img1)
elapsed_time = time.time() - start_time
print(f'Calibration time: {elapsed_time:5f}s ')

''' Recontruction '''

''' FFTW '''
start_time = time.time()
for i in range(0, 10):
    imga = h.reconstruct_fftw(img1)
elapsed_time = time.time() - start_time
print(f'FFTW Reconstruction time: {elapsed_time / 10:5f}s ')

''' rFFTW '''
start_time = time.time()
for i in range(0, N):
    imga = h.reconstruct_rfftw(img1)
elapsed_time = time.time() - start_time
print(f'rFFTW Reconstruction time: {elapsed_time / N:5f}s ')
if isPlot:
    plt.figure()
    plt.imshow(imga, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))

''' ocv '''
try:
    start_time = time.time()
    for i in range(0, N):
        imgb = h.reconstruct_ocv(img1)
    elapsed_time = time.time() - start_time
    print(f'ocv Reconstruction time: {elapsed_time / N:5f}s ')
    if isPlot:
        plt.figure()
    plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))
except AssertionError as error:
    print(error)

''' ocvU '''
try:
    start_time = time.time()
    for i in range(0, N):
        imgb = h.reconstruct_ocvU(img1)
    elapsed_time = time.time() - start_time
    print(f'ocvU Reconstruction time: {elapsed_time / N:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgb.get(), cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))
except AssertionError as error:
    print(error)

''' CuPy '''
try:
    start_time = time.time()
    for i in range(0, N):
        imgb = h.reconstruct_cupy(img1)
    elapsed_time = time.time() - start_time
    print(f'CuPy Reconstruction time: {elapsed_time / N:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgb, cmap=cm.gray)
except AssertionError as error:
    print(error)

''' FFTW '''
start_time = time.time()
for i in range(0, 7 * N):
    imga = h.reconstructframe_fftw(img1[i % 7, :, :], i % 7)
elapsed_time = time.time() - start_time
print(f'FFTW Reconstructframe time: {elapsed_time / (7 * N):5f}s ')
if isPlot:
    plt.figure()
    plt.imshow(imga, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))

''' rFFTW '''
start_time = time.time()
for i in range(0, 7 * N):
    imga = h.reconstructframe_rfftw(img1[i % 7, :, :], i % 7)
elapsed_time = time.time() - start_time
print(f'rFFTW Reconstructframe time: {elapsed_time / (7 * N):5f}s ')
if isPlot:
    plt.figure()
    plt.imshow(imga, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))

''' ocv '''
try:
    start_time = time.time()
    for i in range(0, 7 * N):
        imga = h.reconstructframe_ocv(img1[i % 7, :, :], i % 7)
    elapsed_time = time.time() - start_time
    print(f'ocv Reconstruct frame time: {elapsed_time / (7 * N):5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imga, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))
except AssertionError as error:
    print(error)

''' ocvU '''
try:
    start_time = time.time()
    for i in range(0, 7 * N):
        imga = h.reconstructframe_ocvU(img1[i % 7, :, :], i % 7)
    elapsed_time = time.time() - start_time
    print(f'ocvU Reconstruct frame time: {elapsed_time / (7 * N):5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imga.get(), cmap=cm.hot, clim=(0.0, 0.7 * imga.get().max()))
except AssertionError as error:
    print(error)

''' CuPy '''
try:
    start_time = time.time()
    for i in range(0, 7 * N):
        imgb = h.reconstructframe_cupy(img1[i % 7, :, :], i % 7)
    elapsed_time = time.time() - start_time
    print(f'CuPy Reconstructframe time: {elapsed_time / (7 * N):5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.7 * imgb.max()))
except AssertionError as error:
    print(error)

''' Read image stack'''
data_folder = Path(os.path.dirname(__file__))
filename = "./Raw_img_stack_512_inplane.tif"
filepath = os.path.join(data_folder, filename)

img2 = tif.imread(filepath)
if Nsize != 512:
    img2 = np.single(img2[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
else:
    img2 = np.single(img2)

start_time = time.time()
h.cleanup = False
h.debug = False

''' Calibration cupy'''
try:
    h.calibrate(img2[140:147, :, :])
    elapsed_time = time.time() - start_time
    print(f'Calibration time: {elapsed_time:5f}s ')
    start_time = time.time()
    h.calibrate_cupy(img2[140:147, :, :])
    elapsed_time = time.time() - start_time
    print(f'Cupy Calibration time: {elapsed_time:5f}s ')
except AssertionError as error:
    print(error)
    start_time = time.time()
    # cProfile.run("h.calibrate(img2[140:147, :, :])", sort = 'tottime')
    h.calibrate(img2[140:147, :, :])
    elapsed_time = time.time() - start_time
    print(f'Calibration time: {elapsed_time:5f}s ')

imga = h.reconstruct_rfftw(img2[140:147, :, :])
if isPlot:
    plt.figure()
    plt.imshow(imga, cmap=cm.hot, clim=(0.0, 0.7 * imga.max()))

start_time = time.time()
imgouta = h.batchreconstruct(img2)
elapsed_time = time.time() - start_time
print(f'Batch Reconstruction time (CPU): {elapsed_time:5f}s ')

if isPlot:
    plt.figure()
    plt.imshow(imgouta[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgouta[20, :, :].max()))

start_time = time.time()
imgoutb = h.batchreconstructcompact(img2)
elapsed_time = time.time() - start_time
print(f'Batch Reconstruction compact time (CPU): {elapsed_time:5f}s ')

if isPlot:
    plt.figure()
    plt.imshow(imgoutb[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgoutb[20, :, :].max()))

if isPlot:
    plt.figure()
    plt.imshow(imgoutb[20, :, :] - imgouta[20, :, :], cmap=cm.hot)

''' Batch process GPU compact'''
try:
    start_time = time.time()
    imgout = h.batchreconstructcompact_cupy(img2)
    elapsed_time = time.time() - start_time
    print(f'Batch Reconstruction compact time(CuPy): {elapsed_time:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgout[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgout[20, :, :].max()))
    if isPlot:
        plt.figure()
        plt.imshow(imgout[20, :, :] - imgouta[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgout[20, :, :].max()))
except AssertionError as error:
    print(error)

''' Batch process GPU '''
try:
    start_time = time.time()
    imgout = h.batchreconstruct_cupy(img2)
    elapsed_time = time.time() - start_time
    print(f'Batch Reconstruction time(CuPy): {elapsed_time:5f}s ')
    if isPlot:
        plt.figure()
        plt.imshow(imgout[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgout[20, :, :].max()))
except AssertionError as error:
    print(error)

''' Beads test '''
data_folder = Path(os.path.dirname(__file__))
filename = "SIMdata_2019-11-05_15-45-12.tif"
filepath = os.path.join(data_folder, filename)
imgbeads = np.single(tif.imread(filepath))

hb = HexSimProcessor()
hb.N = 512
hb.magnification = 40
hb.NA = 0.75
hb.wavelength = 0.560
hb.n = 1.0
hb.eta = 0.7
hb.beta = 0.999
hb.alpha = 0.3
hb.w = 0.3
hb.debug = False
hb.cleanup = False

hb.calibrate(imgbeads)

try:
    imgb = hb.reconstruct_ocvU(imgbeads).get()
    if isPlot:
        plt.figure()
        plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.25 * imgb.max()))
except AssertionError as error:
    print(error)

plt.show()
