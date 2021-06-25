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
N = 10  # number of iterations
Nsize = 512

''' Initialize '''
# h=HexSimProcessor
h = HexSimProcessor()
h.debug = False
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

# tif.imwrite('test_reconfactor.tif',h._reconfactor)

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


h.cleanup = False
h.debug = False
h.NA = 1.1
h.magnification = 60
h.wavelength = 0.525
h.beta = 0.99
h.alpha = 0.3
h.n = 1.33

''' Calibration cupy'''
try:
    start_time = time.time()
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

print(f'spots phases: {h.p[0]}, {h.p[1]}, {h.p[2]}')
print(f'spots amplitudes: {h.ampl[0]}, {h.ampl[1]}, {h.ampl[2]}')

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
#
# start_time = time.time()
# imgoutb = h.batchreconstructcompact(img2)
# elapsed_time = time.time() - start_time
# print(f'Batch Reconstruction compact time (CPU): {elapsed_time:5f}s ')
#
# if isPlot:
#     plt.figure()
#     plt.imshow(imgoutb[20, :, :], cmap=cm.hot, clim=(0.0, 0.7 * imgoutb[20, :, :].max()))
#
# if isPlot:
#     plt.figure()
#     plt.imshow(imgoutb[20, :, :] - imgouta[20, :, :], cmap=cm.hot)

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

if isPlot:
    plt.figure()
    imgbsum = np.sum(imgbeads, 0)
    plt.imshow(imgbsum, cmap=cm.hot,clim=(imgbsum.min(), 0.25 * (imgbsum.max() - imgbsum.min()) + imgbsum.min()))

hb.a_type = 'none'
hb.calibrate(imgbeads)

try:
    imgb = hb.reconstruct_ocvU(imgbeads).get()
    if isPlot:
        plt.figure()
        plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.25 * imgb.max()))
        plt.title(hb.a_type)
except AssertionError as error:
    print(error)

hb.a_type = 'exp'
hb.calibrate(imgbeads)

try:
    imgb = hb.reconstruct_ocvU(imgbeads).get()
    if isPlot:
        plt.figure()
        plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.25 * imgb.max()))
        plt.title(hb.a_type)
except AssertionError as error:
    print(error)

hb.a_type = 'sph'
hb.debug = True
hb.calibrate(imgbeads)

try:
    imgb = hb.reconstruct_ocvU(imgbeads).get()
    print('Doing sph on bead image')
    if isPlot:
        print('Displaying sph on bead image')
        plt.figure(100)
        plt.imshow(imgb, cmap=cm.hot, clim=(0.0, 0.25 * imgb.max()))
        plt.title(hb.a_type)
except AssertionError as error:
    print(error)

""" Tests with OTF modifications"""
Nsize = 256

''' Initialize '''
# h=HexSimProcessor
h = HexSimProcessor()
h.debug = True
h.cleanup = False
h.magnification = 60
h.NA = 1.1
h.n = 1.33
h.wavelength = 0.520
h.eta = 0.7
h.beta = 0.995
h.alpha = 0.3
h.w = 1
h.a_type = 'none'
h.a = 0.25

h.N = (Nsize // 2) * 2

''' Read Image '''
data_folder = Path(os.path.dirname(__file__))
filename  = "./SIMdata_2019-11-05_15-21-42.tif"
filepath = os.path.join(data_folder, filename)
# print(data_folder)
# quit()
# filename = str(data_folder / "SIMdata_2019-11-05_15-21-42/SIMdata_2019-11-05_15-21-42.tif")
# filename = "./SIMdata_2019-11-05_15-21-42.tif"
# img1 = tif.imread('/Users/maan/Documents/MATLAB/HexSIMulator/S_1.252021_0209_1828_Raw_Image.tif')
img1 = tif.imread('Tests/Sara_2021_0218_1641_Raw_Image.tif')
# img1 = tif.imread('/Users/maan/Downloads/2021_0408_1454_561nm_Raw.tif')
# img1 = tif.imread('/Users/maan/Downloads/2021_0408_1454_561nm_Raw1_0408_1454_561nm_Raw202105021955_segmented_raw_002.tif')

# if Nsize != 512:
#     img1 = np.single(img1[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
# else:
img1 = np.single(img1)
print(np.sum(img1,(1,2)))
if isPlot:
    plt.figure()
    plt.imshow(np.sum(img1, 0), cmap=cm.hot, interpolation="bicubic")
    plt.figure()
    plt.title('input')
    plt.hist(img1.flatten(), bins='auto', log=True)
    plt.figure()
    plt.imshow(np.std(img1,0), cmap=cm.hot, interpolation="bicubic")

''' Calibration '''
start_time = time.time()
h.calibrate(img1)
elapsed_time = time.time() - start_time
print(f'Calibration time: {elapsed_time:5f}s ')
print(h.N)
''' Recontruction '''

''' FFTW '''
start_time = time.time()
imga = h.reconstruct_rfftw(img1)
elapsed_time = time.time() - start_time
print(f'Batch Reconstruction time: {elapsed_time:5f}s ')

if isPlot:
    plt.figure()
    plt.title('none')
    plt.imshow(imga , cmap=cm.hot, clim=(0.1 * imga.max(), 0.7 * imga.max()), interpolation="bicubic")
    plt.figure()
    plt.title('none')
    plt.hist(imga.flatten(), bins='auto', log=True)

h.debug = False
h.a_type = 'exp'
h.calibrate(img1)
imga = h.reconstruct_rfftw(img1)
if isPlot:
    plt.figure()
    plt.title('exp')
    plt.imshow(imga, cmap=cm.hot, clim=(0.1 * imga.max(), 0.7 * imga.max()), interpolation="bicubic")
    plt.figure()
    plt.title('exp')
    plt.hist(imga.flatten(), bins='auto', log=True)

h.a_type = 'sph'
h.calibrate(img1)
imga = h.reconstruct_rfftw(img1)
if isPlot:
    plt.figure()
    plt.title('sph')
    plt.imshow(imga, cmap=cm.hot, clim=(0.1 * imga.max(), 0.7 * imga.max()), interpolation="bicubic")
    plt.figure()
    plt.title('sph')
    plt.hist(imga.flatten(), bins='auto', log=True)

from SIM_processing import hexSimProcessor
hexSimProcessor.opencv = True

import cProfile
profile = cProfile.Profile()
profile.enable()
h.calibrate(img2)
# h.calibrate_cupy(img2) # To test cupy processing
profile.disable()
profile.dump_stats('hexsim.prof')
# Use "snakeviz hexsim.prof" in terminal window for graphical view of results

try:
    import line_profiler
    lprofile = line_profiler.LineProfiler()
    lprofile.add_function(HexSimProcessor._tfm)
    wrapper = lprofile(h._calibrate)
    wrapper(img2)
    # wrapper(img2, useCupy = True) # To test cupy processing
    lprofile.disable()
    lprofile.print_stats(output_unit=1e-3)
except:
    print('no line_profiler')

plt.show()
