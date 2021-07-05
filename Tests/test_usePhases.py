import os

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import time

# import cProfile
from SIM_processing.simProcessor import SimProcessor

plt.close('all')
isPlot = True
N = 1  # number of iterations
Nsize = 200

''' Initialize '''
# h=HexSimProcessor
h = SimProcessor()
h.debug = True
h.cleanup = False
h.magnification = 60
h.NA = 1.1
h.n = 1.46
h.wavelength = 0.561
h.eta = 0.2
h.alpha = 0.1

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
# img1 = tif.imread('/Users/maan/Imperial College London/Gong, Hai - measurement/Archive measurements/Sara_2021_0218_1641_Raw_Image.tif')
# img1 = tif.imread('/Users/maan/Downloads/2021_0408_1454_561nm_Raw.tif')
# img1 = tif.imread('/Users/maan/Downloads/2021_0408_1454_561nm_Raw1_0408_1454_561nm_Raw202105021955_segmented_raw_002.tif')
img1 = tif.imread('/Users/maan/Downloads/slices_503_to_demodulate.tif')

# if Nsize != 512:
#     img1 = np.single(img1[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
# else:
img1 = np.single(img1)
print(img1.shape)
img = img1.transpose((1,0,2,3)).reshape((1509,200,200))
print(img.shape)

if isPlot:
    plt.figure()
    plt.imshow(np.sum(img, 0), cmap=cm.gray)

''' Calibration Cupy'''
try:
    start_time = time.time()
    h.calibrate_cupy(img)
    elapsed_time = time.time() - start_time
    print(f'Fast Calibration time: {elapsed_time:5f}s ')
except AssertionError as error:
    print(error)

''' Calibration '''
start_time = time.time()
h.usePhases = False
h.calibrate(img[540:549,:,:])
elapsed_time = time.time() - start_time
print(f'Calibration time: {elapsed_time:5f}s ')
print(h.N)
''' Recontruction '''

''' FFTW '''
start_time = time.time()
imga = h.batchreconstruct(img)
elapsed_time = time.time() - start_time
print(f'Batch Reconstruction time: {elapsed_time:5f}s ')
h.usePhases = True
h.calibrate(img[540:549,:,:])
imgc = h.batchreconstruct(img)

if isPlot:
    plt.figure()
    imgasum = np.sum(imga,0)
    plt.imshow(imgasum , cmap=cm.hot, clim=(0.0, 0.7 * imgasum.max()))
    plt.figure()
    imgcsum = np.sum(imgc,0)
    plt.imshow(imgcsum , cmap=cm.hot, clim=(0.0, 0.7 * imgcsum.max()))

imgb = np.sqrt((img[0::3,:,:]-img[1::3,:,:])**2+(img[1::3,:,:]-img[2::3,:,:])**2+(img[2::3,:,:]-img[0::3,:,:])**2)

if isPlot:
    plt.figure()
    imgbsum = np.sum(imgb,0)
    plt.imshow(imgbsum , cmap=cm.hot, clim=(0.0, 0.7 * imgbsum.max()))

tif.imwrite('/Users/maan/temp/img.tif',img)
tif.imwrite('/Users/maan/temp/imga.tif',imga)
tif.imwrite('/Users/maan/temp/imgb.tif',imgb)
tif.imwrite('/Users/maan/temp/imgc.tif',imgc)
plt.show()

