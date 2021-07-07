import os

from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
import time
import openh5dataset

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
h.wavelength = 0.6
h.eta = 0.35
h.alpha = 0.2
h.beta = 0.99
h.a = 0.2
h.a_type = 'sph'

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
# img1 = tif.imread('/Users/maan/Downloads/slices_503_to_demodulate.tif')

dataset_num = 11
file_name = '/Users/maan/OneDrive - Imperial College London/measurement/Milan/210611_152833_PROCHIP_SIM_ROI_dataset51/210528_195219_PROCHIP_multichannel_ROI_dataset11.h5'

t_idx = f'/t{dataset_num:04d}/'
index_list, index_names = openh5dataset.get_datasets_index_by_name(file_name, t_idx)
img1, _ = openh5dataset.get_multiple_h5_datasets(file_name, index_list)

# if Nsize != 512:
#     img1 = np.single(img1[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
# else:
img1 = np.single(img1)
print(img1.shape)
frames = img1.shape[0] * img1.shape[1]
sizexy = img1.shape[2]
img = img1.transpose((1,0,2,3)).reshape((frames,sizexy,sizexy))
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
h.calibrate(img[570:639,:,:])
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
h.calibrate(img[570:639,:,:])
imgc = h.batchreconstruct(img)

if isPlot:
    plt.figure()
    imgasum = np.sum(imga,0)
    plt.imshow(imgasum , cmap=cm.hot, clim=(0.0, 0.7 * imgasum.max()))
    plt.title('normal SIM')
    plt.figure()
    imgcsum = np.sum(imgc,0)
    plt.imshow(imgcsum , cmap=cm.hot, clim=(0.0, 0.7 * imgcsum.max()))
    plt.title('phase corrected SIM')

imgb = np.sqrt((img[0::3,:,:]-img[1::3,:,:])**2+(img[1::3,:,:]-img[2::3,:,:])**2+(img[2::3,:,:]-img[0::3,:,:])**2)

if isPlot:
    plt.figure()
    imgbsum = np.sum(imgb,0)
    plt.imshow(imgbsum , cmap=cm.hot, clim=(0.0, 0.7 * imgbsum.max()))
    plt.title('OS-SIM')

tif.imwrite('/Users/maan/temp/img.tif',img)
tif.imwrite('/Users/maan/temp/imga.tif',imga)
tif.imwrite('/Users/maan/temp/imgb.tif',imgb)
tif.imwrite('/Users/maan/temp/imgc.tif',imgc)

start = 570
length = 69
phase = np.zeros(length)
ampl = np.zeros(length)
h.debug = False
for i in range(0, length, 3):
    p, a = h.find_phase(h.kx, h.ky, img[start + i:start + i + 3,:,:])
    phase[i:i+3] = p
    ampl[i:i+3] = a
p, a = h.find_phase(h.kx, h.ky, img[start:start + length,:,:])
plt.figure(99)
plt.polar(phase[0::3], ampl[0::3], 'gx-')
plt.polar(phase[1::3], ampl[1::3], 'bx-')
plt.polar(phase[2::3], ampl[2::3], 'rx-')
plt.polar(p[0], a[0], color='lightgreen', marker='*', markersize=15)
plt.polar(p[1], a[1], color='lightblue', marker='*', markersize=15)
plt.polar(p[2], a[2], color='lightpink', marker='*', markersize=15)

expected_phase = - np.arange(phase.size) * 2 * np.pi / 3
phase = np.unwrap(phase - expected_phase) + expected_phase - phase[0]
plt.figure(100)
plt.plot(phase, 'gx-')
plt.plot(expected_phase, 'g--')
plt.figure(101)
plt.plot(phase - expected_phase, 'gx-')
plt.plot(np.zeros_like(expected_phase), 'g--')
plt.figure(102)
d = phase - expected_phase;
plt.plot(d[0::3], 'gx-')
plt.plot(d[1::3], 'bx-')
plt.plot(d[2::3], 'rx-')
plt.plot(np.zeros_like(expected_phase)[0::3], 'g--')

plt.show()

