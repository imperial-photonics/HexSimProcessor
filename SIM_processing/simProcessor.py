import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, pi, sqrt, log2, arccos

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft

    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.interfaces.cache.enable()
    fftw = True
except:
    import numpy.fft as fft

try:
    import cv2

    opencv = True
except:
    opencv = False

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.ndimage
    from cupyx.scipy.ndimage.filters import gaussian_filter as gaussian_filter_cupy

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    cupy = True
except:
    cupy = False


from SIM_processing.hexSimProcessor import HexSimProcessor

class SimProcessor(HexSimProcessor):

    def __init__(self):
        self._lastN = 0
        self.kx = np.zeros((3, 1), dtype=np.single)
        self.ky = np.zeros((3, 1), dtype=np.single)
        self.p = np.zeros((3, 1), dtype=np.single)
        self.ampl = np.zeros((3, 1), dtype=np.single)

    def _allocate_arrays(self):
        ''' define matrix '''
        self._reconfactor = np.zeros((3, 2 * self.N, 2 * self.N), dtype=np.single)  # for reconstruction

        self._prefilter = np.zeros((self.N, self.N),
                                   dtype=np.single)  # for prefilter stage, includes otf and zero order supression
        self._postfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        self._carray = np.zeros((3, 2 * self.N, 2 * self.N), dtype=np.complex64)
        self._carray1 = np.zeros((3, 2 * self.N, self.N + 1), dtype=np.complex64)

        self._imgstore = np.zeros((3, 2 * self.N, 2 * self.N), dtype=np.single)
        self._bigimgstore = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if cupy:
            # self._prefilter_cp = cp.zeros((self.N, self.N), dtype=np.single)
            # self._postfilter_cp = cp.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_cp = cp.zeros((3, 2 * self.N, self.N + 1), dtype=np.complex)
            # self._reconfactor_cp = cp.zeros((3, 2 * self.N, 2 * self.N), dtype=np.single)
            self._bigimgstore_cp = cp.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if opencv:
            self._prefilter_ocv = np.zeros((self.N, self.N),
                                           dtype=np.single)  # for prefilter stage, includes otf and zero order supression
            self._postfilter_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocvU = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC2)
            self._bigimgstoreU = cv2.UMat(self._bigimgstore)
            self._imgstoreU = [cv2.UMat((self.N, self.N), s=0.0, type=cv2.CV_32F) for i in range(3)]
        self._lastN = self.N

    def _calibrate(self, img, findCarrier = True, useCupy = False):
        assert len(img) > 2
        self.N = len(img[0, :, :])
        if self.N != self._lastN:
            self._allocate_arrays()

        ''' define k grids '''
        self._dx = self.pixelsize / self.magnification  # Sampling in image plane
        self._res = self.wavelength / (2 * self.NA)
        self._oversampling = self._res / self._dx
        self._dk = self._oversampling / (self.N / 2)  # Sampling in frequency plane
        self._k = np.arange(-self._dk * self.N / 2, self._dk * self.N / 2, self._dk, dtype=np.double)
        self._dx2 = self._dx / 2

        self._kr = np.sqrt(self._k ** 2 + self._k[:,np.newaxis] ** 2, dtype=np.single)
        kxbig = np.arange(-self._dk * self.N, self._dk * self.N, self._dk, dtype=np.single)
        kybig = kxbig[:,np.newaxis]

        '''Sum input images if there are more than 3'''
        if len(img) > 3:
            imgs = np.zeros((3, self.N, self.N), dtype=np.single)
            for i in range(3):
                imgs[i, :, :] = np.sum(img[i:(len(img) // 3) * 3:3, :, :], 0, dtype = np.single)
        else:
            imgs = np.single(img)

        '''Separate bands into DC and 1 high frequency band'''
        M = exp(1j * 2 * pi / 3) ** ((np.arange(0, 2)[:, np.newaxis]) * np.arange(0, 3))

        sum_prepared_comp = np.zeros((2, self.N, self.N), dtype=np.complex64)
        wienerfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)

        for k in range(0, 2):
            for l in range(0, 3):
                sum_prepared_comp[k, :, :] = sum_prepared_comp[k, :, :] + imgs[l, :, :] * M[k, l]

        if findCarrier:
            # minimum search radius in k-space
            mask1 = (self._kr > 1.9 * self.eta)
            if useCupy:
                self.kx, self.ky = self._coarseFindCarrier_cupy(sum_prepared_comp[0, :, :],
                                                                sum_prepared_comp[1, :, :], mask1)
            else:
                self.kx, self.ky = self._coarseFindCarrier(sum_prepared_comp[0, :, :],
                                                           sum_prepared_comp[1, :, :], mask1)

        if useCupy:
            ckx, cky, p, ampl = self._refineCarrier_cupy(sum_prepared_comp[0, :, :],
                                                         sum_prepared_comp[1, :, :], self.kx, self.ky)
        else:
            ckx, cky, p, ampl = self._refineCarrier(sum_prepared_comp[0, :, :],
                                                    sum_prepared_comp[1, :, :], self.kx, self.ky)

        self.kx = ckx # store found kx, ky, p and ampl values
        self.ky = cky
        self.p = p
        self.ampl = ampl

        if self.debug:
            print(f'kx = {ckx}')
            print(f'ky = {cky}')
            print(f'p  = {p}')
            print(f'a  = {ampl}')

        ph = np.single(2 * pi * self.NA / self.wavelength)

        xx = np.arange(-self._dx2 * self.N, self._dx2 * self.N, self._dx2, dtype=np.single)
        yy = xx

        if self.usemodulation:
            A = ampl
        else:
            A = 1

        for idx_p in range(0, 3):
            pstep = idx_p * 2 * pi / 3
            if useCupy:
                self._reconfactor[idx_p, :, :] = (
                        1 + 4 / A * cp.outer(cp.exp(cp.asarray(1j * (ph * cky * yy - pstep + p))),
                                             cp.exp(cp.asarray(1j * ph * ckx * xx))).real).get()
            else:
                self._reconfactor[idx_p, :, :] = (
                        1 + 4 / A * np.outer(np.exp(1j * (ph * cky * yy - pstep + p)),
                                             np.exp(1j * ph * ckx * xx)).real)

        # calculate pre-filter factors

        mask2 = (self._kr < 2)

        self._prefilter = np.single((self._tfm(self._kr, mask2) * self._attm(self._kr, mask2)))
        self._prefilter = fft.fftshift(self._prefilter)

        mtot = np.full((2 * self.N, 2 * self.N), False)

        krbig = sqrt((kxbig - ckx) ** 2 + (kybig - cky) ** 2)
        mask = (krbig < 2)
        mtot = mtot | mask
        wienerfilter[mask] = wienerfilter[mask] + (self._tf(krbig[mask]) ** 2) * self._att(krbig[mask])
        krbig = sqrt((kxbig + ckx) ** 2 + (kybig + cky) ** 2)
        mask = (krbig < 2)
        mtot = mtot | mask
        wienerfilter[mask] = wienerfilter[mask] + (self._tf(krbig[mask]) ** 2) * self._att(krbig[mask])
        krbig = sqrt(kxbig ** 2 + kybig ** 2)
        mask = (krbig < 2)
        mtot = mtot | mask
        wienerfilter[mask] = wienerfilter[mask] + (self._tf(krbig[mask]) ** 2) * self._att(krbig[mask])
        self.wienerfilter = wienerfilter

        if self.debug:
            plt.figure()
            plt.title('WienerFilter')
            plt.imshow(wienerfilter)

        th = np.linspace(0, 2 * pi, 360, dtype = np.single)
        inv = np.geterr()['invalid']
        np.seterr(invalid = 'ignore')
        kmaxth = np.fmax(2, np.fmax(ckx * np.cos(th) + cky * np.sin(th) +
                                        np.sqrt(4 - (ckx * np.sin(th)) ** 2  - (cky * np.cos(th)) ** 2  +
                                            ckx * cky * np.sin(2 * th)),
                                    - ckx * np.cos(th) - cky * np.sin(th) +
                                        np.sqrt(4 - (ckx * np.sin(th)) ** 2  - (cky * np.cos(th)) ** 2  +
                                            ckx * cky * np.sin(2 * th))))
        np.seterr(invalid = inv)

        thbig = np.arctan2(kybig,kxbig)
        kmax = np.single(np.interp(thbig,th,kmaxth, period = 2 * pi))
        wienerfilter = mtot * (1 - krbig * mtot / kmax) / (wienerfilter * mtot + self.w ** 2)

        self._postfilter = fft.fftshift(wienerfilter)

        if opencv:
            self._reconfactorU = [cv2.UMat(self._reconfactor[idx_p, :, :]) for idx_p in range(0, 3)]
            self._prefilter_ocv = np.single(cv2.dft(fft.ifft2(self._prefilter).real))
            pf = np.zeros((self.N, self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._prefilter
            pf[:, :, 1] = self._prefilter
            self._prefilter_ocvU = cv2.UMat(np.single(pf))
            self._postfilter_ocv = np.single(cv2.dft(fft.ifft2(self._postfilter).real))
            pf = np.zeros((2 * self.N, 2 * self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._postfilter
            pf[:, :, 1] = self._postfilter
            self._postfilter_ocvU = cv2.UMat(np.single(pf))

        if cupy:
            self._postfilter_cp = cp.asarray(self._postfilter)

    def reconstruct_ocv(self, img):
        assert opencv, "No opencv present"
        img2 = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        for i in range(3):
            imf = cv2.mulSpectrums(cv2.dft(img[i, :, :]), self._prefilter_ocv, 0)
            self._carray_ocv[0:self.N // 2, 0:self.N] = imf[0:self.N // 2, 0:self.N]
            self._carray_ocv[3 * self.N // 2:2 * self.N, 0:self.N] = imf[self.N // 2:self.N, 0:self.N]
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocv, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactor[i, :, :]))
        self._imgstore = img.copy()
        return cv2.idft(cv2.mulSpectrums(cv2.dft(img2), self._postfilter_ocv, 0),
                        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    def reconstruct_ocvU(self, img):
        assert opencv, "No opencv present"
        img2 = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC1)
        mask = cv2.UMat((self.N // 2, self.N // 2), s=1, type=cv2.CV_8U)
        for i in range(3):
            self._imgstoreU[i] = cv2.UMat(img[i, :, :])
            imf = cv2.multiply(cv2.dft(self._imgstoreU[i], flags=cv2.DFT_COMPLEX_OUTPUT), self._prefilter_ocvU)
            cv2.copyTo(src=cv2.UMat(imf, (0, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (0, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 3 * self.N // 2, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 3 * self.N // 2, self.N // 2, self.N // 2)))
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocvU, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactorU[i]))
        self._bigimgstoreU = cv2.idft(cv2.multiply(cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT), self._postfilter_ocvU),
                                      flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return self._bigimgstoreU

    # region Stream reconstruction function

    # endregion

    def batchreconstruct(self, img):
        nim = img.shape[0]
        r = np.mod(nim, 6)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 6
            img = np.concatenate((img, np.zeros((6 - r, self.N, self.N), np.single)))
            nim = nim + 6 - r
        nim3 = nim // 3
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, 3):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 3, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 3, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + 3, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        img3 = fft.irfft(fft.rfft(img2, nim, 0)[0:nim3 // 2 + 1, :, :], nim3, 0)
        res = fft.irfft2(fft.rfft2(img3) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstructcompact(self, img):
        nim = img.shape[0]
        r = np.mod(nim, 6)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 6
            img = np.concatenate((img, np.zeros((6 - r, self.N, self.N), np.single)))
            nim = nim + 6 - r
        nim3 = nim // 3
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, 3):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 3, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 3, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + 3, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        res = np.zeros((nim3, 2 * self.N, 2 * self.N), dtype=np.single)

        imgf = fft.rfft(img2[:, :self.N, :self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        res[:, :self.N, :self.N] = fft.irfft(imgf, nim3, 0)
        imgf = fft.rfft(img2[:, :self.N, self.N:2 * self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        res[:, :self.N, self.N:2 * self.N] = fft.irfft(imgf, nim3, 0)
        imgf = fft.rfft(img2[:, self.N:2 * self.N, :self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        res[:, self.N:2 * self.N, :self.N] = fft.irfft(imgf, nim3, 0)
        imgf = fft.rfft(img2[:, self.N:2 * self.N, self.N:2 * self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        res[:, self.N:2 * self.N, self.N:2 * self.N] = fft.irfft(imgf, nim3, 0)

        res = fft.irfft2(fft.rfft2(res) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstruct_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.asarray(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 6)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 6
            img = np.concatenate((img, cp.zeros((6 - r, self.N, self.N), np.single)))
            nim = nim + 6 - r
        nim3 = nim // 3
        imf = cp.fft.rfft2(img) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])

        del img
        cp._default_memory_pool.free_all_blocks()

        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((3, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.asarray(self._reconfactor)
        for i in range(0, nim, 3):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 3, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 3, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + 3, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp

        del imf
        del bcarray
        del reconfactor_cp
        cp._default_memory_pool.free_all_blocks()

        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())

        img3 = cp.fft.irfft(cp.fft.rfft(img2, nim, 0)[0:nim3 // 2 + 1, :, :], nim3, 0)
        del img2
        cp._default_memory_pool.free_all_blocks()
        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())
        res = (cp.fft.irfft2(cp.fft.rfft2(img3) * self._postfilter_cp[:, :self.N + 1])).get()
        return res

    def batchreconstructcompact_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.asarray(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 6)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 6
            img = cp.concatenate((img, cp.zeros((6 - r, self.N, self.N), np.single)))
            nim = nim + 6 - r
        nim3 = nim // 3
        imf = cp.fft.rfft2(img) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])

        del img
        cp._default_memory_pool.free_all_blocks()

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((3, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.asarray(self._reconfactor)
        for i in range(0, nim, 3):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 7, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 3, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + 3, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp

        del bcarray
        del reconfactor_cp
        cp._default_memory_pool.free_all_blocks()

        imgout = cp.zeros((nim3, 2 * self.N, 2 * self.N), dtype=np.single)
        imf = cp.fft.rfft(img2[:, 0:self.N, 0:self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        imgout[:, 0:self.N, 0:self.N] = cp.fft.irfft(imf, nim3, 0)
        imf = cp.fft.rfft(img2[:, self.N:2 * self.N, 0:self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        imgout[:, self.N:2 * self.N, 0:self.N] = cp.fft.irfft(imf, nim3, 0)
        imf = cp.fft.rfft(img2[:, 0:self.N, self.N:2 * self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        imgout[:, 0:self.N, self.N:2 * self.N] = cp.fft.irfft(imf, nim3, 0)
        imf = cp.fft.rfft(img2[:, self.N:2 * self.N, self.N:2 * self.N], nim, 0)[:nim3 // 2 + 1, :, :]
        imgout[:, self.N:2 * self.N, self.N:2 * self.N] = cp.fft.irfft(imf, nim3, 0)

        del imf
        cp._default_memory_pool.free_all_blocks()

        res = (cp.fft.irfft2(cp.fft.rfft2(imgout) * self._postfilter_cp[:, :self.N + 1])).get()

        return res

