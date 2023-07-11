import os
import sys
import numpy as np
import scipy.interpolate


def shapedfilter_hrtf(sdelay, freq, gain, fs, ctap, ctap2):
    """
    """
    sdelay = sdelay.reshape((-1, 1)) # Ensure sdelay is an M x 1 matrix
    assert np.all(sdelay >= 0), "The sample delay must be positive"
    ntaps = 2 * ctap - 1
    N = ctap - 1
    fc = 0.9
    # Design the non-integer delay filter
    x = np.ones((1, ntaps))
    x[0, 0] = 0
    x = np.matmul(np.ones(sdelay.shape), np.arange(-N, N + 1).reshape((1, -1))) - np.matmul(sdelay, x)
    h = 0.5 * fc * (1 + np.cos(np.pi * x / N)) * np.sinc(fc * x)
    freq = freq.reshape((-1)) # Ensure freq is a vector
    if ctap2 > 1:
        df = np.arange(0, ctap2) * (np.pi / (ctap2 - 1)) # Determine FFT points
        freq = np.array([-np.spacing(1)] + list(2 * np.pi * freq) + [np.pi])
        gain = np.concatenate([gain[:, :1], gain, gain[:, -1:]], axis=1)
        # Interpolate reflection frequency-dependence to get gains at FFT points
        G = scipy.interpolate.interp1d(freq.reshape([-1]), gain)(df)
        # Combine the non-integer delay filter and the wall/sphere filter
        G[:, ctap2-1] = np.real(G[:, ctap2-1])
        # Transform into appropriate wall transfer function
        G = np.concatenate([G, np.conj(G[:, 1:ctap2-1])[:, ::-1]], axis=1)
        gt = np.real(np.fft.ifft(G.T, axis=0))
        # Zero-pad and FFT
        g = np.concatenate([
            0.5 * gt[(ctap2-1):(ctap2), :],
            gt[ctap2 : (2 * ctap2 - 2), :],
            gt[0 : ctap2 - 1, :],
            0.5 * gt[(ctap2-1):(ctap2), :],
            np.zeros((2 * ctap - 2, gt.shape[1])),
        ], axis=0)
        G = np.fft.fft(g, axis=0)
        # Zero-pad and FFT the delay filter
        H = np.fft.fft(
            np.concatenate([
                h.T,
                np.zeros((2 * ctap2 - 2, gt.shape[1])),
            ], axis=0),
            axis=0
        )
        # Convolve wall transfer function and delay filter
        HOUT = H * G
        # Obtain total impulse response
        hout = np.real(np.fft.ifft(HOUT, axis=0)).T
    else:
        # Scale impulse response only if wall reflections are
        # frequency-independent and sphere is not present
        hout = h * np.matmul(gain[:, 0], np.ones(h[0:1, :].shape))
    return hout


def acoeff_hrtf(material, freq=[125, 250, 500, 1000, 2000, 4000]):
    """
    """
    freqtable = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    walls = [
        [0.03, 0.03, 0.03, 0.04, 0.05, 0.07], # 1  : Brick
        [0.10, 0.05, 0.06, 0.07, 0.09, 0.08], # 2  : Concrete, painted
        [0.35, 0.25, 0.18, 0.12, 0.07, 0.04], # 3  : Window Glass
        [0.01, 0.01, 0.01, 0.01, 0.02, 0.02], # 4  : Marble
        [0.12, 0.09, 0.07, 0.05, 0.05, 0.04], # 5  : Plaster on Concrete
        [0.28, 0.22, 0.17, 0.09, 0.10, 0.11], # 6  : Plywood
        [0.36, 0.44, 0.31, 0.29, 0.39, 0.25], # 7  : Concrete block, coarse
        [0.14, 0.35, 0.55, 0.72, 0.70, 0.65], # 8  : Heavyweight drapery
        [0.08, 0.32, 0.99, 0.76, 0.34, 0.12], # 9  : Fiberglass wall treatment, 1 in
        [0.86, 0.99, 0.99, 0.99, 0.99, 0.99], # 10 : Fiberglass wall treatment, 7 in
        [0.40, 0.90, 0.80, 0.50, 0.40, 0.30], # 11 : Wood panelling on glass fiber blanket
    ]
    floors = [
        [0.04, 0.04, 0.07, 0.06, 0.06, 0.07], # 12 : Wood parquet on concrete
        [0.02, 0.03, 0.03, 0.03, 0.03, 0.02], # 13 : Linoleum
        [0.02, 0.06, 0.14, 0.37, 0.60, 0.65], # 14 : Carpet on concrete
        [0.08, 0.24, 0.57, 0.69, 0.71, 0.73], # 15 : Carpet on foam rubber padding
    ]
    ceilings = [
        [0.14, 0.10, 0.06, 0.05, 0.04, 0.03], # 16 : Plaster, gypsum, or lime on lath
        [0.25, 0.28, 0.46, 0.71, 0.86, 0.93], # 17 : Acoustic tiles, 0.625", 16" below ceiling
        [0.52, 0.37, 0.50, 0.69, 0.79, 0.78], # 18 : Acoustic tiles, 0.5", 16" below ceiling
        [0.10, 0.22, 0.61, 0.66, 0.74, 0.72], # 19 : Acoustic tiles, 0.5" cemented to ceiling
        [0.58, 0.88, 0.75, 0.99, 1.00, 0.96], # 20 : Highly absorptive panels, 1", 16" below ceiling
    ]
    others = [
        [0.19, 0.37, 0.56, 0.67, 0.61, 0.59], # 21 : Upholstered seats
        [0.39, 0.57, 0.80, 0.94, 0.92, 0.87], # 22 : Audience in upholstered seats
        [0.11, 0.26, 0.60, 0.69, 0.92, 0.99], # 23 : Grass
        [0.15, 0.25, 0.40, 0.55, 0.60, 0.60], # 24 : Soil
        [0.01, 0.01, 0.01, 0.02, 0.02, 0.03], # 25 : Water surface
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00], # 26 : Anechoic
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60], # 27 : Uniform (0.6) absorbtion coefficient
        [0.20, 0.20, 0.20, 0.20, 0.20, 0.20], # 28 : Uniform (0.2) absorbtion coefficient
        [0.80, 0.80, 0.80, 0.80, 0.80, 0.80], # 29 : Uniform (0.8) absorbtion coefficient
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14], # 30 : Uniform (0.14) absorbtion coefficient
        [0.08, 0.08, 0.10, 0.10, 0.12, 0.12], # 31 : Artificial - absorbs more at high freqs
        [0.05, 0.05, 0.20, 0.20, 0.10, 0.10], # 32 : Artificial with absorption higher in middle ranges
        [0.12, 0.12, 0.10, 0.10, 0.08, 0.08], # 33 : Artificial  - absorbs more at low freqs
    ]
    atable = np.array(walls + floors + ceilings + others, dtype=float)
    if (material >= 0) and (material < 1):
        # If 0 <= material < 1, walls are set to uniform absorption
        # with a coefficient value equal to `material`
        alpha = material * np.ones_like(freq, dtype=float)
    else:
        alpha = np.zeros_like(freq, dtype=float)
        for itr, f in enumerate(freq):
            if f == 0:
                alpha[itr] = 0
            else:
                alpha[itr] = np.interp(
                    f,
                    freqtable,
                    atable[material - 1],
                    left=atable[material - 1, 0],
                    right=atable[material - 1, -1])
    return alpha, freq
