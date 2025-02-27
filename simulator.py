import functools
import glob
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.io
import scipy.signal
import soundfile as sf


def normalize_angle(angle, lower=-180, upper=180):
    """
    Based on `normalize` function from:
    https://gist.github.com/phn/1111712/35e8883de01916f64f7f97da9434622000ac0390
    """
    assert lower < upper, f"Invalid lower and upper limits: [{lower}, {upper}]"
    angle = np.array(angle)
    IDX_ABOVE = np.logical_or(angle > upper, angle == lower)
    IDX_BELOW = np.logical_or(angle < lower, angle == upper)
    angle[IDX_ABOVE] = lower + np.mod(
        np.abs(angle[IDX_ABOVE] + upper),
        np.abs(lower) + np.abs(upper),
    )
    angle[IDX_BELOW] = upper - np.mod(
        np.abs(angle[IDX_BELOW] - lower),
        np.abs(lower) + np.abs(upper),
    )
    return angle


def load_kemar_hrtfs(npz_filename="kemar_hrtfs/hrtfs.npz"):
    """
    Helper function to load KEMAR HRTFs from Gardner & Martin (1994).
    Source: https://sound.media.mit.edu/resources/KEMAR.html

    Returns
    -------
    hrtf_locs (float array w/ shape [368, 3]): polar (1.4m, azim, elev)
    hrtf_firs (float array w/ shape [368, 128, 2]): impulse responses
    hrtf_sr (int): sampling rate of impulse responses (44100 Hz)
    """
    if os.path.exists(npz_filename):
        f = np.load(npz_filename)
        return f["hrtf_locs"], f["hrtf_firs"], f["hrtf_sr"]
    else:
        # Azimuths measured at +/- 40° elevation do not occur at integer angles
        azim_elev40 = np.linspace(0, 180, 56 // 2 + 1)
        list_fn_hrtf = []
        for elev in np.arange(-40, 91, 10, dtype=int):
            list_fn_hrtf.extend(sorted(glob.glob(f"kemar_hrtfs/elev{elev}/*wav")))
        hrtf_locs = []
        hrtf_firs = []
        for fn_hrtf in list_fn_hrtf:
            hrtf, hrtf_sr = sf.read(fn_hrtf)
            tmp = os.path.basename(fn_hrtf).replace(".wav", "")
            elev, azim = [float(_) for _ in tmp.strip("Ha").split("e")]
            if np.abs(elev) == 40:
                azim = azim_elev40[np.argmin(np.abs(azim_elev40 - azim))]
            hrtf_locs.append([1.4, azim, elev])
            hrtf_firs.append(hrtf)
        hrtf_locs = np.array(hrtf_locs)
        hrtf_firs = np.array(hrtf_firs)
        np.savez(
            npz_filename, hrtf_locs=hrtf_locs, hrtf_firs=hrtf_firs, hrtf_sr=hrtf_sr
        )
        print(f"[load_kemar_hrtfs] cache file for future calls: {npz_filename}")
    return hrtf_locs, hrtf_firs, hrtf_sr


def get_material_from_material_int(material_int):
    """
    The MATLAB function `acoeff_hrtf.m` maps integer codes to materials with different
    acoustic absorption coefficients. This function maps the same integer codes to
    dict-like objects corresponding to materials included in the MATLAB implementation.
    """
    df = pd.read_csv("materials_original.csv")
    df.absorption_coefficients = df.absorption_coefficients.map(
        lambda _: np.array(eval(_), dtype=float)
    )
    df.absorption_frequencies = df.absorption_frequencies.map(
        lambda _: np.array(eval(_), dtype=float)
    )
    df = df[df.material_int == material_int]
    msg = f"`material_int={material_int}` does not match"
    msg += " an integer code in the MATLAB implementation. "
    msg += "Specify material as a dict with `absorption_frequencies`"
    msg += " and `absorption_coefficients` fields."
    assert len(df) == 1, msg
    return df.iloc[0]


def acoeff_hrtf(material, freq=[125, 250, 500, 1000, 2000, 4000], verbose=False):
    """
    Python implementation of `acoeff_hrtf.m` by msaddler (2023/07).
    """
    freq = np.array(freq, dtype=float)
    if isinstance(material, (np.integer, int)):
        # If material is an integer, convert to dict
        # (mapping from original MATLAB implementation)
        material = get_material_from_material_int(material)
    if "absorption_coefficients" in dict(material):
        # Otherwise, material must be dict-like with
        # `absorption_frequencies` and `absorption_coefficients`
        absorption_frequencies = dict(material)["absorption_frequencies"]
        absorption_coefficients = dict(material)["absorption_coefficients"]
    else:
        msg = "`material` must have fields: "
        msg += "`absorption_frequencies` and `absorption_coefficients`"
        raise ValueError(msg)
    if (verbose > 1) and ("material_name" in dict(material)):
        print(f"... material: {dict(material)['material_name']}")
    alpha = np.zeros_like(freq, dtype=float)
    for itr, f in enumerate(freq):
        if f == 0:
            alpha[itr] = 0
        else:
            alpha[itr] = np.interp(
                f,
                absorption_frequencies,
                absorption_coefficients,
                left=absorption_coefficients[0],
                right=absorption_coefficients[-1],
            )
    return freq, alpha


def shapedfilter_hrtf(sdelay, freq, gain, sr, ctap, ctap2):
    """
    Python implementation of `shapedfilter_hrtf.m` by msaddler (2023/07).
    """
    sdelay = sdelay.reshape((-1, 1))  # Ensure sdelay is an M x 1 matrix
    assert np.all(sdelay >= 0), "The sample delay must be positive"
    ntaps = 2 * ctap - 1
    N = ctap - 1
    fc = 0.9
    # Design the non-integer delay filter
    x = np.ones((1, ntaps))
    x[0, 0] = 0
    x = np.matmul(
        np.ones(sdelay.shape), np.arange(-N, N + 1).reshape((1, -1))
    ) - np.matmul(sdelay, x)
    h = 0.5 * fc * (1 + np.cos(np.pi * x / N)) * np.sinc(fc * x)
    freq = freq.reshape((-1))  # Ensure freq is a vector
    if ctap2 > 1:
        # Determine FFT points
        df = np.arange(0, ctap2) * (np.pi / (ctap2 - 1))
        freq = np.array([-np.spacing(1)] + list(2 * np.pi * freq) + [np.pi])
        gain = np.concatenate([gain[:, :1], gain, gain[:, -1:]], axis=1)
        # Interpolate reflection frequency-dependence to get gains at FFT points
        G = scipy.interpolate.interp1d(freq.reshape([-1]), gain)(df)
        # Combine the non-integer delay filter and the wall/sphere filter
        G[:, ctap2 - 1] = np.real(G[:, ctap2 - 1])
        # Transform into appropriate wall transfer function
        G = np.concatenate([G, np.conj(G[:, 1 : ctap2 - 1])[:, ::-1]], axis=1)
        gt = np.real(np.fft.ifft(G.T, axis=0))
        # Zero-pad and FFT
        g = np.concatenate(
            [
                0.5 * gt[(ctap2 - 1) : (ctap2), :],
                gt[ctap2 : (2 * ctap2 - 2), :],
                gt[0 : ctap2 - 1, :],
                0.5 * gt[(ctap2 - 1) : (ctap2), :],
                np.zeros((2 * ctap - 2, gt.shape[1])),
            ],
            axis=0,
        )
        G = np.fft.fft(g, axis=0)
        # Zero-pad and FFT the delay filter
        H = np.fft.fft(
            np.concatenate(
                [
                    h.T,
                    np.zeros((2 * ctap2 - 2, gt.shape[1])),
                ],
                axis=0,
            ),
            axis=0,
        )
        # Convolve wall transfer function and delay filter
        HOUT = H * G
        # Obtain total impulse response
        hout = np.real(np.fft.ifft(HOUT, axis=0)).T
    else:
        # Scale impulse response only if wall reflections are
        # frequency-independent and sphere is not present
        hout = h * np.matmul(gain[:, 0:1], np.ones(h[0:1, :].shape))
    return hout


def func_to_parallelize(
    itr_loc,
    h=None,
    nearest_hrtf_loc=None,
    flip=None,
    ctap=None,
    ctap2=None,
    lead_zeros=None,
    hrtf_delay=None,
    rel_dist=None,
    sr=None,
    c=None,
    ntaps=None,
    nfreq=None,
    gains=None,
    s_locations_pol=None,
    hrtf_locs=None,
    hrtf_firs=None,
):
    """
    Contents of the parfor loop in `impulse_generate_hrtf.m` (line 170).
    """
    hrtf_temp = np.zeros_like(h)
    IDX_loc = nearest_hrtf_loc == itr_loc
    IDX_noflip = np.logical_and(IDX_loc, ~flip)
    IDX_flip = np.logical_and(IDX_loc, flip)
    # Treat non-flipped sources
    if IDX_noflip.sum() > 0:
        # Get sample delays to the measured location
        thit = (
            ctap
            + ctap2
            - lead_zeros
            + hrtf_delay[itr_loc]
            + (rel_dist[IDX_noflip] * sr / c)
        )
        ihit = np.floor(thit)
        fhit = thit - ihit
        gains_noflip = gains[IDX_noflip, :]
        # Get scale factors to account for distance traveled
        m_sc = 1 / hrtf_locs[nearest_hrtf_loc[IDX_noflip], 0]
        s_sc = 1 / s_locations_pol[IDX_noflip, 0]
        rel_sc = s_sc / m_sc
        # Eliminate locations that are too far away to enter into impulse response
        v = ihit <= ntaps + ctap + ctap2
        if v.sum() > 0:
            # Initialize temporary impulse response vector
            ht = np.zeros((h.shape[0] + ctap + 1 + ctap2 + 1, 1), dtype=float)
            # Indices into ht. Each row corresonds to one source image location,
            # with the center determined by ihit. Within a row, there are
            # (2 * ctap - 1) + (2 * ctap2 - 1) - 1 values that account for
            # non-integer dela, fhit, and for frequency-dependent wall reflections /
            # sphere diffraction
            ht_ind = ihit[v].reshape(-1, 1) * np.ones(
                (1, 2 * ctap - 1 + 2 * ctap2 - 1 - 1)
            )
            ht_ind = ht_ind + np.arange(
                -ctap - ctap2 + 1 + 1, ctap + ctap2 - 1
            ).reshape((1, -1))
            ht_ind = ht_ind.astype(int)
            # For each source location, determine the impulse response
            # (generate filter to incorporate frequency gains, non-integer
            # delay and scattering off rigid sphere)
            h_temp = rel_sc[v].reshape(-1, 1) * shapedfilter_hrtf(
                fhit[v], nfreq, gains_noflip[v], sr, ctap, ctap2
            )
            # Add impulse response segments into the overall impulse response
            for k in range(v.sum()):
                ht[ht_ind[k], 0] = ht[ht_ind[k], 0] + h_temp[k, :]
            # Incorporate HRTF impulse response and add into overall impulse response
            # matrix
            hrtf = hrtf_firs[itr_loc]
            new_vals = np.stack(
                [
                    scipy.signal.convolve(ht[: h.shape[0], 0], hrtf[:, 0], mode="full"),
                    scipy.signal.convolve(ht[: h.shape[0], 0], hrtf[:, 1], mode="full"),
                ],
                axis=1,
            )
            hrtf_temp = hrtf_temp + new_vals[: hrtf_temp.shape[0]]
    # Treat flipped sources
    if IDX_flip.sum() > 0:
        # Get sample delays to the measured location
        thit = (
            ctap
            + ctap2
            - lead_zeros
            + hrtf_delay[itr_loc]
            + (rel_dist[IDX_flip] * sr / c)
        )
        ihit = np.floor(thit)
        fhit = thit - ihit
        gains_flip = gains[IDX_flip, :]
        # Get scale factors to account for distance traveled
        m_sc = 1 / hrtf_locs[nearest_hrtf_loc[IDX_flip], 0]
        s_sc = 1 / s_locations_pol[IDX_flip, 0]
        rel_sc = s_sc / m_sc
        # Eliminate locations that are too far away to enter into impulse response
        v = ihit <= ntaps + ctap + ctap2
        if v.sum() > 0:
            # Initialize temporary impulse response vector
            ht = np.zeros((h.shape[0] + ctap + 1 + ctap2 + 1, 1), dtype=float)
            # Indices into ht. Each row corresonds to one source image location,
            # with the center determined by ihit. Within a row, there are
            # (2 * ctap - 1) + (2 * ctap2 - 1) - 1 values that account for non-integer
            # delay, fhit, and for frequency-dependent wall reflections / sphere
            # diffraction
            ht_ind = ihit[v].reshape(-1, 1) * np.ones(
                (1, 2 * ctap - 1 + 2 * ctap2 - 1 - 1)
            )
            ht_ind = ht_ind + np.arange(
                -ctap - ctap2 + 1 + 1, ctap + ctap2 - 1
            ).reshape((1, -1))
            ht_ind = ht_ind.astype(int)
            # For each source location, determine the impulse response (generate filter
            # to incorporate frequency gains, non-integer delay and scattering off rigid
            # sphere
            h_temp = rel_sc[v].reshape(-1, 1) * shapedfilter_hrtf(
                fhit[v], nfreq, gains_flip[v], sr, ctap, ctap2
            )
            # Add impulse response segments into the overall impulse response
            for k in range(v.sum()):
                ht[ht_ind[k], 0] = ht[ht_ind[k], 0] + h_temp[k, :]
            # Incorporate HRTF impulse response and add into overall impulse response
            # matrix
            hrtf = hrtf_firs[itr_loc]
            new_vals = np.stack(
                [
                    scipy.signal.convolve(ht[: h.shape[0], 0], hrtf[:, 1], mode="full"),
                    scipy.signal.convolve(ht[: h.shape[0], 0], hrtf[:, 0], mode="full"),
                ],
                axis=1,
            )
            hrtf_temp = hrtf_temp + new_vals[: hrtf_temp.shape[0]]
    return hrtf_temp


def impulse_generate_hrtf(
    h=None,
    head_cent=None,
    head_azim=None,
    s_locations=None,
    s_reflections=None,
    hrtf_locs=None,
    hrtf_locs_xyz=None,
    hrtf_locs_xyz_logdist=None,
    hrtf_firs=None,
    hrtf_delay=None,
    sr=None,
    c=None,
    ntaps=None,
    ctap=None,
    ctap2=None,
    fgains=None,
    nfreq=None,
    lead_zeros=None,
    use_hrtf_symmetry=None,
    use_log_distance=None,
    use_jitter=None,
    pool=None,
    verbose=True,
):
    """
    Python implementation of `impulse_generate_hrtf.m` by msaddler (2023/07).
    """
    jitter_reflects = 5

    """
    Part I: Form variables to be used in impulse response generation
    """
    # Determine overall source gains (based on number of reflections
    # through each wall) for each source location.
    gains = np.ones((s_locations.shape[0], nfreq.shape[0]), dtype=float)
    for itr_wall in range(6):
        gains = gains * np.power(
            fgains[itr_wall : itr_wall + 1, :],
            s_reflections[:, itr_wall : itr_wall + 1],
        )
    # If use_hrtf_symmetry is active, convert 180° to 360° sources to 0° to 180° sources
    s_locations_relh = s_locations - head_cent.reshape((1, -1))
    s_locations_pol = np.zeros_like(s_locations)
    s_locations_pol[:, 0] = np.sqrt(np.sum(np.square(s_locations_relh), axis=1))
    s_locations_pol[:, 1] = (
        np.rad2deg(np.angle(s_locations_relh[:, 0] - 1j * s_locations_relh[:, 1]))
        - head_azim
    )
    s_locations_pol[:, 1] = normalize_angle(
        s_locations_pol[:, 1],
        lower=-180,
        upper=180,
    )  # Bugfix to ensure angles outside 0° to 180° get `flipped` (msaddler, 2025/01)
    s_locations_pol[:, 2] = np.rad2deg(
        np.arcsin(s_locations_relh[:, 2] / s_locations_pol[:, 0])
    )
    if use_hrtf_symmetry:
        flip = s_locations_pol[:, 1] < 0
        s_locations_pol[:, 1] = np.abs(s_locations_pol[:, 1])
        r = s_locations_pol[:, 0]
        s_locations = np.stack(
            [
                r
                * np.cos(np.deg2rad(s_locations_pol[:, 1] + head_azim))
                * np.cos(np.deg2rad(s_locations_pol[:, 2])),
                r
                * -np.sin(np.deg2rad(s_locations_pol[:, 1] + head_azim))
                * np.cos(np.deg2rad(s_locations_pol[:, 2])),
                r * np.sin(np.deg2rad(s_locations_pol[:, 2])),
            ],
            axis=1,
        )
        s_locations = s_locations + head_cent.reshape((1, -1))
    else:
        flip = np.zeros((s_locations.shape[0]), dtype=bool)

    # If use_log_distance is active, form s_locations_logdist
    if use_log_distance:
        r = np.log(s_locations_pol[:, 0]) - np.log(0.05)
        s_locations_logdist = np.stack(
            [
                r
                * np.cos(np.deg2rad(s_locations_pol[:, 1] + head_azim))
                * np.cos(np.deg2rad(s_locations_pol[:, 2])),
                r
                * -np.sin(np.deg2rad(s_locations_pol[:, 1] + head_azim))
                * np.cos(np.deg2rad(s_locations_pol[:, 2])),
                r * np.sin(np.deg2rad(s_locations_pol[:, 2])),
            ],
            axis=1,
        )
        s_locations_logdist = s_locations_logdist + head_cent.reshape((1, -1))
        D = (
            hrtf_locs_xyz_logdist[:, np.newaxis, :]
            - s_locations_logdist[np.newaxis, :, :]
        )
    else:
        D = hrtf_locs_xyz[:, np.newaxis, :] - s_locations[np.newaxis, :, :]
    # For each source, determine the closest measurement spot
    D = np.sqrt(np.sum(np.square(D), axis=2))
    nearest_hrtf_loc = np.argmin(D, axis=0)

    """
    Part II: Based on the center of the head, introduce a
    1 percent jitter to add into all source-to-mic distances
    that are reflected by more than 5 walls (if use_jitter)
    """
    if use_jitter:
        jitt = np.random.randn(s_locations_pol.shape[0])
        jitt[s_reflections.sum(axis=1) < jitter_reflects] = 0
        s_locations_pol[:, 0] = s_locations_pol[:, 0] + jitt
    # Calculate the relative additional distance between each
    # (jittered) source and the corresponding measurement location
    rel_dist = s_locations_pol[:, 0] - hrtf_locs[nearest_hrtf_loc, 0]

    """
    Part III: For each measurement location, generate impulse
    response from corresponding sources to measured location
    and then incorporate HRTFs (treat flips / no flips accordingly)
    """
    f = functools.partial(
        func_to_parallelize,
        h=h,
        nearest_hrtf_loc=nearest_hrtf_loc,
        flip=flip,
        ctap=ctap,
        ctap2=ctap2,
        lead_zeros=lead_zeros,
        hrtf_delay=hrtf_delay,
        rel_dist=rel_dist,
        sr=sr,
        c=c,
        ntaps=ntaps,
        nfreq=nfreq,
        gains=gains,
        s_locations_pol=s_locations_pol,
        hrtf_locs=hrtf_locs,
        hrtf_firs=hrtf_firs,
    )
    list_loc = [_ for _ in range(hrtf_locs.shape[0]) if _ in nearest_hrtf_loc]
    if verbose > 1:
        print(f"... processing {len(list_loc)} unique source locations")
    if pool is not None:
        list_hrtf_temp = pool.map(f, list_loc)
    else:
        list_hrtf_temp = [f(_) for _ in list_loc]
    h = h + np.sum(list_hrtf_temp, axis=0)
    return h, s_locations


def room_impulse_hrtf(
    src_loc=[5, 5, 5],
    head_cent=[2, 2, 2],
    head_azim=0,
    walls=[10, 10, 10],
    wtypes=[3, 3, 3, 3, 3, 3],
    sr=44100,
    c=344.5,
    dur=0.5,
    hrtf_locs=None,
    hrtf_firs=None,
    use_hrtf_symmetry=True,
    use_log_distance=False,
    use_jitter=True,
    use_highpass=True,
    pool=None,
    verbose=True,
):
    """
    Python implementation of `room_impulse_hrtf.m` by msaddler (2023/07).
    """
    src_loc = np.array(src_loc, dtype=float)
    head_cent = np.array(head_cent, dtype=float)
    head_azim = np.array(head_azim, dtype=float)
    assert (
        hrtf_locs.shape[0] == hrtf_firs.shape[0]
    ), "hrtf_locs.shape[0] != hrtf_firs.shape[0]"
    hrtf_delay = (np.sqrt(np.sum(np.square(src_loc - head_cent))) / c) * np.ones(
        (hrtf_locs.shape[0],)
    )

    # Frequency-dependent reflection coefficients for each wall
    fgains = np.zeros((6, 6), dtype=float)
    for itr_wall, material in enumerate(wtypes):
        freq, alpha = acoeff_hrtf(material=material, verbose=verbose)
        fgains[itr_wall, :] = np.sqrt(1 - alpha)
    # True when fgains is frequency-independent
    uniform_walls = len(np.unique(fgains)) == 1

    nfreq = freq / sr  # Frequencies as a fraction of sampling rate
    ntaps = int(sr * dur)  # Number of taps in output BRIR

    """
    Part I: Initialization
    """
    # Center tap of lowpass to create non-integer delay impulse (as in Peterson)
    ctap = 11
    if uniform_walls:
        ctap2 = 1  # If walls are uniform, use a single-tap filter to scale gain
    else:
        ctap2 = 33  # For frequency-dependent wall reflections, use a longer filter

    # Convert measured HRTF locations into room (xyz) coordinates
    # (and log distance locations)
    hrtf_locs_xyz = np.ones_like(hrtf_locs)
    r = hrtf_locs[:, 0]
    hrtf_locs_xyz[:, 0] = (
        r
        * np.cos(np.deg2rad(hrtf_locs[:, 1] + head_azim))
        * np.cos(np.deg2rad(hrtf_locs[:, 2]))
    )
    hrtf_locs_xyz[:, 1] = (
        r
        * -np.sin(np.deg2rad(hrtf_locs[:, 1] + head_azim))
        * np.cos(np.deg2rad(hrtf_locs[:, 2]))
    )
    hrtf_locs_xyz[:, 2] = r * np.sin(np.deg2rad(hrtf_locs[:, 2]))
    hrtf_locs_xyz = hrtf_locs_xyz + head_cent.reshape([1, -1])
    hrtf_locs_xyz_logdist = np.ones_like(hrtf_locs)
    r = np.log(hrtf_locs[:, 0]) - np.log(0.05)
    hrtf_locs_xyz_logdist[:, 0] = (
        r
        * np.cos(np.deg2rad(hrtf_locs[:, 1] + head_azim))
        * np.cos(np.deg2rad(hrtf_locs[:, 2]))
    )
    hrtf_locs_xyz_logdist[:, 1] = (
        r
        * -np.sin(np.deg2rad(hrtf_locs[:, 1] + head_azim))
        * np.cos(np.deg2rad(hrtf_locs[:, 2]))
    )
    hrtf_locs_xyz_logdist[:, 2] = r * np.sin(np.deg2rad(hrtf_locs[:, 2]))
    hrtf_locs_xyz_logdist = hrtf_locs_xyz_logdist + head_cent.reshape((1, -1))

    # Calculate the number of lead zeros to strip
    idx_min = np.argmin(
        np.sqrt(np.sum(np.square(src_loc.reshape((1, -1)) - hrtf_locs_xyz), axis=1))
    )
    src_mloc = hrtf_locs_xyz[idx_min, :]  # Nearest measured loc or direct path
    rel_dist = np.linalg.norm(src_loc - head_cent, 2) - np.linalg.norm(
        src_mloc - head_cent, 2
    )
    lead_zeros = hrtf_delay[idx_min] + np.floor(sr * rel_dist / c)

    # Initialize output matrix (will later truncate to exactly ntaps in length)
    h = np.zeros((ntaps + ctap + ctap2 + hrtf_firs.shape[1], 2), dtype=float)

    """
    Part II: determine source image locations and corresponding impulse
    response contribution from each source.  To speed up process yet ease
    the computational burden, for every 10000 source images, break off and
    determine impulse response.

    The algorithm for determining source images is as follows:
    1. Calculate maximum distance which provides relevant sources
        (i.e., those that arrive within the imp_resp duration)
    2. By looping through the X dimension, generate images of
        the (0,0,0) corner of the room, restricting the
        distance below the presecribed level.
    3. Use the coordinates of each (0,0,0) image to generate 8
        source images
    4. Generate corresponding number of reflections from each wall
        for each source image.
    """
    # Maximum source distance to be in impulse response
    dmax = np.ceil((ntaps + lead_zeros) * c / sr + np.max(walls))
    # Initialize locations matrix
    s_locations = np.ones((20000, 3), dtype=float)
    # Initialize reflections matrix
    s_reflections = np.ones((20000, 6), dtype=float)
    # Vector to get locations from the (0, 0, 0) corner images
    src_pts = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ],
        dtype=float,
    ) * src_loc.reshape((1, -1))
    Nx = np.ceil(dmax / (2 * walls[0]))  # Appropriate number of (0, 0, 0)
    loc_num = 0
    for nx in np.arange(Nx, -1, -1, dtype=int):
        if nx < Nx:
            ny = int(
                np.ceil(
                    np.sqrt(np.square(dmax) - np.square(nx * 2 * walls[0]))
                    / (2 * walls[1])
                )
            )
            nz = int(
                np.ceil(
                    np.sqrt(np.square(dmax) - np.square(nx * 2 * walls[0]))
                    / (2 * walls[2])
                )
            )
        else:
            ny = 0
            nz = 0
        X = nx * np.ones(
            ((2 * ny + 1) * (2 * nz + 1), 1), dtype=float
        )  # Form images of (0,0,0)
        Y = np.matmul(
            np.arange(-ny, ny + 1, dtype=float).reshape((-1, 1)),
            np.ones((1, 2 * nz + 1), dtype=float),
        ).reshape((-1, 1))
        Z = np.matmul(
            np.ones((2 * ny + 1, 1), dtype=float),
            np.arange(-nz, nz + 1, dtype=float).reshape((1, -1)),
        ).reshape((-1, 1))
        if nx != 0:
            # If nx !=0, do both +nx and -nx
            X = np.concatenate([-X, X], axis=0)  # Images of (0, 0, 0)
            Y = np.concatenate([Y, Y], axis=0)
            Z = np.concatenate([Z, Z], axis=0)
        Xw = 2 * walls[0] * X
        Yw = 2 * walls[1] * Y
        Zw = 2 * walls[2] * Z

        # For each image of (0, 0, 0), get the 8 source images and number of
        # reflections at each wall
        for k in range(8):
            s_refs = np.zeros((X.shape[0], 6), dtype=float)
            s_locs = np.concatenate([Xw, Yw, Zw], axis=1) + src_pts[k, :].reshape(
                (1, -1)
            )
            s_refs[:, 0:1] = (src_pts[k, 0] > 0) * np.abs(X) + (
                src_pts[k, 0] < 0
            ) * np.abs(X - 1)
            s_refs[:, 1:2] = np.abs(X)
            s_refs[:, 2:3] = (src_pts[k, 1] > 0) * np.abs(Y) + (
                src_pts[k, 1] < 0
            ) * np.abs(Y - 1)
            s_refs[:, 3:4] = np.abs(Y)
            s_refs[:, 4:5] = (src_pts[k, 2] > 0) * np.abs(Z) + (
                src_pts[k, 2] < 0
            ) * np.abs(Z - 1)
            s_refs[:, 5:6] = np.abs(Z)

            while (loc_num + s_locs.shape[0]) > 20000:
                m = 20000 - loc_num
                s_locations[slice(loc_num, loc_num + m), :] = s_locs[slice(0, m), :]
                s_reflections[slice(loc_num, loc_num + m), :] = s_refs[slice(0, m), :]
                # Get impulse response contributions
                h, s_locations = impulse_generate_hrtf(
                    h=h,
                    head_cent=head_cent,
                    head_azim=head_azim,
                    s_locations=s_locations,
                    s_reflections=s_reflections,
                    hrtf_locs=hrtf_locs,
                    hrtf_locs_xyz=hrtf_locs_xyz,
                    hrtf_locs_xyz_logdist=hrtf_locs_xyz_logdist,
                    hrtf_firs=hrtf_firs,
                    hrtf_delay=hrtf_delay,
                    sr=sr,
                    c=c,
                    ntaps=ntaps,
                    ctap=ctap,
                    ctap2=ctap2,
                    fgains=fgains,
                    nfreq=nfreq,
                    lead_zeros=lead_zeros,
                    use_hrtf_symmetry=use_hrtf_symmetry,
                    use_log_distance=use_log_distance,
                    use_jitter=use_jitter,
                    pool=pool,
                    verbose=verbose,
                )
                loc_num = 0  # Reset loc_num counter and continue
                s_locs = s_locs[slice(m, s_locs.shape[0]), :]
                s_refs = s_refs[slice(m, s_refs.shape[0]), :]

            s_locations[slice(loc_num, loc_num + s_locs.shape[0]), :] = s_locs
            s_reflections[slice(loc_num, loc_num + s_refs.shape[0]), :] = s_refs
            loc_num = loc_num + s_locs.shape[0]

    # When all locations have been generated, process the final ones
    s_locations = s_locations[0:loc_num, :]
    s_reflections = s_reflections[0:loc_num, :]
    h, s_locations = impulse_generate_hrtf(
        h=h,
        head_cent=head_cent,
        head_azim=head_azim,
        s_locations=s_locations,
        s_reflections=s_reflections,
        hrtf_locs=hrtf_locs,
        hrtf_locs_xyz=hrtf_locs_xyz,
        hrtf_locs_xyz_logdist=hrtf_locs_xyz_logdist,
        hrtf_firs=hrtf_firs,
        hrtf_delay=hrtf_delay,
        sr=sr,
        c=c,
        ntaps=ntaps,
        ctap=ctap,
        ctap2=ctap2,
        fgains=fgains,
        nfreq=nfreq,
        lead_zeros=lead_zeros,
        use_hrtf_symmetry=use_hrtf_symmetry,
        use_log_distance=use_log_distance,
        use_jitter=use_jitter,
        pool=pool,
        verbose=verbose,
    )

    """
    Part III: Finalize output
    """
    if use_highpass:
        # Highpass filter if desired
        b, a = scipy.signal.butter(2, 0.005, btype="high")
        h = scipy.signal.lfilter(b, a, h, axis=0)
    # Restrict to `ntaps` in length
    hout = h[:ntaps, :]
    return hout, lead_zeros


def is_valid_position(point, walls, buffer=0):
    """
    Helper function to check if a position is inside the room walls.
    """
    assert len(point) == len(walls)
    for p, w in zip(point, walls):
        if p < buffer or p > w - buffer:
            return False
    return True


def get_brir(
    room_materials=[26, 26, 26, 26, 26, 26],
    room_dim_xyz=[10, 10, 3],
    head_pos_xyz=[5, 5, 1.5],
    head_azim=0,
    src_azim=0,
    src_elev=0,
    src_dist=1.4,
    buffer=0,
    sr=44100,
    c=344.5,
    dur=0.5,
    hrtf_locs=None,
    hrtf_firs=None,
    use_hrtf_symmetry=True,
    use_log_distance=False,
    use_jitter=True,
    use_highpass=True,
    incorporate_lead_zeros=True,
    processes=8,
    strict=True,
    verbose=1,
):
    """
    Main function to generate binaural room impulse response (BRIR) from
    a room description, a listener position, and a source position.
    """
    if (hrtf_locs is None) or (hrtf_firs is None):
        hrtf_locs, hrtf_firs, hrtf_sr = load_kemar_hrtfs()
        assert sr == hrtf_sr, "sampling rate does not match kemar_hrtfs"
        assert use_hrtf_symmetry, "kemar_hrtfs require use_hrtf_symmetry=True"
        if verbose:
            msg = "[get_brir] loaded kemar_hrtfs (Gardner & Martin, 1994): {}"
            print(msg.format(hrtf_firs.shape))
    msg = "room_materials shape: [wall_x0, wall_x, wall_y0, wall_y, floor, ceiling]"
    assert len(room_materials) == 6, msg
    room_dim_xyz = np.array(room_dim_xyz)
    msg = "room_dim_xyz shape: [x_len (length), y_len (width), z_len (height)]"
    assert room_dim_xyz.shape == (3,), msg
    head_pos_xyz = np.array(head_pos_xyz)
    msg = "head_pos_xyz shape: [x_head, y_head, z_head]"
    assert head_pos_xyz.shape == (3,), msg
    src_pos_xyz = np.array(
        [
            src_dist
            * np.cos(np.deg2rad(src_elev))
            * np.cos(np.deg2rad(src_azim + head_azim))
            + head_pos_xyz[0],
            src_dist
            * np.cos(np.deg2rad(src_elev))
            * np.sin(np.deg2rad(src_azim + head_azim))
            + head_pos_xyz[1],
            src_dist * np.sin(np.deg2rad(src_elev)) + head_pos_xyz[2],
        ]
    )
    if strict:
        assert is_valid_position(
            head_pos_xyz, room_dim_xyz, buffer=buffer
        ), "Invalid head position"
        assert is_valid_position(
            src_pos_xyz, room_dim_xyz, buffer=buffer
        ), "Invalid source position"
    if verbose:
        print(
            "[get_brir] head_pos: {}, src_pos: {}, room_dim: {}".format(
                head_pos_xyz.tolist(), src_pos_xyz.tolist(), room_dim_xyz.tolist()
            )
        )
    t0 = time.time()
    with multiprocessing.Pool(processes=processes) as pool:
        h_out, lead_zeros = room_impulse_hrtf(
            src_loc=src_pos_xyz,
            head_cent=head_pos_xyz,
            head_azim=-head_azim,  # convention is positive azimuth = clockwise
            walls=room_dim_xyz,
            wtypes=room_materials,
            sr=sr,
            c=c,
            dur=dur,
            hrtf_locs=hrtf_locs,
            hrtf_firs=hrtf_firs,
            use_hrtf_symmetry=use_hrtf_symmetry,
            use_log_distance=use_log_distance,
            use_jitter=use_jitter,
            use_highpass=use_highpass,
            pool=pool,
            verbose=verbose,
        )
    if verbose:
        print(f"[get_brir] time elapsed: {time.time() - t0} seconds")
    if incorporate_lead_zeros:
        lead_zeros = int(np.round(lead_zeros))
        if verbose:
            print(f"[get_brir] incorporated {lead_zeros} leading zeros")
        if lead_zeros >= 0:
            h_out = np.pad(h_out, ((lead_zeros, 0), (0, 0)))
            brir = h_out[: int(dur * sr)]
        else:
            h_out = np.pad(h_out, ((0, -lead_zeros), (0, 0)))
            brir = h_out[-lead_zeros:]
    else:
        brir = h_out
    return brir
