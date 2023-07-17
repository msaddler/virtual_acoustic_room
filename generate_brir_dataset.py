import os
import sys
import time
import resource
import h5py
import numpy as np
import pandas as pd

import simulator


def get_display_str(itr, n_itr, n_skip=0, t_start=None):
    """
    Returns display string to print BRIR generation time and memory usage
    """
    disp_str = '| example: {:08d} of {:08d} |'.format(itr, n_itr)
    disp_str += ' mem: {:06.3f} GB |'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    if t_start is not None:
        time_per_signal = (time.time() - t_start) / (itr + 1 - n_skip) # Seconds per signal
        time_remaining = (n_itr - itr) * time_per_signal / 60.0 # Estimated minutes remaining
        disp_str += ' time_per_example: {:06.2f} sec |'.format(time_per_signal)
        disp_str += ' time_remaining: {:06.0f} min |'.format(time_remaining)
    return disp_str


def main(df, fn, processes=8):
    """
    Main function for iterating over manifest in parallel, generating BRIRs,
    and writing outputs to hdf5 files
    """
    N = len(df)
    sr = df.iloc[0].sr
    dur = df.iloc[0].dur
    hrtf_locs, hrtf_firs, hrtf_sr = simulator.load_kemar_hrtfs(npz_filename='kemar_hrtfs/hrtfs.npz')
    assert sr == hrtf_sr, "sampling rate does not match kemar_hrtfs"
    if not os.path.exists(fn):
        with h5py.File(fn, 'w-') as f:
            f.create_dataset('brir', shape=[N, int(sr * dur), 2], dtype=float)
            f.create_dataset('index_brir', shape=[N], dtype=int, fillvalue=0)
            f.create_dataset('index_room', shape=[N], dtype=int, fillvalue=0)
            f.create_dataset('src_dist', shape=[N], dtype=float, fillvalue=0)
            f.create_dataset('src_azim', shape=[N], dtype=float, fillvalue=0)
            f.create_dataset('src_elev', shape=[N], dtype=float, fillvalue=0)
            f.create_dataset('sr', shape=[N], dtype=int, fillvalue=0)
    print(f"[generate_bir_dataset] {fn}")
    for k in df.columns:
        print(f"|__ {k}: {df.iloc[0][k]}")
    print(f"[generate_bir_dataset] {fn}")
    with h5py.File(fn, 'r+') as f:
        n_skip = 0
        t_start = time.time()
        for itr in range(N):
            dfi = df.iloc[itr]
            assert dfi.index_brir == itr
            if f['sr'][itr] == 0:
                kwargs_get_brir = dict(dfi)
                index_brir = kwargs_get_brir.pop('index_brir')
                index_room = kwargs_get_brir.pop('index_room')
                np.random.seed((index_room * N) + index_brir)
                brir = simulator.get_brir(
                    **kwargs_get_brir,
                    hrtf_locs=hrtf_locs,
                    hrtf_firs=hrtf_firs,
                    processes=processes,
                    strict=True,
                    verbose=0)
                f['brir'][itr] = brir
                f['index_brir'][itr] = index_brir
                f['index_room'][itr] = index_room
                f['src_dist'][itr] = dfi.src_dist
                f['src_azim'][itr] = dfi.src_azim
                f['src_elev'][itr] = dfi.src_elev
                f['sr'][itr] = dfi.sr
                print(get_display_str(itr, n_itr=N, n_skip=n_skip, t_start=t_start))
            else:
                n_skip = n_skip + 1
    print(f"[END]: {fn}")
    return


if __name__ == "__main__":
    index_room = int(sys.argv[1])
    fn_manifest_brir = "/om2/user/msaddler/spatial_audio_pipeline/assets/brir/v00/manifest_brir.pdpkl"
    df_manifest_brir = pd.read_pickle(fn_manifest_brir)
    df = df_manifest_brir[df_manifest_brir.index_room == index_room]
    assert len(df) > 0, f"Found no matching BRIRs for index_room={index_room}"
    fn = os.path.join(os.path.dirname(fn_manifest_brir), 'room{:04.0f}.hdf5'.format(index_room))
    if 'SLURM_CPUS_ON_NODE' in os.environ:
        processes = int(os.environ['SLURM_CPUS_ON_NODE'])
        print(f'Set processes=SLURM_CPUS_ON_NODE={processes}')
    else:
        processes = 8
    main(df, fn, processes=processes)
