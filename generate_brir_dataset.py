import os
import sys
import h5py
import numpy as np
import pandas as pd

import simulator


def main(df, fn, processes=8):
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
        for itr in range(N):
            print(f"Generating BRIR {itr} of {N}")
            dfi = df.iloc[itr]
            assert dfi.index_brir == itr
            if f['sr'][itr] == 0:
                kwargs_get_brir = dict(dfi)
                index_brir = kwargs_get_brir.pop('index_brir')
                index_room = kwargs_get_brir.pop('index_room')
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
            else:
                print(f"... skipped index {itr} (already exists)")
    print(f"[END]: {fn}")
    return


if __name__ == "__main__":
    fn_manifest_brir = "/om2/user/msaddler/spatial_audio_pipeline/assets/brir/v00/manifest_brir.pdpkl"
    df_manifest_brir = pd.read_pickle(fn_manifest_brir)
    index_room = 0
    df = df_manifest_brir[df_manifest_brir.index_room == index_room]
    assert len(df) > 0, f"Found no matching BRIRs for index_room={index_room}"
    
    fn = os.path.join(os.path.dirname(fn_manifest_brir), 'room{:04.0f}.hdf5'.format(index_room))
    main(df, fn, processes=8)
