import numpy as np
import torch as pt


# all residue names (# - fill after the last when sequence is shorter than N_res_max)
all_resnames = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '#'])

# selected_locations = ['membrane', 'junction', 'cytoplasm', 'cytoskeleton', 'vesicle',
#                       'endoplasmic reticulum', 'golgi apparatus', 'lysosome', 'mitochondrion',
#                       'nucleus', 'synapse', 'extracellular']

selected_locations = ['membrane', 'cytoplasm', 'mitochondrion', 'nucleus']


def onehot(x, v):
    m = (x.reshape(-1, 1) == np.array(v).reshape(1, -1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1, 1)], axis=1)


def encode_res(seq, max_len=1024, device=pt.device("cpu")):
    seq_nonspace = seq.replace(" ", "").ljust(max_len, '#')
    s = np.array(list(seq_nonspace))
    qr = pt.from_numpy(onehot(s, all_resnames).astype(np.float32)).to(device)
    return qr.T


def encode_location(loc_str):
    n_loc = len(selected_locations)
    b = loc_str.lower()
    on = np.zeros(n_loc, dtype=int)
    for i, l in enumerate(selected_locations):
        if l in b:
            on[i] = 1
    return on

