import numpy as np
import torch as pt
import pickle
import lzma
from src.data_encoding import encode_res, all_resnames, encode_location
from glob import glob

import os

def read_fasta(ifile):
    seq = ""
    with open(ifile, 'r') as iFile:
        for i in iFile:
            if i[0] == ">":
                ids = i.strip().split("|")[2].split(" ")[0]
            else:
                j = i.strip()
                seq += j
    return seq, ids


class SeqDataset(pt.utils.data.Dataset):
    def __init__(self, filepath, nres_max):
        super(SeqDataset, self).__init__()
        with lzma.open(filepath, "r") as f:
            self.combined_data = pickle.load(f)
        self.ids = np.array(self.combined_data['ids'])
        self.seq = np.array(self.combined_data['seq'])
        self.loc = np.array(self.combined_data['loc'])
        self.nres_max = nres_max

        n1 = len(self.ids)
        n2 = len(self.seq)
        n3 = len(self.loc)

        if (n1 != n2) or (n1 != n3):
            raise ValueError("Different size of ids/seq/loc")

        # filter the seq greater than nres_max
        lengths = []
        for i in self.seq:
            lengths.append(len(i))
        lengths = np.array(lengths)
        length_mask = lengths <= self.nres_max
        self.ids = self.ids[length_mask]
        self.seq = self.seq[length_mask]
        self.loc = self.loc[length_mask]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # seq_enc = pt.unsqueeze(encode_res(self.seq[index], self.nres_max), dim=1)
        seq_enc = encode_res(self.seq[index], self.nres_max)
        loc_enc = pt.tensor(encode_location(self.loc[index]), dtype=pt.float64)
        return seq_enc, loc_enc

class FastaDataset(pt.utils.data.Dataset):

    def __init__(self, folder, nres_max):
        self.sdata = []
        self.iddata = []
        self.paths = glob(os.path.join(folder, "*"))
        self.nresmax = nres_max
        for i in self.paths:
            S, I = read_fasta(i)
            self.sdata.append(S)
            self.iddata.append(I)

        self.sdata = np.array(self.sdata)
        self.iddata = np.array(self.iddata)

        # filter the seq greater than nres_max
        lengths = []
        for i in self.sdata:
            lengths.append(len(i))
        lengths = np.array(lengths)
        length_mask = lengths <= self.nresmax
        self.sdata = list(self.sdata[length_mask])
        self.iddata = list(self.iddata[length_mask])

        self.sdata_enc = []
        for i in self.sdata:
            self.sdata_enc.append(encode_res(i, self.nresmax))

    def __len__(self):
        return len(self.sdata_enc)

    def __getitem__(self, index):
        i = self.iddata[index]
        s = self.sdata_enc[index]

        return i, s

    def getall(self):
        return self.iddata, pt.stack(self.sdata_enc)


