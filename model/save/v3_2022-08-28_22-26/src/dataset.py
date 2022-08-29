import numpy as np
import torch as pt
import pickle
import lzma
from src.data_encoding import encode_res, encode_location, selected_locations
from glob import glob
import os


def read_fasta(ifile):
    """ Take file (fasta sequence) and extract the protein name and sequence

    Parameters
    ----------
    ifile : str
        path to the file

    Return
    ------
    sequence, protein name

    """
    seq = ""
    with open(ifile, 'r') as iFile:
        for i in iFile:
            if i[0] == ">":
                ids = i.strip().split("|")[2].split(" ")[0]
            else:
                j = i.strip()
                seq += j
    return seq, ids


def get_stat_from_dataset(seq_dataset):
    """Get distribution of one-hot vectors."""
    all_locations = []
    for i in seq_dataset:
        j = i[1].numpy()
        all_locations.append(j)
    summary = np.sum(np.array(all_locations), axis=0)
    report = ""
    for i, value in enumerate(summary):
        report += f'{selected_locations[i]}: {str(int(value))}\n'
    return report

class SeqDataset(pt.utils.data.Dataset):
    """
    A class used to store Dataset for training

    Attributes
    ----------
    ids : numpy.ndarray
        list of protein ID
    seq : numpy.ndarray
        list of sequences
    loc : numpy.ndarray
        list of locations (each is string)
    nres_max : int
        the maximal length of sequence

    Methods
    -------
    get_stat()
        Return statistics of locations
    """

    def __init__(self, filepath, seq_max):
        """
        Parameters
        ----------
        filepath : str
            Path to the pre-processed file
        seq_max : int
            Maximal length of the sequence, bigger sequences will be ignored

        Raises
        ------
        NotImplementedError
            If number of sequence, ID or locations is not the same.
        """
        super(SeqDataset, self).__init__()
        with lzma.open(filepath, "r") as f:
            self.combined_data = pickle.load(f)
        self.ids = np.array(self.combined_data['ids'])
        self.seq = np.array(self.combined_data['seq'])
        self.loc = np.array(self.combined_data['loc'])
        self.seq_max = seq_max

        n1 = len(self.ids)
        n2 = len(self.seq)
        n3 = len(self.loc)

        if (n1 != n2) or (n1 != n3):
            raise NotImplementedError("Different size of ids/seq/loc")

        # filter the seq greater than seq_max
        lengths = []
        for i in self.seq:
            lengths.append(len(i))
        lengths = np.array(lengths)
        length_mask = lengths <= self.seq_max
        self.ids = self.ids[length_mask]
        self.seq = self.seq[length_mask]
        self.loc = self.loc[length_mask]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # seq_enc = pt.unsqueeze(encode_res(self.seq[index], self.seq_max), dim=1)
        seq_enc = encode_res(self.seq[index], self.seq_max)
        loc_enc = pt.tensor(encode_location(self.loc[index]), dtype=pt.float64)
        return seq_enc, loc_enc

    def get_stat(self):
        """Get distribution of one-hot vectors."""
        all_locations = []

        for i in self.loc:
            all_locations.append(encode_location(i))
        return np.sum(np.array(all_locations), axis=0)


class FastaDataset(pt.utils.data.Dataset):
    """
    A class used to evaluate examples

    Attributes
    ----------
    iddata : numpy.ndarray
        list of protein names
    sdata : numpy.ndarray
        list of sequences
    sdata : numpy.ndarray
        list of encoded sequences
    nresmax : int
        the maximal length of sequence

    Methods
    -------
    get_all()
        Return ID and encoded sequence (as tensor)
    """

    def __init__(self, folder, nres_max):
        """
        Parameters
        ----------
        iddata : numpy.ndarray
            list of protein names
        sdata : numpy.ndarray
            list of sequences
        sdata : numpy.ndarray
            list of encoded sequences
        nresmax : int
            the maximal length of sequence

        Raises
        ------
        NotImplementedError
            If number of sequence, ID or locations is not the same.
        """
        self.iddata = []
        self.sdata = []
        self.nresmax = nres_max
        for i in glob(os.path.join(folder, "*")):
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

    def get_all(self):
        """ Return ID and encoded sequence (as tensor)

        Parameters
        ----------

        Return
        ------
        protein names, sequences

        """
        return self.iddata, pt.stack(self.sdata_enc)
