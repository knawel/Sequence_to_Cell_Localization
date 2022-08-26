import re

import numpy as np
import pickle
import lzma
import gzip
from tqdm import tqdm
import sys

from src.data_encoding import encode_location, encode_res


config_dataset = {
    # input filepaths
    "data_filepaths": "./data/uniprot_trembl_human.dat.gz",
    # output filepath
    "dataset_filepath": "./data/data_seq_locations.xz",
}


def init_flags():
    return {
    "new prot": True,
    "location": False,
    "sequence": False
    }


def init_record():
    return ["", "", ""]


if __name__ == "__main__":
    sys.stderr.write("Initialization... ")

    # prepare and store all data
    flags = init_flags()
    record = init_record()
    data = []

    num_lines = sum(1 for line in gzip.open(config_dataset["data_filepaths"], 'r'))
    sys.stderr.write(" ready\n")
    sys.stderr.write("Reading file\n")
    with gzip.open(config_dataset["data_filepaths"], 'rt', encoding='utf-8') as iFile:
        for i in tqdm(iFile, total=num_lines):

            # for i in iFile:
            line_start = i[:2]

            # when new protein starts
            if line_start == "//":
                if len(record[2]) > 2:
                    data.append(record)
                flags = init_flags()
                record = init_record()

            # when uniprot ID
            if line_start == "AC":
                id_ = i.split()[1]
                id_prot = id_.split(';')[0]
                record[0] += id_prot

            # when started the sequence part (till the end of record "//")
            if line_start == "SQ":
                flags['sequence'] = True

            if flags['sequence']:
                record[1] += i.strip()

            # if comment, looking for cellular location section
            if line_start == "CC":
                if "-!-" in i:
                    if "-!- SUBCELLULAR LOCATION:" in i:
                        flags["location"] = True
                    else:  # if another section started, for example SIMILARITY
                        flags["location"] = False
                if "---" in i:
                    flags["location"] = False

            if flags["location"]:
                record[2] += i.strip()

    ids_np = np.array(data)
    id_list = ids_np[:, 0]

    len_list = []
    seq_list = []
    loc_list = []

    sys.stderr.write("Extracting data\n")
    # for i in ids_np:
    for i in tqdm(ids_np, total=len(ids_np)):

        j = i[1]

        # seq
        s = j.split(';')[3]
        seq_list.append(s)

        # location
        loc1 = i[2]
        # loc4 = encode_location(loc1) # if encode
        loc4 = loc1
        loc_list.append(loc4)


    # store metadata
    sys.stderr.write("Writing data\n")
    data_to_write = {
        "ids": id_list,
        "seq": seq_list,
        "loc": loc_list
    }
    with lzma.open(config_dataset["dataset_filepath"], "wb") as f:
        pickle.dump(data_to_write, f)

    sys.stderr.write("Done\n")
