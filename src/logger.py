"""This file contains the logger object to print and save the debug information and results."""
import os
import pandas as pd
from time import time
from datetime import timedelta


class Logger:
    """Object to store and write debug information and results."""

    def __init__(self, log_dir, log_name, verbose=True):
        """Create logger with a log directory and an identifing name."""
        # define log filepath
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_str_filepath = os.path.join(log_dir, log_name+'.log')
        self.log_prg_filepath = os.path.join(log_dir, log_name+'progress')
        self.epoch = 0
        # define logs
        self.log_s = ''
        self.log_l = []

        # debug flag
        self.verbose = verbose

        # start timer
        self.t0 = time()
        self.ts = self.t0

    def print(self, line_raw):
        """Write string to log file and print to console if verbose active. New line is added.
        Parameters
        ----------
        line_raw : str
            line to print
        """

        # convert line to string
        line = str(line_raw)

        # update log and append to log file
        self.log_s += line + '\n'

        # update log file
        with open(self.log_str_filepath, 'a') as fs:
            fs.write(line + '\n')

        # debug print
        if self.verbose:
            print(line)

    def store_progress(self, loss: float, is_train: bool, epoch=-1):
        """Write to files the learning progress"""
        if epoch >= 0:
            self.epoch = epoch
        else:
            # update file
            if is_train:
                with open(self.log_prg_filepath + "_train.txt", 'a') as fs:
                    fs.write(f'{self.epoch};{loss:.2f}\n')
            else:
                with open(self.log_prg_filepath + "_test.txt", 'a') as fs:
                    fs.write(f'{self.epoch};{loss:.2f}\n')

