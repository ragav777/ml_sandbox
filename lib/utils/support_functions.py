#########################################################################################################
#  Description: Collection of support functions that'll be used often
#
#########################################################################################################
import numpy as np
import pandas as pd
import random
import os

#########################################################################################################
__author__ = 'DataCentric1'
__pass__ = 1
__fail__ = 0

#########################################################################################################

# Class to specify color and text formatting for prints
class Color:
    def __init__(self):
        pass

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


#  Returns number of lines in a file in a memory / time efficient way
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


#  Save numpy array from .npy file to txt file
def save_npy_array_to_txt(npy_fname, txt_fname):

    np.savetxt(txt_fname, np.load(npy_fname), fmt='%s')

    return __pass__


#  Save numpy array from .npy file to csv file. TODO - Double check fn
def save_npy_array_to_csv(npy_fname, csv_fname):

    temp_array = np.load(npy_fname)

    index_row, index_col = temp_array.shape

    print index_row
    print index_col

    f = open(csv_fname, 'w')

    for i in range(index_row):
        f.write(temp_array[i, 0])
        f.write(",")
        f.write(temp_array[i, 1])
        f.write("\n")

    f.close()

    return __pass__


# Returns random floating point value within the range specified
def random_float(low, high):
    return random.random()*(high-low) + low


# Returns all elements in the list with format 0.2f
def format_float_0_2f(list_name):
    return "["+", ".join(["%.2f" % x for x in list_name])+"]"


# Load Model Data for a CSV
def load_model_data(data_csv='dummy.csv'):
    """
    Reads a CSV file, Returns a Pandas Data Frame

    :param data_csv:
    :return dataproc:
    """
    if os.path.isfile(data_csv):
        data = pd.read_csv(data_csv, sep=',')
        return data
    else:
        raise ValueError('Input file %s not available', data_csv)
