import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help = "filename")
    args = parser.parse_args()
    f = args.file

    times = h5py.File(f, 'r')['times'][...]
    corrs = h5py.File(f, 'r')['corrs'][...]
    plt.loglog(times, corrs)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$|C(T,x=0,t)|$')
    plt.show()
