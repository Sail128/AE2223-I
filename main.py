import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
# from tabulate import tabulate as tb

def getFileList(parent_folder:str):
    dirs = os.listdir(parent_folder)
    FileList = {}
    for dir in dirs:
        files = os.listdir("{}/{}".format(parent_folder,dir))
        sets = set()
        for file in files:
            if file.endswith(".dat"):
                keys = file.split("_")
                sets.add("_".join(keys[0:4]))
        FileList[dir] = sets
    return FileList

def L2error(setKey):
    pass


def executeParallel(setKey, threads = 8):
    pool = ThreadPool(threads)
    result = pool.map(L2error,dirnames)
    pool.close()
    pool.join()
    return setKey, result

def main():
    FileList = getFileList("Data")
    print(FileList)


if __name__ == "__main__":
    main()