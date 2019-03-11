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

def phi_exact_calc(x,y):
    phi_exact = np.sin(np.pi*x) * np.sin(np.pi*y)
    return  phi_exact

def u_exact_calc(x,y):
    u_exact_i = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
    u_exact_j = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
    return u_exact_i, u_exact_j

def f_exact_calc(x,y):
    f_exact_i, f_exact_j = -np.pi * np.pi * np.sin(np.pi*x) * np.sin(np.pi*y)
    return f_exact_i, f_exact_j

def L2error(setKey):
    pass


def executeParallel(setKey, threads = 8):
    pool = ThreadPool(threads)
    result = pool.map(L2error,setKey)
    pool.close()
    pool.join()
    return setKey, result

def main():
    FileList = getFileList("Data")
    print(FileList)



if __name__ == "__main__":
    main()
