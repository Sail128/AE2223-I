import os
import numpy as np
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import warnings
import pandas as pd

from divergence import divergence
from e_functions import *

Testing = True
TestMultithread = False
Plotting = False


def getFileList(parent_folder: str):
    """finds all the relevant files which are required and cleans the names to be used in processing.

    Arguments:
        parent_folder {str} -- [description]

    Returns:
        [type] -- [description]
    """

    dirs = os.listdir(parent_folder)
    FileList = {}
    for dir in dirs:
        files = os.listdir("{}/{}".format(parent_folder, dir))
        sets = set()
        for file in files:
            if file.endswith(".dat"):
                keys = file.split("_")
                if parent_folder.split("/")[-1] == "C0_3":
                    sets.add("_".join(keys[0:6]))
                else:
                    sets.add("_".join(keys[0:4]))
        FileList[dir] = list(sets)
    return FileList


def L2error(setKey: str):
    """
    calculates the L2 error for a given expermient.
    https://www.rocq.inria.fr/modulef/Doc/GB/Guide6-10/node21.html

    Arguments:
        setKey {str} -- [description]

    Returns:
        [tuple] -- the tuple of the results
    """
    # Get the specific key for the file in the style "K_#_N_#"
    key = setKey.split("/")[-1].split("_")
    K = int(key[1])
    N = int(key[3])
    h = 2/K
    # Get all the related files to start processing
    xs = np.genfromtxt("{}_xif.dat".format(setKey))
    ys = np.genfromtxt("{}_etaf.dat".format(setKey))
    ux = np.genfromtxt("{}_ux.dat".format(setKey))
    uy = np.genfromtxt("{}_uy.dat".format(setKey))
    phi = np.genfromtxt("{}_phi.dat".format(setKey))
    # TODO: divq
    # print("{}/cont_data/K_{}_N_{}_c_0{}_.dat".format("/".join(setKey.split("/")[0:-1]),K,N,setKey.split("/")[1][-1]))
    try:
        divq = np.genfromtxt("{}/cont_data/K_{}_N_{}_c_0{}_divq.dat".format(
            "/".join(setKey.split("/")[0:-1]), K, N, setKey.split("/")[1][-1]))
    except OSError as e:
        divq = np.zeros(xs.shape)

    try:
        w_h = np.genfromtxt("{}_w_h.dat".format(setKey))
    except OSError as e:
        w_h = np.ones(xs.shape)

    # calculate the error and norm of the error for Phi both the L1 and L2 norm and relative norm
    phi_e = np.vectorize(phi_exact)(xs, ys)
    phi_error = phi_e-phi

    l2phi = np.sqrt(np.sum(w_h*phi_error*phi_error))
    relL2phi = np.sqrt(np.sum(w_h*phi_error*phi_error)) / \
        np.linalg.norm(phi_e, ord=2)
    # l1phi = np.sum(w_h*np.abs(phi_error))
    # relL1phi = np.sum(w_h*np.abs(phi_error))/np.sum(np.abs(phi_e))

    # Calculate the error and norm for U
    ue = np.vectorize(u_abs)(xs, ys)
    u = np.sqrt(ux*ux+uy*uy)
    u_error = ue-u
    l2u = np.sqrt(np.sum(w_h*u_error*u_error))
    relL2u = np.sqrt(np.sum(w_h*u_error*u_error)) / \
        np.linalg.norm(ue, ord=2)

    # Calculate the error and norm for div(u)-f_exact. should be 0
    l2divu_f = np.sqrt(np.sum(w_h*divq*divq))

    return [K, N, h,
            l2phi, relL2phi,
            l2u, relL2u,
            l2divu_f]


def executeParallel(inputs, threadFunction, threads=8):
    """Simple paralell execution setup, with progressbar

    Arguments:
        setKeys {list} -- list of inputs to be mapped to the thread function

    Keyword Arguments:
        threads {int} -- number of threads to execute on (default: {8})
        threadFunction {function} -- function to execute on the thread with input from the list

    Returns:
        {list} -- list of results returned from the function
    """

    pool = ThreadPool(threads)
    result = list(tqdm(pool.imap(threadFunction, inputs), total=len(inputs)))
    pool.close()
    pool.join()
    # print(result)
    result = pd.DataFrame(
        data=result, columns=["K", "N", "h", "l2phi",
                              "relL2phi", "l2u", "relL2u", "l2divu_f"]
    )
    # Sort the dataframe on element size and polynomial degree
    result.sort_values(["h", "N"], ascending=[True, True], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def main(C):
    # Define the parent directory of the data to be processed.
    # Assumed structure is: parent_dir: exp_1,exp_2,...,exp_N (each containg the experiment files)
    parent_dir = "Data/{}".format(C)
    save_dir = "test_error"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    print(FileList.keys())
    experiments = list(FileList.keys())  # remove slice to get full files
    print(experiments)
    for exp in experiments:
        # generate the map of inputs for calculating the error
        input_map = list(map(
            (lambda x: "{}/{}/{}".format(parent_dir, exp, x)), FileList[exp]))  # remove slice to get everything
        print("Starting analysis of experiment {}".format(exp))
        print(input_map[0:3])
        # calculte the error in a multithreaded way
        results = executeParallel(input_map, L2error)
        # write to file
        results.to_csv(
            "{}/{}_errors_{}.dat".format(save_dir, exp, C), index=False)

# Test code for debugging purposes


def test(C):
        # Define the parent directory of the data to be processed.
    # Assumed structure is: parent_dir: exp_1,exp_2,...,exp_N (each containg the experiment files)
    parent_dir = "Data/{}".format(C)
    save_dir = "test_error"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    print(FileList.keys())
    experiments = list(FileList.keys())  # remove slice to get full files
    print(experiments)
    for exp in experiments[2:3]:
        # generate the map of inputs for calculating the error
        input_map = list(map(
            (lambda x: "{}/{}/{}".format(parent_dir, exp, x)), FileList[exp]))  # remove slice to get everything
        print("Starting analysis of experiment {}".format(exp))
        print(input_map[0:3])
        # calculte the error in a multithreaded way
        results = executeParallel(input_map, L2error)
        # write to file
        results.to_csv(
            "{}/{}_errors_{}.dat".format(save_dir, exp, C), index=False)


if __name__ == "__main__":
    if Testing:
        test("C0_0")
    else:
        main("C0_0")
        main("C0_3")
