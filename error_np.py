import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import warnings


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
                sets.add("_".join(keys[0:4]))
        FileList[dir] = list(sets)
    return FileList


def phi_exact_calc(x: float, y: float):
    """calculates the exact potential

    Arguments:
        x {[float]} --
        y {[float]} --

    Returns:
        float -- potential
    """
    phi_exact = np.sin(np.pi*x) * np.sin(np.pi*y)
    return phi_exact


def u_exact_calc(x: float, y: float):
    """calculates the exact velocity

    Arguments:
        x {float} -- [description]
        y {float} -- [description]

    Returns:
        float -- ux, uy
    """
    u_exact_i = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
    u_exact_j = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
    return u_exact_i, u_exact_j


def u_exact_x(x: float, y: float):
    return np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)


def u_exact_y(x: float, y: float):
    return np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)


def f_exact_calc(x: float, y: float):
    f_exact_i, f_exact_j = -np.pi * np.pi * np.sin(np.pi*x) * np.sin(np.pi*y)
    return f_exact_i, f_exact_j


def L2error(setKey: str):
    """
    calculates the L2 error for a given result.
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
    # Get all the related files to start processing
    xs = np.genfromtxt("{}_xif.dat".format(setKey))
    ys = np.genfromtxt("{}_etaf.dat".format(setKey))
    ux = np.genfromtxt("{}_ux.dat".format(setKey))
    uy = np.genfromtxt("{}_uy.dat".format(setKey))
    phi = np.genfromtxt("{}_phi.dat".format(setKey))
    try:
        w_h = np.genfromtxt("{}_w_h.dat".format(setKey))
    except OSError as e:
        w_h = np.ones(xs.shape)
    

    # calculate the error and norm of the error for Phi both the L1 and L2 norm and relative norm
    phi_exact = np.vectorize(
        phi_exact_calc)(xs, ys)
    phi_error = phi_exact-phi

    l2phi = np.sqrt(np.sum(w_h*phi_error*phi_error))
    relL2phi = np.sqrt(np.sum(w_h*phi_error*phi_error)) / \
        np.linalg.norm(phi_exact, ord=2)
    l1phi = np.sum(w_h*np.abs(phi_error))
    relL1phi = np.sum(w_h*np.abs(phi_error))/np.sum(np.abs(phi_exact))

    # Calculate the error and norm for U
    ux_exact = np.vectorize(u_exact_x)(xs, ys)
    uy_exact = np.vectorize(u_exact_y)(xs, ys)
    ux_error = ux_exact - ux
    uy_error = uy_exact - uy

    l2ux = np.sqrt(np.sum(w_h*ux_error*ux_error))
    relL2ux = np.sqrt(np.sum(w_h*ux_error*ux_error)) / \
        np.linalg.norm(ux_error, ord=2)
    l1ux = np.sum(w_h*np.abs(ux_error))
    relL1ux = np.sum(w_h*np.abs(ux_error))/np.sum(np.abs(ux_error))

    l2uy = np.sqrt(np.sum(w_h*uy_error*uy_error))
    relL2uy = np.sqrt(np.sum(w_h*uy_error*uy_error)) / \
        np.linalg.norm(uy_error, ord=2)
    l1uy = np.sum(w_h*np.abs(uy_error))
    relL1uy = np.sum(w_h*np.abs(uy_error))/np.sum(np.abs(uy_error))

    return [K, N, l2phi, relL2phi, l1phi, relL1phi, np.sqrt(l2ux**2+l2uy**2), np.sqrt(relL2ux**2+relL2uy**2), np.sqrt(l1ux**2+l1uy**2), np.sqrt(relL1ux**2+relL1uy**2)]


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
    return result


def sortfunction(x):
    """small function to be able to sort the list in order

    Arguments:
        x {[type]} -- value to sort

    Returns:
        [type] -- sorting key
    """
    y = x[0].split("_")
    if(len(y[3]) == 1):
        y[3] = "0"+y[3]
    return float(y[1]+"." + y[3])


def main():
    # Define the parent directory of the data to be processed.
    # Assumed structure is: parent_dir: exp_1,exp_2,...,exp_N (each containg the experiment files)
    parent_dir = "Data"
    c = "np_c0"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    print(FileList.keys())
    experiments = list(FileList.keys())
    print(experiments)
    for exp in experiments:
        # generate the map of inputs for calculating the error
        input_map = list(map(
            (lambda x: "{}/{}/{}".format(parent_dir, exp, x)), FileList[exp]))
        # calculte the error in a multithreaded way
        results = executeParallel(input_map, L2error)
        # sort the returned map using K and N
        results.sort(key=sortfunction)
        # opening an output file
        output_file = open("{}_errors_{}.dat".format(exp, c),
                           "w+")
        # writing the header
        output_file.write(
            "#K,N,l2phi, relL2phi, l1phi, relL1phi, l2u, relL2u, l1u, relL1u \n")
        # write all the data to the file
        for line in results:
            K, N, l2phi, relL2phi, l1phi, relL1phi, l2u, relL2u, l1u,relL1u = line
            output_file.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                K, N, l2phi, relL2phi, l1phi, relL1phi, l2u, relL2u, l1u, relL1u))
        # close the file
        output_file.close()


if __name__ == "__main__":
    main()
