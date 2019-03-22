import os
import numpy as np
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import warnings
import pandas as pd

from divergence import divergence
from e_functions import *

Testing = False
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
    h = 2/K
    if Plotting:
        print(setKey)
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
    phi_e = np.vectorize(
        phi_exact)(xs, ys)
    phi_error = phi_e-phi

    l2phi = np.sqrt(np.sum(w_h*phi_error*phi_error))
    relL2phi = np.sqrt(np.sum(w_h*phi_error*phi_error)) / \
        np.linalg.norm(phi_e, ord=2)
    # l1phi = np.sum(w_h*np.abs(phi_error))
    # relL1phi = np.sum(w_h*np.abs(phi_error))/np.sum(np.abs(phi_e))

    # Calculate the error and norm for U
    ux_exact = np.vectorize(u_exact_x)(xs, ys)
    uy_exact = np.vectorize(u_exact_y)(xs, ys)
    ux_error = ux_exact - ux
    uy_error = uy_exact - uy

    l2ux = np.sqrt(np.sum(w_h*ux_error*ux_error))
    relL2ux = np.sqrt(np.sum(w_h*ux_error*ux_error)) / \
        np.linalg.norm(ux_exact, ord=2)
    # l1ux = np.sum(w_h*np.abs(ux_error))
    # relL1ux = np.sum(w_h*np.abs(ux_error))/np.sum(np.abs(ux_exact))

    l2uy = np.sqrt(np.sum(w_h*uy_error*uy_error))
    relL2uy = np.sqrt(np.sum(w_h*uy_error*uy_error)) / \
        np.linalg.norm(uy_exact, ord=2)
    # l1uy = np.sum(w_h*np.abs(uy_error))
    # relL1uy = np.sum(w_h*np.abs(uy_error))/np.sum(np.abs(uy_exact))

    # Calculate the error and norm for div(u)-f_exact. should be 0
    dx = (np.abs(xs[0, 0]-xs[-1, -1]))/(xs.shape[0]-1)
    dy = (np.abs(ys[0, 0]-ys[-1, -1]))/(ys.shape[1]-1)
    f_e = np.vectorize(f_exact)(xs,ys)
    divu = divergence([ux, uy], di=[dx,dy])
    Zero_error = divu-f_e
    l2divu_f = np.linalg.norm(Zero_error,ord=2)
    if Plotting:
        print(np.sum(Zero_error))
        print(l2divu_f)
        print(divu)
        print(f_e)
    
    if Plotting:
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        map1 = ax1.contourf(xs, ys, divu, 50, cmap=cm.plasma)
        fig1.colorbar(map1)
        ax1.contour(xs, ys, divu, 10, colors='k',
                    linewidths=1, linestyles='solid')
        ax1.quiver(xs, ys, ux, uy)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        map2 = ax2.contourf(xs, ys, Zero_error, 50, cmap=cm.plasma)
        fig2.colorbar(map2)
        
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        map3 = ax3.contourf(xs, ys, divu, 50, cmap=cm.plasma)
        fig3.colorbar(map3)

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        map4 = ax4.contourf(xs, ys, f_e, 50, cmap=cm.plasma)
        fig4.colorbar(map4)
        plt.show()

    return [    K, N, h, 
                l2phi, relL2phi,
                np.sqrt(l2ux**2+l2uy**2), np.sqrt(relL2ux**2+relL2uy**2),
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
        data=result, columns=["K", "N", "h", "l2phi", "relL2phi", "l2u", "relL2u", "l2divu_f"]
    )
    # Sort the dataframe on element size and polynomial degree
    result.sort_values(["h", "N"], ascending=[True, True], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def sortfunction(x):
    """small function to be able to sort the list in order

    Arguments:
        x {[type]} -- value to sort

    Returns:
        [type] -- sorting key
    """
    if(x[1] < 10):
        y = "0"+str(x[1])
    return float(str(x[0])+"." + y[3])


def main(C):
    # Define the parent directory of the data to be processed.
    # Assumed structure is: parent_dir: exp_1,exp_2,...,exp_N (each containg the experiment files)
    parent_dir = "Data/{}".format(C)
    save_dir = "test"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    print(FileList.keys())
    experiments = list(FileList.keys()) # remove slice to get full files
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

def test(C):
    parent_dir = "Data/{}".format(C)
    save_dir = "test"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    experiments = list(FileList.keys()) # remove slice to get full files
    print(experiments)
    if input("test multithread y/n: ") == "y":
        for exp in experiments[0:2]:
            # generate the map of inputs for calculating the error
            input_map = list(map(
                (lambda x: "{}/{}/{}".format(parent_dir, exp, x)), FileList[exp][0:15]))  # remove slice to get everything
            print("Starting analysis of experiment {}".format(exp))
            print(input_map[0:3])
            # calculte the error in a multithreaded way
            results = executeParallel(input_map, L2error)
            # write to file
            print(results.to_csv(index=False))
    else:
        global Plotting
        Plotting=True
        #print(L2error("{}/{}/{}".format(parent_dir, experiments[0], FileList[experiments[0]][0])))
        print(L2error("{}/{}/{}".format(parent_dir, experiments[0], "K_10_N_5")))

if __name__ == "__main__":
    if Testing:
        test("C0_0")
    else:
        main("C0_0")
        main("C0_3")
