"""
This file contains code to test the algorithms and asses performance
"""
import os
import numpy as np
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import warnings
import pandas as pd
import time

import test.test.L2Error as tl2 

import numpy as np
import warnings

from divergence import divergence
from test.test.e_functions import *

multiThreaded = True
Plotting = False

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
        divq = np.genfromtxt("{}/cont_data/K_{}_N_{}_c_0{}_divq.dat".format("/".join(setKey.split("/")[0:-1]),K,N,setKey.split("/")[1][-1]))
    except OSError as e:
        print("error: ", "{}/cont_data/K_{}_N_{}_c_0{}_divq.dat".format("/".join(setKey.split("/")[0:-1]),K,N,setKey.split("/")[1][-1]))
        divq = np.zeros(xs.shape)

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
    ue = np.vectorize(u_abs)(xs,ys)
    u = np.sqrt(ux*ux+uy*uy)
    u_error = ue-u
    l2u = np.sqrt(np.sum(w_h*u_error*u_error))
    relL2u = np.sqrt(np.sum(w_h*u_error*u_error)) / \
        np.linalg.norm(ue, ord=2)

    # Calculate the error and norm for div(u)-f_exact. should be 0
    # dx = (np.abs(xs[0, 0]-xs[-1, -1]))/(xs.shape[0])
    # dy = (np.abs(ys[0, 0]-ys[-1, -1]))/(ys.shape[1])
    # f_e = np.vectorize(f_exact)(xs,ys)
    #divu = divergence([ux, uy], di=[dx,dy])
    #Zero_error = divu-f_e
    l2divu_f = np.sqrt(np.sum(w_h*divq*divq))
    #x,y = Zero_error.shape

    #f=20 #croppingfactor what fraction to remove this is due to high erros on the boundry 
    #dif_crop = Zero_error[:-int(y/f),:-int(x/f)][int(y/f):][...,int(x/f):]
    #l2divu_f_cropped = np.linalg.norm(dif_crop,ord=2)
    # x_crop = xs[:-int(y/f),:-int(x/f)][int(y/f):][...,int(x/f):]
    # y_crop = ys[:-int(y/f),:-int(x/f)][int(y/f):][...,int(x/f):]

    #Debugging code only used during debugging
    if Plotting:
        print(np.sum(divq))
        print(l2divu_f)
        # print(divu)
        # print(f_e)
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_title("divergence of u")
        map1 = ax1.contourf(xs, ys, divq, 50, cmap=cm.plasma)
        fig1.colorbar(map1)
        ax1.contour(xs, ys, divq, 10, colors='k',
                    linewidths=1, linestyles='solid')
        # ax1.quiver(xs, ys, ux, uy)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title("divu-f")
        map2 = ax2.contourf(xs, ys, divq, 50, cmap=cm.plasma)
        fig2.colorbar(map2)

        # fig5 = plt.figure()
        # ax5 = fig5.add_subplot(111)
        # ax5.set_title("divu-f cropped")
        # x_crop = xs[:-int(y/f),:-int(x/f)][int(y/f):][...,int(x/f):]
        # y_crop = ys[:-int(y/f),:-int(x/f)][int(y/f):][...,int(x/f):]
        # map5 = ax5.contourf(x_crop, y_crop, dif_crop, 50, cmap=cm.plasma)
        # fig5.colorbar(map5)

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        ax4.set_title("f exact")
        map4 = ax4.contourf(xs, ys, f_e, 50, cmap=cm.plasma)
        fig4.colorbar(map4)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_title("phi")
        map3 = ax3.contourf(xs, ys, phi, 50, cmap=cm.plasma)
        fig3.colorbar(map3)
        plt.show()

    return [    K, N, h, 
                l2phi, relL2phi,
                l2u, relL2u,
                l2divu_f]

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

def test(C):
    parent_dir = "Data/{}".format(C)
    save_dir = "test"
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    experiments = list(FileList.keys())  # remove slice to get full files
    print(experiments)
    if multiThreaded:
        global Plotting
        Plotting = False
        for exp in experiments[2:3]:
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
        #print(L2error("{}/{}/{}".format(parent_dir, experiments[0], FileList[experiments[0]][0])))
        if Plotting:
            error = L2error("{}/{}/{}".format(parent_dir, experiments[0], "K_7_N_10"))
            Plotting = False
        n = 5
        pure = []
        comp = []
        pstart = time.time_ns()
        for i in range(n):
            start = time.time_ns()
            error = L2error("{}/{}/{}".format(parent_dir, experiments[0], "K_7_N_10"))
            end = time.time_ns()
            pure.append(end-start)
        pend=time.time_ns()
        lstart=time.time_ns()
        for i in range(1):
            start = time.time_ns()
            error1 = tl2.L2error("{}/{}/{}".format(parent_dir, experiments[0], "K_7_N_10"))
            end = time.time_ns()
            comp.append(end-start)
        lend=time.time_ns()
        print( "python: {} calculated in: {}s per call and {}s overall".format(error, (sum(pure)/len(pure))*(10**-9),(pend-pstart)*(10**-9)))
        print( "build: {} calculated in: {}s per call and {}s overall".format(error1, (sum(comp)/len(comp))*(10**-9),(lend-lstart)*(10**-9)))


if __name__ == "__main__":
    test("C0_0")