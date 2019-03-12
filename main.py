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
    key = setKey.split("/")[-1]
    # Get all the related files to start processing
    xs = np.genfromtxt("{}_xif.dat".format(setKey))
    ys = np.genfromtxt("{}_etaf.dat".format(setKey))
    ux = np.genfromtxt("{}_ux.dat".format(setKey))
    uy = np.genfromtxt("{}_uy.dat".format(setKey))
    phi = np.genfromtxt("{}_phi.dat".format(setKey))
    w_h = np.genfromtxt("{}_w_h.dat".format(setKey))
    # Get the shape of the data to process (n_rows, n_columns)
    ymax, xmax = xs.shape
    # Initialize the variables
    L2_phi = 0
    L2_ux = 0
    L2_uy = 0

    L2_phi_sqr = 0
    L2_ux_sqr = 0
    L2_uy_sqr = 0

    L2_phi_abs = 0
    L2_ux_abs = 0
    L2_uy_abs = 0
    # Run all the sommations
    for j in range(ymax):
        for i in range(xmax):
            # calculating the exact velocity
            u_ext_x, u_ext_y = u_exact_calc(xs[j, i], ys[j, i])
            # calculating the exact potential
            phi_ext = phi_exact_calc(xs[j, i], ys[j, i])
            # calculating (r_exact - r_calc)*w
            L2_phi += ((phi_ext - phi[j, i]))*w_h[j, i]
            L2_ux += ((u_ext_x - ux[j, i]))*w_h[j, i]
            L2_uy += ((u_ext_y - uy[j, i]))*w_h[j, i]

            # calculating ((r_exact - r_calc)^2)*w
            L2_phi_sqr += ((phi_ext - phi[j, i])**2)*w_h[j, i]
            L2_ux_sqr += ((u_ext_x - ux[j, i])**2)*w_h[j, i]
            L2_uy_sqr += ((u_ext_y - uy[j, i])**2)*w_h[j, i]

            # calculating (| r_exact - r_calc|)*w
            L2_phi_abs += abs((phi_ext - phi[j, i]))*w_h[j, i]
            L2_ux_abs += abs((u_ext_x - ux[j, i]))*w_h[j, i]
            L2_uy_abs += abs((u_ext_y - uy[j, i]))*w_h[j, i]
        # This value can be >0 and throw a warning. This catches that warning and stops it from showing the value is set to nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: 
                out_L_phi = np.sqrt(L2_phi)
            except RuntimeWarning as rw:
                out_L_phi = np.nan
            try:
                out_L_u = np.sqrt(L2_ux+L2_uy)
            except RuntimeWarning as rw:
                out_L_u = np.nan
        
    return key, out_L_phi, out_L_u, np.sqrt(L2_phi_sqr), np.sqrt(L2_ux_sqr+L2_uy_sqr), np.sqrt(L2_phi_abs), np.sqrt(L2_ux_abs+L2_uy_abs),


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
    # Get the list of experiments for each of the types as a dict with the keys being the experiment names
    FileList = getFileList(parent_dir)
    print(FileList.keys())
    experiments = list(FileList.keys())
    print(experiments)

    for exp in experiments:
        # generate the map of inputs for calculating the error
        input_map = list(map(
            (lambda x: "{}/{}/{}".format(parent_dir, exp, x)), FileList[exp]))[:15]
        # calculte the error in a multithreaded way
        results = executeParallel(input_map, L2error) 
        # sort the returned map using K and N
        results.sort(key=sortfunction) 
        # opening an output file
        output_file = open("{}_errors.dat".format(exp),
                           "w+")  
        # writing the header
        output_file.write(
            "#K,N,L(phi),L(u),L^2(phi),L^2(u),abs(L)(phi),abs(L)(u) \n")  
        # write all the data to the file
        for line in results:
            a, l2phi, l2u, lphisqr, lusqr, lphi, lu = line
            a = a.split("_")
            output_file.write("{},{},{},{},{},{},{},{}\n".format(
                a[1], a[3], l2phi, l2u, lphisqr, lusqr, lphi, lu))
        # close the file
        output_file.close()


if __name__ == "__main__":
    main()
