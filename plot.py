
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from general_functions import general_plotter as gp


linestyles = ["b-", "bs-", "bo-", "b^-", "b*-",
              "r-", "rs-", "ro-", "r^-", "r*-",
              "g-", "gs-", "go-", "g^-", "g*-",
              "c-", "cs-", "co-", "c^-", "c*-",
              "m-", "ms-", "mo-", "m^-", "m*-", ]

# TODO: Rewrite for the selected plots possibilly resturcture this file
# Not sure if this file is even necessary


def main():
    parent_dir = "error_div"
    infiles = []
    datapanel = {}
    for file in os.listdir(parent_dir):
        if file.endswith(".dat"):
            infiles.append(file)
            datapanel[file.split(".")[0]] = pd.read_csv(
                "{}/{}".format(parent_dir, file))

    print(infiles)
    data = pd.Panel(data=datapanel)
    print(data.shape)
    print(list(data.items))

    # plotting the first plot
    fig1, ax1 = plt.subplots()
    # 'primal_dual_errors_C0_0',
    # 'primal_gauss_errors_C0_0',
    # 'primal_primal_errors_C0_0'
    sets = ['primal_dual_errors_C0_0',
            'primal_gauss_errors_C0_0',
            'primal_primal_errors_C0_0']
    setmarks = ["s", "^", "o"]
    markers = ["rv-","gv-","bv-",
                "r^-", "g^-","b^-",
                "ro-","go-","bo-"]
    Ks = [1, 5, 15]
    colors = ["r", "g", "b"]
    for i in range(3):
        for j in range(3):
            data[sets[i]][(data[sets[i]]["K"] == Ks[j])].plot(
                x="N",
                y="l2phi",
                ax=ax1,
                label="{}, K {}".format(sets[i],Ks[i]),
                marker=setmarks[i],
                color=colors[j]
                )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
