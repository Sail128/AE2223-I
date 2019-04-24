
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from general_functions import general_plotter as gp

import graphingGUI as gg


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
    if len(infiles) == 0:
        print("empty directory set")
        return 0
    data = pd.Panel(data=datapanel)

    print(data.shape)

    running = True
    while(running):
        answers = prompt(questions)  # , style=custom_style_2)
        if len(answers['datasets']) == 0:
            pprint("Please select atleast one dataset")
            continue
        pprint(answers)
        graphs.append(answers)
        plot(data, answers)


if __name__ == "__main__":
    main()
