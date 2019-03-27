
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from general_functions import general_plotter as gp


linestyles = [  "b-","bs-","bo-","b^-","b*-",
                "r-","rs-","ro-","r^-","r*-",
                "g-","gs-","go-","g^-","g*-",
                "c-","cs-","co-","c^-","c*-",
                "m-","ms-","mo-","m^-","m*-",]

# TODO: Rewrite for the selected plots possibilly resturcture this file
def plot(data:pd.Panel):
    pass 



def main():
    parent_dir = "errors"
    infiles = []
    datapanel = {}
    for file in os.listdir(parent_dir):
        if file.endswith(".dat"):
            infiles.append(file)
            datapanel[file.split(".")[0]] = pd.read_csv("{}/{}".format(parent_dir,file))

    print(infiles)
    data = pd.Panel(data=datapanel)
    print(data.shape)
    print(list(data.items))
    print(data["primal_primal_errors_c0"][(data["primal_primal_errors_c0"]["K"]==1)])
    data["primal_primal_errors_c0"][(data["primal_primal_errors_c0"]["K"]==1)].plot(x="N",y="relL2phi")
    

if __name__ == "__main__":
    main()
