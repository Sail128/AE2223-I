
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


def subset(arr, condition):
    retlst = []
    for a in arr:
        if condition(a):
            retlst.append(a)
    return retlst


def plot(datafile:str):
    data = np.genfromtxt(datafile,delimiter=",")
    title = datafile.split(".")[0] + " phi"
    plots = []
    for i in range(2,21):
        k = np.array(subset(data, lambda x: int(x[1])==i))
        #print(k)
        plots.append((k[:,0],k[:,4],"N={}".format(i),{"style":linestyles[i]}))

    gp(plots,
    title=title,
    legend=True,
    yscale="log",
    ylabel = "L2-error",
    xlabel="K")
    plots=[]
    for i in range(1,21):
        k = np.array(subset(data, lambda x: int(x[0])==i))
        #print(k)
        plots.append((k[:,1],k[:,4],"k={}".format(i),{"style":linestyles[i]}))

    gp(plots,
    title=title,
    legend=True,
    yscale="log",
    ylabel = "L2-error",
    xlabel="N")

    # k2 = np.array(subset(data, lambda x: x[0]==2))
    # k3 = np.array(subset(data, lambda x: x[0]==3))
    # k5 = np.array(subset(data, lambda x: x[0]==5))
    # print(k5[:,4])
    # gp( [   (k2[:,1],k2[:,4],"k=2",{"style":"ro-"}),
    #         (k3[:,1],k3[:,4],"k=3",{"style":"gs-"}),
    #         (k5[:,1],k5[:,4],"k=5",{"style":"b^-"})
    #     ],
    #     title = title,
    #     legend=True
    #     )



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
    # plt.plot(data["primal_primal_errors_c0"][("K"==1)]["N","l2phi"])
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    main()
