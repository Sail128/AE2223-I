import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate as tb


def getFileList(parent_folder:str):
    dirs = os.listdir(parent_folder)
    FileList = {}
    for dir in dirs:
        FileList[dir] = os.listdir("{}/{}".format(parent_folder,dir))

    return FileList




def main():
    print(tb(getFileList("Data")))
    print("test")


if __name__ == "__main__":
    main()
