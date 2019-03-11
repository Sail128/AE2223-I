def general_plotter(
    plots,
    title:str=None,
    xlabel:str=None,
    xlim:tuple=None,
    xinvert:bool=False,
    ylabel:str=None,
    ylim:tuple=None,
    yinvert:bool=False,
    grid:bool=False,
    legend=False,
    fname:str=None,
    dpi:int=200,
    figsize:tuple=None, #todo documentation
    tightlayout:bool=False,
    show:bool=True,
    usetex:bool=False,
    framelines:str=None,
    axvline:list=[],
    axhline:list=[]
    ):
    """This function is a general plotting function.
    It sets up a plot with the given parameters.
    Compress plot setup down to one function. for more varities option use pyplot by itself

    Arguments:
        plots {list of tuples} -- [(xs,ys,label,{   style:str (a 3 char string describing marker, color and linestyle),
                                                    marker:str (a valid marker style),
                                                    markersize:float,
                                                    linewidth:int}))]

    Keyword Arguments:
        title {str} -- title displayed above the plot (default: {None})
        xlabel {str} -- label of the x-axis (default: {None})
        xlim {tuple} -- the limit of the x-axis (default: {None})
        xinvert {bool} -- invert the x-axis (default: {False})
        ylabel {str} -- label of the y-axis(default: {None})
        ylim {tuple} -- the limit of the y-axis (default: {None})
        yinvert {bool} -- invert the y-axis (default: {False})
        grid {bool} -- turns on or off the grid (default: {False})
        legend {bool or int} -- turns on the legend if passed a value. An int maybe passed for locating the legend. This is the same as pyplot (default: {False})
        fname {str} -- filename to save an image of the plot. If passed it will try to save the file with that filename (default: {None})
        dpi {int} -- image resolution. only used when fname is defined (default: {200})
        show {bool} -- Show the plot in it's own window. Set false if executing in seperate threads (default: {True})
        usetex {bool} -- use latex to format all text (default: {True})
        framlines {str} -- string denoting which framelines to show. Default is all (default: {None})
        axvline {list} -- (x,{**kwargs}) a list of vertical lines to be drawn into the diagram (default: {None})
        axhline {list} -- (y,{**kwargs}) a list of horizontal lines to be drawn into the diagram (default: {None})
    """
    from matplotlib import pyplot as plt

    if figsize!=None:
        fig = plt.figure(figsize=figsize)
    else:
        fig=plt.figure()
    plt.rc('text', usetex=usetex)
    for plot in plots:
        if len(plot) == 4:
            xs,ys,label,linestyle = plot
            if "style" in linestyle:
                style =  linestyle["style"]
                del linestyle["style"]
                plt.plot(xs, ys, style, label=label, **linestyle)
            else:
                plt.plot(xs, ys, label=label,  **linestyle)
        elif len(plot) == 3:
            xs,ys,label = plot
            plt.plot(xs,ys,label=label)
        else:
            print("You passed too many values for a plot. There can be either 3 or 4.")
            return 0
    for vline in axvline:
        if len(vline) == 2:
            plt.axvline(x=vline[0],**vline[1])
        else:
            plt.axvline(x=vline[0])
    for hline in axhline:
        plt.axhline(y=hline[0],**hline[1])
    if title!=None:
        plt.title(title)
    if xlabel !=None:
        plt.xlabel(xlabel)
    if xlim!=None:
        plt.xlim(xlim)
    if xinvert:
        plt.gca().invert_xaxis()
    if ylabel!=None:
        plt.ylabel(ylabel)
    if ylim != None:
        plt.ylim(ylim)
    if yinvert:
        plt.gca().invert_yaxis()
    if grid:
        plt.grid()
    #setup legend
    if type(legend)==int:
        plt.legend(loc=legend)
    else:
        if legend:
            plt.legend(loc=0)
    # draw framelines
    if framelines != None:
        ax = plt.gca()
        if "r" not in framelines:
            ax.spines["right"].set_visible(False)
        if "l" not in framelines:
            ax.spines["left"].set_visible(False)
        if "t" not in framelines:
            ax.spines["top"].set_visible(False)
        if "b" not in framelines:
            ax.spines["bottom"].set_visible(False)

    if tightlayout == True:
        fig.tight_layout()
    #save the figure with fname
    if fname!=None:
        plt.savefig(fname)
    else:
        if not show:
            print("Why do you want to create a graph that you don't save or show.\nThis is utterly useless")
    if show:
        plt.show()
    plt.close()
    return 1

if __name__ == "__main__":
    import numpy as np
    #should extent this example to show more of the capabilities
    x = np.arange(0, 5, 0.1)  
    y = np.sin(x)
    general_plotter( [
        (x,y,"test",{"style":"r.-.", "markersize":6.5}),
        (x,y+1,"test",{"style":"gx-", "markersize":3})
    ],
    axvline=[(0.5,{"label":"test vline"})],
    axhline=[(1.5,{"label":"test hline","color":'red'})],
    framelines="lb",
    legend=True,
    figsize=(10,7))
<<<<<<< HEAD
=======


>>>>>>> 6cd555cdbf6c6c77f1d0497d09df2cb2e4512fa8
