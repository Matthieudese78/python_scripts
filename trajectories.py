#!/bin/python3
# %%
import numpy as np
# import scipy
# import numpy.linalg as LA
import pandas as pd
import matplotlib as mpl
# from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib as tikz
# from scipy import signal
from matplotlib.cm import ScalarMappable
# colorbar :
# import matplotlib.cm as cm
from matplotlib import gridspec
# import itertools
import matplotlib.colors as pltcolors

# from matplotlib.ticker import StrMethodFormatter

# %%
color1 = ["blue", "red", "green", "orange", "purple", "pink"]
plt.rc("lines", linewidth=0.8)
resolution = 600
view = [20, -50]

# %% troncature de la map inferno :
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap

cmapinf = truncate_colormap(mpl.cm.inferno, 0.0, 0.9, n=100)

def colordf(df, **kwargs):
    di = df[kwargs["colx"]]
    cdi = int(kwargs["ampl"] * di / kwargs["max"])
    return cdi

def colordflog(df, **kwargs):
    di = df[kwargs["colx"]]
    cdi = int(kwargs["ampl"] * np.log10(1.+di)) 
    return cdi

def color_from_value(df, **kwargs):
    vmax = np.max(df[kwargs["colx"]])
    print(f'vmax = {vmax}')
    if kwargs["logcol"]:
      dict_col = {
          "color": df.apply(
              colordflog, max=vmax, colx=kwargs["colx"], ampl=kwargs["ampl"], axis=1
          )
      }
    else:
      dict_col = {
          "color": df.apply(
              colordf, max=vmax, colx=kwargs["colx"], ampl=kwargs["ampl"], axis=1
          )
      }
    return pd.DataFrame(dict_col)

def colordf_impacts(df, **kwargs):
    di = df[kwargs["colx"]]
    cdi = kwargs['color_normal']
    if (di>1.e-10):
        cdi = kwargs["color_impact"]
    return cdi

def color_impact(df, **kwargs):
    dict_col = {
        "color": df.apply(
            colordf_impacts, colx=kwargs["colx"],
            color_normal=kwargs["color_normal"],
            color_impact=kwargs["color_impact"],axis=1
        )
    }
    return pd.DataFrame(dict_col)
# %% Attribution de couleurs dans le Dataframe :

# %% ######### trajectories :
# relative trajs. :

def diffcolind(df, **kwargs):
    return df.iloc[kwargs["ind1"]]["col1"] - df.iloc[kwargs["ind2"]]["col1"] 

def relaind(df, **kwargs):
    col3 = kwargs["col3"]
    dict1 = {col3: df.apply(diffcolind, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df.loc[df1.index, col3] = df1
    return "Done"

def diffcol(df, **kwargs):
    col1 = kwargs["col1"]
    col2 = kwargs["col2"]
    return df[col1] - df[col2]


def rela(df, **kwargs):
    col3 = kwargs["col3"]
    dict1 = {col3: df.apply(diffcol, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df.loc[df1.index, col3] = df1
    return "Done"


# plots default_kwargs :
ncurve_default = 5
ndf_default = 4
defkwargs = {
    "title1": "plotraj3d_default_title",
    "title_save": "plotraj3d_default_title",
    "colx": "uxplam_h",
    "coly": "uyplam_h",
    "colz": "uzplam_h",
    "rep_save": "~/default_graphs/",
    "label1": [f"curve_{i}" for i in np.arange(ncurve_default)],
    "labelx": r"$\mathbf{X}$",
    "labely": r"$\mathbf{Y}$",
    "labelz": r"$\mathbf{Z}$",
    "color1": ["blue" for i in np.arange(ncurve_default)],
    "view": view,
    "linestyle1": [["solid", "dashdot"][i % 2] for i in np.arange(ndf_default)],
    "lslabel": [f"$defleg_{i}$" for i in np.arange(ndf_default)],
    "title_ls": "linestyle def_title :",
    "title_col": "colors def_title :",
    "title_both": "colors def_both :",
    "leg_col": False,
    "leg_ls": False,
    "leg_both": False,
    "loc_ls": "lower left",
    "loc_col": "lower right",
    "loc_both": "upper center",
    "cust_leg" : False,
    "title_cust" : None,
    "labelcust" : [None],
    "colorcust" : ["blue"],
    "loc_cust" : (1.05,1.05),
    "lscust" : ["-"],
    "lind": np.arange(1, 10, 1),
    "loc_leg": "upper right",
    "mtype": "o",
    "msize": 4,
    "colcol": 'defcol',
    "ampl": 200.,
    "rcirc" : 1.,
    "rcircdot" : 0.5,
    "excent" : 0.25,
    "spinz" : 0., 
    "scatter" : False,
    "endpoint" : [True]*50,
    "xpower" : 2,
    "ypower" : 2,
    "lgrid" : False,
    "axline": None,
    "labelxline": "",
    "legendfontsize": 14,
    "equalaxis": False,
    "offsetangle" : 0.,
    "annotations" : [],
    "xymax" : [],
    "xmin" : [],
    "ymin" : [],
    "xmax" : [],
    "ymax" : [],
    "agreg" : "mean",
    "sol" : None,
    "alpha" : 1.,
    "y2" : None,
}


# %%
def pltraj2d_list_2axes(**kwargs):
    kwargs = defkwargs | kwargs
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    X = kwargs["x"]
    Y = kwargs["y"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(X) == list:
        if type(Y) != list:
            print(f"Les 3 inputs doivent etre des lists!")
        x_data = [xi for xi in X]
        axes.set_xlim(xmin=np.min(x_data[0]),xmax=np.max(x_data[0]))
    else:
        x_data = X
        axes.set_xlim(xmin=np.min(X),xmax=np.max(X))

    if type(Y) == list:
        y_data = [yi for yi in Y]
    else:
        y_data = Y

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['xpower'], kwargs['xpower']))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['ypower'], kwargs['ypower']))
    axes.yaxis.set_major_formatter(formatter)

    if type(X) != list:
        plt.plot(X, Y, label=label1, linewidth=lw, color=color1)
    else:
        [
            axes.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
            for i, xi in enumerate(x_data)
        ]
    if isinstance(kwargs['sol'],float):
      axes.plot(kwargs['x'],[kwargs['sol']]*len(kwargs['x']),c='black',linestyle='--',label=kwargs['labelsol'])
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.legend(loc=kwargs['loc_leg'])

    if isinstance(kwargs['y2'],list):
        ax2 = axes.twinx()
        for i,yi in enumerate(kwargs['y2']):
          ax2.plot(kwargs['x'],yi,linestyle='-.',label=kwargs['label2'][i])
        ax2.legend(loc=kwargs['loc_leg2'])
        ax2.set_ylabel(kwargs['labely2'])

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    plt.close("all")

def pltraj2d_list_sol(**kwargs):
    kwargs = defkwargs | kwargs
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    X = kwargs["x"]
    Y = kwargs["y"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(X) == list:
        if type(Y) != list:
            print(f"Les 3 inputs doivent etre des lists!")
        x_data = [xi for xi in X]
        axes.set_xlim(xmin=np.min(x_data[0]),xmax=np.max(x_data[0]))
    else:
        x_data = X
        axes.set_xlim(xmin=np.min(X),xmax=np.max(X))

    if type(Y) == list:
        y_data = [yi for yi in Y]
    else:
        y_data = Y

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['xpower'], kwargs['xpower']))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['ypower'], kwargs['ypower']))
    axes.yaxis.set_major_formatter(formatter)

    if type(X) != list:
        plt.plot(X, Y, label=label1, linewidth=lw, color=color1)
    else:
        [
            axes.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
            for i, xi in enumerate(x_data)
        ]
    if isinstance(kwargs['sol'],float):
      if type(kwargs['x'])==list:
        X = kwargs['x'][0]
      else: 
        X = kwargs['x']
      axes.plot(X,[kwargs['sol']]*len(X),c='black',linestyle='--',label=kwargs['labelsol'])
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)

    if ((isinstance(label1,list) and (label1)) or (isinstance(label1,str))):
        axes.legend(loc=kwargs['loc_leg'])
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    plt.close("all")

def pltraj2d_list(**kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    X = kwargs["x"]
    Y = kwargs["y"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(X) == list:
        if type(Y) != list:
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [xi for xi in X]
        axes.set_xlim(xmin=np.min(x_data[0]),xmax=np.max(x_data[0]))
    else:
        x_data = X
        axes.set_xlim(xmin=np.min(X),xmax=np.max(X))

    if type(Y) == list:
        y_data = [yi for yi in Y]
    else:
        y_data = Y

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['xpower'], kwargs['xpower']))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['ypower'], kwargs['ypower']))
    axes.yaxis.set_major_formatter(formatter)

    # if isinstance(label1,str):
    #     plt.legend(loc=loc_leg)
    # axes.set_ylim(ymax=1.1*np.max(Y))
    if (isinstance(kwargs['xmax'],float)):
        axes.set_xlim(xmax=kwargs['xmax'])
    if (isinstance(kwargs['ymax'],float)):
        axes.set_ylim(ymax=kwargs['ymax'])
    if (isinstance(kwargs['ymin'],float)):
        axes.set_ylim(ymin=kwargs['ymin'])

    if type(X) != list:
        plt.plot(X, Y, label=label1, linewidth=lw, color=color1)
    else:
        [
            axes.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)

    if (isinstance(kwargs['annotations'],list) and kwargs['annotations']):
        for i,Pi in enumerate(kwargs['annotations']):
            # axes.annotate(r'$Peak_{%d}$' % (i+1) + " : " + r"${:.2f}$".format(Pi[0]) + " Hz" , (Pi[0], Pi[1]),
            axes.annotate(r'$Peak_{%d}$' % (i+1) + " : " + r"${:.2f}$".format(Pi[0]) + " Hz" , Pi,
                xytext=(5,2), textcoords='offset points',
                family='sans-serif', fontsize=9, color='black') 
    if ((isinstance(label1,list) and (label1)) or (isinstance(label1,str))):
        plt.legend(loc=kwargs['loc_leg'])

    axes.set_xlabel(labelx,fontsize=14)
    axes.set_ylabel(labely,fontsize=14)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")

def pltraj3d(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(colx) == list:
        if (type(coly) != list) or (type(colz) != list):
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    if type(colz) == list:
        z_data = [df[coli] for coli in colz]
    else:
        z_data = df[colz]

    if type(colx) != list:
        plt.plot(df[colx], df[coly], df[colz], label=label1, linewidth=lw, color=color1)
        axes.scatter(
            x_data.tolist()[0],
            df[coly].tolist()[0],
            df[colz].tolist()[0],
            s=10,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            x_data.tolist()[len(x_data) - 1],
            df[coly].tolist()[len(y_data) - 1],
            df[colz].tolist()[len(z_data) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        [
            plt.plot(
                xi, y_data[i], z_data[i], label=label1[i], linewidth=lw, color=color1[i]
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi[0].tolist(),
                y_data[i].tolist()[0],
                z_data[i].tolist()[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[len(xi) - 1],
                y_data[i].tolist()[len(y_data[i]) - 1],
                z_data[i].tolist()[len(z_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    if isinstance(label1, str):
        plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    plt.close("all")

def pltraj3d_ind(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
        z_data = [df.iloc[indi][colz] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]
        z_data = df.iloc[ind][colz]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, z_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, z_data, label=label1, linewidth=lw, color=color1)
        axes.scatter(
            x_data.iloc[0],
            y_data.iloc[0],
            z_data.iloc[0],
            s=10,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            x_data.iloc[-1],
            y_data.iloc[-1],
            z_data.iloc[-1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], z_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], z_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]
        [
            axes.scatter(
                xi.iloc[0],
                y_data[i].iloc[0],
                z_data[i].iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.iloc[-1],
                y_data[i].iloc[-1],
                z_data[i].iloc[-1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)


    if (isinstance(label1, str)):
        plt.legend(loc=loc_leg)
    if (isinstance(label1, list) and (label1)):
        if (label1[0] != None):
            g = [
                plt.plot([], [], color='w', marker=kwargs['markers'][i], markerfacecolor=color1[i], ms=6.)[0]
                for i, coli in enumerate(x_data)
            ]
            leg = plt.legend(
                handles=g, labels=label1,  loc=kwargs['loc_leg'], title=None, fontsize=8
            )
            axes.add_artist(leg)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj3d_ccirc(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
        z_data = [df.iloc[indi][colz] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]
        z_data = df.iloc[ind][colz]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, z_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, z_data, label=label1, linewidth=lw, color=color1)

        if (kwargs['endpoint']):
          axes.scatter(
              x_data.iloc[0],
              y_data.iloc[0],
              z_data.iloc[0],
              s=10,
              c="black",
              marker="o",
              edgecolors="red",
          )
          axes.scatter(
              x_data.iloc[-1],
              y_data.iloc[-1],
              z_data.iloc[-1],
              s=10,
              c="yellow",
              marker="o",
              edgecolors="red",
          )
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], z_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], z_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]
        for i, xi in enumerate(x_data):
          if (kwargs['endpoint'][i]):
            axes.scatter(
                xi.iloc[0],
                y_data[i].iloc[0],
                z_data[i].iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            axes.scatter(
                xi.iloc[-1],
                y_data[i].iloc[-1],
                z_data[i].iloc[-1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )

    xmax = np.max([np.abs(xi).max() for i,xi in enumerate(x_data)])
    ymax = np.max([np.abs(xi).max() for i,xi in enumerate(y_data)])
    zmax = np.max([np.abs(xi).max() for i,xi in enumerate(z_data)])
    xmax = np.max([xmax,ymax,zmax])
    axes.set_xlim([-xmax,xmax])
    axes.set_ylim([-xmax,xmax])
    # axes.set_zlim([-0.01*xmax,xmax])
    # axes.set_aspect('auto')
    axes.grid(True)
    # axes.set_facecolor('None')
    axes.set_facecolor("white")

    g = [
        plt.plot([], [], color='w', marker=kwargs['markers'][i], markerfacecolor=color1[i], ms=6.)[0]
        for i, coli in enumerate(x_data)
    ]
    leg = plt.legend(
        handles=g, labels=label1,  loc="upper left", title=None, fontsize=8
    )
    axes.add_artist(leg)
    # if (isinstance(label1, str) or (label1)):
    #     plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj2d_pion(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]
    xpower = kwargs["xpower"]
    ypower = kwargs["ypower"]
    rcirc = kwargs["rcirc"]
    rcircdot = kwargs["rcircdot"]
    excent = kwargs["excent"]
    spinz = kwargs["spinz"]
    O1 = kwargs["offsetangle"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, label=label1, linewidth=lw, color=color1)
        if (kwargs["endpoint"]):
            axes.scatter(
                x_data.iloc[0],
                y_data.iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            axes.scatter(
                x_data.iloc[-1],
                y_data.iloc[-1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]

        for i, xi in enumerate(x_data):
            if (kwargs["endpoint"][i]):
                axes.scatter(
                xi.iloc[0],
                y_data[i].iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
                )
                axes.scatter(
                    xi.iloc[-1],
                    y_data[i].iloc[-1],
                    s=10,
                    c="yellow",
                    marker="o",
                    edgecolors="red",
                )

    # circles : 
    # spin1 = df.iloc[(df['spin'].abs()).idxmin()]['spin']
    # spin2 = df.iloc[(df['spin'].abs()).idxmax()]['spin']

    # postion initiale globale :
    spin1 = df.iloc[0]['spin'] + (O1*np.pi/180.)
    spin2 = df.iloc[-1]['spin'] + (O1*np.pi/180.)
    # temps max de toutes les listes d'index :
        # il faut que l'index soit ordonnes / temps ! 
    if (type(ind)==list):
        ind = [idx for idx in ind if not idx.empty]
        if any(not idx.empty for idx in ind):
            iend = max([np.max(df.loc[i].index) for i in ind ])
            spin2 = df.loc[iend]['spin'] + (O1*np.pi/180.)
    else:
        if not ind.empty:
            spin2 = df.loc[ind].iloc[-1]['spin'] + (O1*np.pi/180.)

    print(f"spin1 = {spin1}")
    print(f"spin2 = {spin2}")
    exc1 = (np.cos(spin1)*excent[0] - np.sin(spin1)*excent[1],
            np.sin(spin1)*excent[0] + np.cos(spin1)*excent[1])
    exc2 = (np.cos(spin2)*excent[0] - np.sin(spin2)*excent[1],
            np.sin(spin2)*excent[0] + np.cos(spin2)*excent[1])

    e21x = np.cos(2.*np.pi/3.)*exc1[0] - np.sin(2.*np.pi/3.)*exc1[1]
    e21y = np.sin(2.*np.pi/3.)*exc1[0] + np.cos(2.*np.pi/3.)*exc1[1]
    e21 = np.array([e21x,e21y])
    e31x = np.cos(-2.*np.pi/3.)*exc1[0] - np.sin(-2.*np.pi/3.)*exc1[1]
    e31y = np.sin(-2.*np.pi/3.)*exc1[0] + np.cos(-2.*np.pi/3.)*exc1[1]
    e31 = np.array([e31x,e31y])

    e22x = np.cos(2.*np.pi/3.)*exc2[0] - np.sin(2.*np.pi/3.)*exc2[1]
    e22y = np.sin(2.*np.pi/3.)*exc2[0] + np.cos(2.*np.pi/3.)*exc2[1]
    e22 = np.array([e22x,e22y])
    e32x = np.cos(-2.*np.pi/3.)*exc2[0] - np.sin(-2.*np.pi/3.)*exc2[1]
    e32y = np.sin(-2.*np.pi/3.)*exc2[0] + np.cos(-2.*np.pi/3.)*exc2[1]
    e32 = np.array([e32x,e32y])

    spin1 = spin1*180./np.pi
    spin2 = spin2*180./np.pi
    awidth = kwargs['arcwidth'] 
    arc1 = patches.Arc(exc1, 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=-awidth, theta2=awidth, edgecolor='red', facecolor='none',ls='--')
    axes.add_patch(arc1)
    arc2 = patches.Arc(tuple(e21), 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=(-awidth+120.), theta2=(awidth+120.), edgecolor='green', facecolor='none',ls='--')
    axes.add_patch(arc2)
    arc3 = patches.Arc(tuple(e31), 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=(-awidth+240.), theta2=(awidth+240.), edgecolor='blue', facecolor='none',ls='--')
    axes.add_patch(arc3)

    arc1 = patches.Arc(exc2, 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=-awidth, theta2=awidth, edgecolor='red', facecolor='none',ls='-')
    axes.add_patch(arc1)
    arc2 = patches.Arc(tuple(e22), 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=(-awidth+120.), theta2=(awidth+120.), edgecolor='green', facecolor='none',ls='-')
    axes.add_patch(arc2)
    arc3 = patches.Arc(tuple(e32), 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=(-awidth+240.), theta2=(awidth+240.), edgecolor='blue', facecolor='none',ls='-')
    axes.add_patch(arc3)
    # max clearance :
    clmax = kwargs['clmax']
    draw_circle=plt.Circle((0.,0.),clmax,fill=False, ls='--', color='black')
    axes.add_artist(draw_circle)
    # arc3 = patches.Arc((0.,0.), 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=0., theta2=360., edgecolor='black', facecolor='none',ls='-')
    # axes.add_patch(arc3)
    # axes limits :
    if isinstance(kwargs['xymax'],float):
        vmt = kwargs['xymax']
    else:
        vmx = np.max(np.abs(df[colx]))
        vmy = np.max(np.abs(df[coly]))
        vmt = np.max([vmx,vmy])
    axes.set_ylim([-1.05*vmt,1.05*vmt])
    axes.set_xlim([-1.05*vmt,1.05*vmt])
    # axes.set_aspect("equal",adjustable='box')
    axes.set_aspect("equal")
    # fin circle
    axes.set_facecolor("white")
    axes.grid(kwargs["lgrid"])

    # g = [
    #     plt.plot([], [], color=color1[i], marker=kwargs['markers'][i], ms=5.)[0]
    #     for i, coli in enumerate(x_data)
    # ]
    # leg = plt.legend(
    #     handles=g, labels=label1,  loc="upper left", title=None, fontsize=8
    # )
    # axes.add_artist(leg)
    if (label1[0] != None):
        g = [
            plt.plot([], [], color='w', marker=kwargs['markers'][i], markerfacecolor=color1[i], ms=6.)[0]
            for i, coli in enumerate(x_data)
        ]
        leg = plt.legend(
            handles=g, labels=label1,  loc="upper left", title=None, fontsize=8
        )
        axes.add_artist(leg)
    # if (isinstance(label1, str) or (label1)):
    #     plt.legend(loc=loc_leg,linestyle='solid',fontsize=8.)

    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-xpower, xpower))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-ypower, ypower))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj2d_mufgen(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]
    xpower = kwargs["xpower"]
    ypower = kwargs["ypower"]
    axline = kwargs["axline"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, label=label1, linewidth=lw, color=color1)
        if isinstance(kwargs['axline'],float): 
            plt.axhline(y=kwargs['axline'], color='black', linestyle='--',label=kwargs['labelxline'])
        
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]
        if (isinstance(kwargs['axline'],list) and kwargs['axline']): 
            [ plt.axhline(y=axli, color='black', linestyle='--',label=kwargs['labelxline'][i]) for i,axli in enumerate(kwargs['axline']) ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(kwargs["lgrid"])

    if (isinstance(label1, str) or (label1)):
        plt.legend(loc=loc_leg)

    # if (isinstance(kwargs['labelxline'],str) or (kwargs['labelxline'])):
    #     plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx,fontsize=kwargs['legendfontsize'])
    axes.set_ylabel(labely,fontsize=kwargs['legendfontsize'])

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-xpower, xpower))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-ypower, ypower))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj2d_ind(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]
    xpower = kwargs["xpower"]
    ypower = kwargs["ypower"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, label=label1, linewidth=lw, color=color1)
        if (kwargs["endpoint"]):
            axes.scatter(
                x_data.iloc[0],
                y_data.iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            axes.scatter(
                x_data.iloc[-1],
                y_data.iloc[-1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]

        for i, xi in enumerate(x_data):
            if (kwargs["endpoint"][i]):
                axes.scatter(
                xi.iloc[0],
                y_data[i].iloc[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
                )
                axes.scatter(
                    xi.iloc[-1],
                    y_data[i].iloc[-1],
                    s=10,
                    c="yellow",
                    marker="o",
                    edgecolors="red",
                )

    # axes.set_facecolor('None')
    if (kwargs['equalaxis']):
        # axes.axis("equal")
        vmx = np.max(np.abs(df[colx]))
        vmy = np.max(np.abs(df[coly]))
        vmt = np.max([vmx,vmy])
        axes.set_ylim([-1.05*vmt,1.05*vmt])
        axes.set_xlim([-1.05*vmt,1.05*vmt])
        axes.set_aspect("equal")
    axes.set_facecolor("white")
    axes.grid(kwargs["lgrid"])

    if (isinstance(label1, str) or (label1)):
        plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-xpower, xpower))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-ypower, ypower))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj2d_mufgen(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    scatter = kwargs["scatter"]
    sp = kwargs["msize"]
    xpower = kwargs["xpower"]
    ypower = kwargs["ypower"]
    axline = kwargs["axline"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    if type(ind) == list:
        x_data = [df.iloc[indi][colx] for indi in ind]
        y_data = [df.iloc[indi][coly] for indi in ind]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]

    if type(ind) != list:
        if (scatter):
            axes.scatter(x_data, y_data, label=label1, linewidth=lw, color=color1,s=sp)
        else:
            plt.plot(x_data, y_data, label=label1, linewidth=lw, color=color1)
        if isinstance(kwargs['axline'],float): 
            plt.axhline(y=kwargs['axline'], color='black', linestyle='--',label=kwargs['labelxline'])
        
    else:
        if (scatter):
            [
                axes.scatter(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i], s=sp
                )
                for i, xi in enumerate(x_data)
            ]
        else:
            [
                plt.plot(
                    xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i]
                )
                for i, xi in enumerate(x_data)
            ]
        if (isinstance(kwargs['axline'],list) and kwargs['axline']): 
            [ plt.axhline(y=axli, color='black', linestyle='--',label=kwargs['labelxline'][i]) for i,axli in enumerate(kwargs['axline']) ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(kwargs["lgrid"])

    if (isinstance(label1, str) or (label1)):
        plt.legend(loc=loc_leg)

    # if (isinstance(kwargs['labelxline'],str) or (kwargs['labelxline'])):
    #     plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx,fontsize=kwargs['legendfontsize'])
    axes.set_ylabel(labely,fontsize=kwargs['legendfontsize'])

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-xpower, xpower))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-ypower, ypower))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

def pltraj2d(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    xpower = kwargs["xpower"]
    ypower = kwargs["ypower"]
    print(f"pltraj2d : {kwargs['endpoint']}")
        
    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(colx) == list:
        if type(coly) != list:
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-xpower, xpower))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-ypower, ypower))
    axes.yaxis.set_major_formatter(formatter)

    if type(colx) != list:
        axes.set_xlim(xmin=np.min(x_data),xmax=np.max(x_data))
        if ("alpha" in kwargs):
            plt.plot(df[colx], df[coly], label=label1, linewidth=lw, color=color1,alpha=kwargs["alpha"])
        if not ("alpha" in kwargs):
            plt.plot(df[colx], df[coly], label=label1, linewidth=lw, color=color1)
        if (kwargs['endpoint']):
            axes.scatter(
               df[colx][0], df[coly][0], s=10, c="black", marker="o", edgecolors="red"
            )
            axes.scatter(
                df[colx][len(df[colx]) - 1],
                df[coly][len(df[coly]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
    else:
        axes.set_xlim(xmin=np.min(x_data[0]),xmax=np.max(x_data[0]))
        if not ("alpha" in kwargs):
            [
                plt.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
                for i, xi in enumerate(x_data)
            ]
        if ("alpha" in kwargs):
            [
                plt.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i],alpha=kwargs["alpha"])
                for i, xi in enumerate(x_data)
            ]
        for i,xi in enumerate(x_data):
            if (kwargs['endpoint'][i]):
                axes.scatter(
                xi[0], y_data[i][0], s=10, c="black", marker="o", edgecolors="red")
                axes.scatter(
                xi[len(xi) - 1],
                y_data[i][len(y_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red")
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)
    if isinstance(label1, (str,list)):
        plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")

def PSD(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    fs = kwargs["fs"]
    nfft = kwargs["NFFT"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(colx) == list:
        if type(coly) != list:
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df.iloc[kwargs['ind']][coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df.iloc[kwargs['ind']][coli] for coli in coly]
    else:
        y_data = df.iloc[kwargs['ind']][coly]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    
    if type(colx) != list:
        power, freq = plt.psd(1.e6*y_data, NFFT=nfft, Fs=fs, scale_by_freq=0.29, color=color1)
        plt.close('all')

        power_density = 10.*np.log10(power)

        axes.plot(freq,power_density,label=label1,linewidth=lw, color=color1)

        # plt.plot(freq, power, color=color1)
        # plt.plot(df[colx], df[coly], label=label1, linewidth=lw, color=color1)
        # peaks, heights = scipy.signal.find_peaks(power,height=1.e12)[0]
 
        # for peak_x, peak_y in zip(df.iloc[peaks][colx], df.iloc[peaks][coly]):
        # axes.annotate(r'$P$', (Pi[0], Pi[1]),
        #         xytext=(4,5), textcoords='offset points',
        #         family='sans-serif', fontsize=12, color='black') \
        # for peak_x, peak_y in zip(df.iloc[peaks][colx], df.iloc[peaks][coly]):
            # print(f"x = {peak_x}, y = {peak_y}") 
            # plt.annotate(f'Peak\n({peak_x:.2f}, {peak_y:.2f})',    xy=(peak_x,peak_y),textcoords='offset points',
            # xytext=(4, 5),fontsize=12)
            # # arrowprops=dict(facecolor='black', arrowstyle='->'))

    else:
        [
            plt.psd(xi, y_data[i], NFFT=nfft, Fs=fs, color=color1[i])
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor('white')
    axes.grid(True)
    # axes.grid(linewidth=0.1)
    plt.xticks(np.arange(min(freq), max(freq)+1,10))
    if isinstance(label1, str):
        plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")

def pltraj2d_circs(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    rcirc = kwargs["rcirc"]
    rcircdot = kwargs["rcircdot"]
    excent = kwargs["excent"]
    spinz = kwargs["spinz"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(colx) == list:
        if type(coly) != list:
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)

    if type(colx) != list:
        plt.plot(df[colx], df[coly], label=label1, linewidth=lw, color=color1)
        axes.scatter(
            df[colx][0], df[coly][0], s=10, c="black", marker="o", edgecolors="red"
        )
        axes.scatter(
            df[colx][len(df[colx]) - 1],
            df[coly][len(df[coly]) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        [
            plt.plot(xi, y_data[i], label=label1[i], linewidth=lw, color=color1[i])
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi[0], y_data[i][0], s=10, c="black", marker="o", edgecolors="red"
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi[len(xi) - 1],
                y_data[i][len(y_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # circles : 

    spin1 = df.iloc[(df['spin'].abs()).idxmin()]['spin']
    spin2 = df.iloc[(df['spin'].abs()).idxmax()]['spin']
    # spin2 = df.iloc[(df['spin'].abs()).idxmax()]['spin']
    # spin1 = df['spin'].head(1).iloc[0]
    # spin2 = df['spin'].tail(1).iloc[0]
    print(f"spin1 = {spin1}")
    print(f"spin2 = {spin2}")
    exc1 = (np.cos(spin1)*excent[0] - np.sin(spin1)*excent[1],
            np.sin(spin1)*excent[0] + np.cos(spin1)*excent[1])
    exc2 = (np.cos(spin2)*excent[0] - np.sin(spin2)*excent[1],
            np.sin(spin2)*excent[0] + np.cos(spin2)*excent[1])

    e21x = np.cos(2.*np.pi/3.)*exc1[0] - np.sin(2.*np.pi/3.)*exc1[1]
    e21y = np.sin(2.*np.pi/3.)*exc1[0] + np.cos(2.*np.pi/3.)*exc1[1]
    e21 = np.array([e21x,e21y])
    e31x = np.cos(-2.*np.pi/3.)*exc1[0] - np.sin(-2.*np.pi/3.)*exc1[1]
    e31y = np.sin(-2.*np.pi/3.)*exc1[0] + np.cos(-2.*np.pi/3.)*exc1[1]
    e31 = np.array([e31x,e31y])

    e22x = np.cos(2.*np.pi/3.)*exc2[0] - np.sin(2.*np.pi/3.)*exc2[1]
    e22y = np.sin(2.*np.pi/3.)*exc2[0] + np.cos(2.*np.pi/3.)*exc2[1]
    e22 = np.array([e22x,e22y])
    e32x = np.cos(-2.*np.pi/3.)*exc2[0] - np.sin(-2.*np.pi/3.)*exc2[1]
    e32y = np.sin(-2.*np.pi/3.)*exc2[0] + np.cos(-2.*np.pi/3.)*exc2[1]
    e32 = np.array([e32x,e32y])

    spin1 = spin1*180./np.pi
    spin2 = spin2*180./np.pi

    arc1 = patches.Arc(exc1, 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=-15.81607, theta2=15.81607, edgecolor='red', facecolor='none',ls='--')
    axes.add_patch(arc1)
    arc2 = patches.Arc(tuple(e21), 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=(-15.81607+120.), theta2=(15.81607+120.), edgecolor='green', facecolor='none',ls='--')
    axes.add_patch(arc2)
    arc3 = patches.Arc(tuple(e31), 2.*rcirc, 2.*rcirc,  -90.+spin1, theta1=(-15.81607+240.), theta2=(15.81607+240.), edgecolor='blue', facecolor='none',ls='--')
    axes.add_patch(arc3)

    arc1 = patches.Arc(exc2, 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=-15.81607, theta2=15.81607, edgecolor='red', facecolor='none',ls='-')
    axes.add_patch(arc1)
    arc2 = patches.Arc(tuple(e22), 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=(-15.81607+120.), theta2=(15.81607+120.), edgecolor='green', facecolor='none',ls='-')
    axes.add_patch(arc2)
    arc3 = patches.Arc(tuple(e32), 2.*rcirc, 2.*rcirc,  -90.+spin2, theta1=(-15.81607+240.), theta2=(15.81607+240.), edgecolor='blue', facecolor='none',ls='-')
    axes.add_patch(arc3)

    # with plt.Circle :
    # draw_circle=plt.Circle(excent, rcirc,fill=False, color='red')
    # axes.add_artist(draw_circle)
    # draw_circle=plt.Circle((e2x,e2y), rcirc,fill=False, color='green')
    # axes.add_artist(draw_circle)
    # draw_circle=plt.Circle((e3x,e3y), rcirc,fill=False, color='blue')
    # axes.add_artist(draw_circle)

    # axes limits :
    vmx = np.max(np.abs(df[colx]))
    vmy = np.max(np.abs(df[coly]))
    vmt = np.max([vmx,vmy])
    axes.set_ylim([-vmt,vmt])
    axes.set_xlim([-vmt,vmt])
    # axes.set_ylim([-2.05*(rcirc-LA.norm(excent)),2.05*(rcirc-LA.norm(excent))])
    # axes.set_xlim([-2.05*(rcirc-LA.norm(excent)),2.05*(rcirc-LA.norm(excent))])
    axes.set_aspect("equal",adjustable='box')
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)
    if isinstance(label1,str):
        plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    tikz.save(rep_save + title_save + ".tex")
    plt.close("all")

def pltraj3d_dfs(dfs, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    linestyle1 = kwargs["linestyle1"]
    lslabel = kwargs["lslabel"]
    title_ls = kwargs["title_ls"]
    title_col = kwargs["title_col"]
    leg_col = kwargs["leg_col"]
    leg_ls = kwargs["leg_ls"]
    loc_ls = kwargs["loc_ls"]
    loc_col = kwargs["loc_col"]
    title_both = kwargs["title_both"]
    leg_both = kwargs["leg_both"]
    loc_both = kwargs["loc_both"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    for j, df in enumerate(dfs):
        if type(colx) == list:
            if (type(coly) != list) or (type(colz) != list):
                print(f"Les 3 inputs doivent etre des lists!")
                return
            x_data = [df[coli] for coli in colx]
        else:
            x_data = df[colx]

        if type(coly) == list:
            y_data = [df[coli] for coli in coly]
        else:
            y_data = df[coly]

        if type(colz) == list:
            z_data = [df[coli] for coli in colz]
        else:
            z_data = df[colz]

        if type(colx) != list:
            plt.plot(
                df[colx],
                df[coly],
                df[colz],
                label=label1,
                linewidth=lw,
                linestyle=linestyle1[j],
                color=color1[j],
            )
            axes.scatter(
                x_data[0],
                df[coly][0],
                df[colz][0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            axes.scatter(
                x_data[len(x_data) - 1],
                df[coly][len(y_data) - 1],
                df[colz][len(z_data) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
        else:
            [
                plt.plot(
                    xi,
                    y_data[i],
                    z_data[i],
                    label=label1[i],
                    linewidth=lw,
                    linestyle=linestyle1[j],
                    color=color1[j],
                )
                for i, xi in enumerate(x_data)
            ]
            [
                axes.scatter(
                    xi[0],
                    y_data[i][0],
                    z_data[i][0],
                    s=10,
                    c="black",
                    marker="o",
                    edgecolors="red",
                )
                for i, xi in enumerate(x_data)
            ]
            [
                axes.scatter(
                    xi[len(xi) - 1],
                    y_data[i][len(y_data[i]) - 1],
                    z_data[i][len(z_data[i]) - 1],
                    s=10,
                    c="yellow",
                    marker="o",
                    edgecolors="red",
                )
                for i, xi in enumerate(x_data)
            ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    # plt.legend(bbox_to_anchor=(1.05, 1))
    # axes.legend(bbox_to_anchor=(1.05, 1))
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)

    # ajout d'ume legende pour les dataframes :
    if leg_ls:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c="black", ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        # leg = plt.legend(handles=g, labels=lslabel, bbox_to_anchor=(1.25,0.8), title=title_ls, fontsize=8)
        leg = plt.legend(
            handles=g, labels=lslabel, loc=loc_ls, title=title_ls, fontsize=8
        )
        axes.add_artist(leg)
        # ajout d'ume legende pour les courbes :

    if leg_both:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c=color1[i], ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        # leg = plt.legend(handles=g, labels=lslabel, bbox_to_anchor=(1.25,0.8), title=title_ls, fontsize=8)
        leg = plt.legend(
            handles=g, labels=lslabel, loc=loc_both, title=title_both, fontsize=8
        )
        axes.add_artist(leg)

    if leg_col:
        if type(colx) == list:
            g = [
                plt.plot([], [], linestyle="solid", color=color1[i], ms=6)[0]
                for i, coli in enumerate(color1)
            ]
            # leg = plt.legend(handles=g, labels=label1, bbox_to_anchor=(1.25,0.6), title=title_col, fontsize=8)
            leg = plt.legend(
                handles=g, labels=label1, loc=loc_col, title=title_col, fontsize=8
            )
            axes.add_artist(leg)
        else:
            plt.legend(bbox_to_anchor=(1.25, 0.6))

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    plt.close("all")


def pltraj2d_dfs(dfs, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    # for key, value in kwargs.items():
    #     print(f"{key}: {value}")
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    linestyle1 = kwargs["linestyle1"]
    lslabel = kwargs["lslabel"]
    title_ls = kwargs["title_ls"]
    title_col = kwargs["title_col"]
    title_both = kwargs["title_both"]
    leg_col = kwargs["leg_col"]
    leg_ls = kwargs["leg_ls"]
    leg_both = kwargs["leg_both"]
    loc_col = kwargs["loc_col"]
    loc_ls = kwargs["loc_ls"]
    loc_both = kwargs["loc_both"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    for j, df in enumerate(dfs):
        if type(colx) == list:
            if type(coly) != list:
                print(f"Les 3 inputs doivent etre des lists!")
                return
            x_data = [df[coli] for coli in colx]
        else:
            x_data = df[colx]

        if type(coly) == list:
            y_data = [df[coli] for coli in coly]
        else:
            y_data = df[coly]

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes.xaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes.yaxis.set_major_formatter(formatter)

        if type(colx) != list:
            plt.plot(
                df[colx],
                df[coly],
                label=label1[j],
                linewidth=lw,
                linestyle=linestyle1[j],
                color=color1[j],
                alpha=kwargs['alpha'],
            )
            if kwargs['endpoint']:
                axes.scatter(
                    df[colx][0], df[coly][0], s=10, c="black", marker="o", edgecolors="red"
                )
                axes.scatter(
                    df[colx][len(df[colx]) - 1],
                    df[coly][len(df[coly]) - 1],
                    s=10,
                    c="yellow",
                    marker="o",
                    edgecolors="red",
                )
            axes.set_xlim(xmin=df[colx].min(),xmax=df[colx].max())
        else:
            xmini = np.min([df[colxi].min() for colxi in colx])
            xmaxi = np.max([df[colxi].max() for colxi in colx])
            axes.set_xlim(xmin=xmini,xmax=xmaxi)
            [
                plt.plot(
                    xi,
                    y_data[i],
                    label=label1[i],
                    linewidth=lw,
                    linestyle=linestyle1[j],
                    color=color1[i],
                    alpha=kwargs['alpha'],
                )
                for i, xi in enumerate(x_data)
            ]
            if kwargs['endpoint']:
                [
                    axes.scatter(
                        xi[0], y_data[i][0], s=10, c="black", marker="o", edgecolors="red"
                    )
                    for i, xi in enumerate(x_data)
                ]
                [
                    axes.scatter(
                        xi[len(xi) - 1],
                        y_data[i][len(y_data[i]) - 1],
                        s=10,
                        c="yellow",
                        marker="o",
                        edgecolors="red",
                    )
                    for i, xi in enumerate(x_data)
                ]

        if isinstance(kwargs['sol'],float):
          if isinstance(kwargs['colx'],list):
            axes.plot(df[kwargs['colx'][0]],[kwargs['sol']]*len(df[kwargs['colx'][0]]),c='black',linestyle='-',label=kwargs['labelsol'])
          if isinstance(kwargs['colx'],str):
            axes.plot(df[kwargs['colx']],[kwargs['sol']]*len(df[kwargs['colx']]),c='black',linestyle='-',label=kwargs['labelsol'])
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)
    # ajout d'ume legende pour les dataframes :
    # handles, labels = axes.get_legend_handles_labels()
    if leg_ls:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c="black", ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        legls = plt.legend(
            handles=g, labels=lslabel, loc=loc_ls, title=title_ls, fontsize=10
        )
        axes.add_artist(legls)

    if leg_both:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c=color1[i], ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        # leg = plt.legend(handles=g, labels=lslabel, bbox_to_anchor=(1.25,0.8), title=title_ls, fontsize=8)
        leg = plt.legend(
            handles=g, labels=lslabel, loc=loc_both, title=title_both, fontsize=10
        )
        axes.add_artist(leg)

    if leg_col:
        if type(colx) == list:
            # ajout d'ume legende pour les courbes :
            g = [
                plt.plot([], [], linestyle="solid", color=color1[i], ms=6)[0]
                for i, coli in enumerate(color1)
            ]
            # leg = plt.legend(handles=g, labels=label1, bbox_to_anchor=(1.25,0.6), title=title_col, fontsize=8)
            leg = plt.legend(
                handles=g, labels=label1, loc=loc_col, title=title_col, fontsize=10
            )
            axes.add_artist(leg)
        else:
            plt.legend(bbox_to_anchor=(1.0, 1))

    # ajout d'ume legende pour les courbes :
    if kwargs['cust_leg']:
        gcust = [
            plt.plot([], [], linestyle=kwargs['lscust'][i],color=coli, ms=7)[0]
            for i, coli in enumerate(kwargs['colorcust'])
        ]

        # handles.extend(gcust)

        # axes.legend(gcust,kwargs['lslabel']+kwargs['labelcust'])
        leg = plt.legend(handles=gcust, labels=kwargs['labelcust'], loc=kwargs['loc_cust'], title=kwargs['title_cust'], fontsize=10)

        axes.add_artist(leg)
    
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    #
    if isinstance(kwargs['y2'],list):
      ax2 = axes.twinx()
      for j,df in enumerate(dfs):
        for i,yi in enumerate(kwargs['y2']):
          ax2.plot(df[kwargs['x2']],df[yi],linestyle=linestyle1[j],color=kwargs['color2'][i],label=kwargs['label2'][i],alpha=kwargs['alpha'])
      ax2.set_ylabel(kwargs['labely2'])
      if (not kwargs['cust_leg']):
        ax2.legend(loc=kwargs['loc_leg2'])
    #
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")


def pltraj2d_dfs_truncate(dfs, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    linestyle1 = kwargs["linestyle1"]
    lslabel = kwargs["lslabel"]
    title_ls = kwargs["title_ls"]
    title_col = kwargs["title_col"]
    title_both = kwargs["title_both"]
    leg_col = kwargs["leg_col"]
    leg_ls = kwargs["leg_ls"]
    leg_both = kwargs["leg_both"]
    loc_ls = kwargs["loc_ls"]
    loc_col = kwargs["loc_col"]
    loc_both = kwargs["loc_both"]
    lind = kwargs["lind"]
    # indices finaux et initiaux :
    ind1 = lind[0]
    ind2 = np.max(lind)
    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)

    for j, df in enumerate(dfs):
        if type(colx) == list:
            if type(coly) != list:
                print(f"Les 3 inputs doivent etre des lists!")
                return
            x_data = [df[coli] for coli in colx]
        else:
            x_data = df[colx]

        if type(coly) == list:
            y_data = [df[coli] for coli in coly]
        else:
            y_data = df[coly]

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes.xaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes.yaxis.set_major_formatter(formatter)

        if type(colx) != list:
            plt.plot(
                df[colx][lind],
                df[coly][lind],
                label=label1,
                linewidth=lw,
                linestyle=linestyle1[j],
                color=color1[j],
            )
            axes.scatter(
                df[colx][ind1],
                df[coly][ind1],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            axes.scatter(
                df[colx][ind2],
                df[coly][ind2],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
        else:
            [
                plt.plot(
                    xi[lind],
                    y_data[i][lind],
                    label=label1[i],
                    linewidth=lw,
                    linestyle=linestyle1[j],
                    color=color1[j],
                )
                for i, xi in enumerate(x_data)
            ]
            [
                axes.scatter(
                    xi[ind1],
                    y_data[i][ind1],
                    s=10,
                    c="black",
                    marker="o",
                    edgecolors="red",
                )
                for i, xi in enumerate(x_data)
            ]
            [
                axes.scatter(
                    xi[ind2],
                    y_data[i][ind2],
                    s=10,
                    c="yellow",
                    marker="o",
                    edgecolors="red",
                )
                for i, xi in enumerate(x_data)
            ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)

    # ajout d'ume legende pour les dataframes :
    if leg_ls:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c="black", ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        # leg = plt.legend(handles=g, labels=lslabel, bbox_to_anchor=(1.25,0.8), title=title_ls, fontsize=8)
        leg = plt.legend(
            handles=g, labels=lslabel, loc=loc_ls, title=title_ls, fontsize=8
        )
        axes.add_artist(leg)

    if leg_both:
        g = [
            plt.plot([], [], linestyle=linestyle1[i], c=color1[i], ms=6)[0]
            for i, lsi in enumerate(linestyle1)
        ]
        # leg = plt.legend(handles=g, labels=lslabel, bbox_to_anchor=(1.25,0.8), title=title_ls, fontsize=8)
        leg = plt.legend(
            handles=g, labels=lslabel, loc=loc_both, title=title_both, fontsize=8
        )
        axes.add_artist(leg)

    if leg_col:
        if type(colx) == list:
            # ajout d'ume legende pour les courbes :
            g = [
                plt.plot([], [], linestyle="solid", color=color1[i], ms=6)[0]
                for i, coli in enumerate(color1)
            ]
            # leg = plt.legend(handles=g, labels=label1, bbox_to_anchor=(1.25,0.6), title=title_col, fontsize=8)
            leg = plt.legend(
                handles=g, labels=label1, loc=loc_col, title=title_col, fontsize=8
            )
            axes.add_artist(leg)
        else:
            plt.legend(bbox_to_anchor=(1.0, 1))

    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")


# %%

def scat2d(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]
    mp = kwargs["mtype"]
    sp = kwargs["msize"]

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()

    axes.set_title(title1)
    if type(colx) == list:
        if type(coly) != list:
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    # axes.axis("equal")

    if type(colx) != list:
        plt.scatter(df[colx], df[coly], label=label1, marker=mp, s=sp, color=color1)
        axes.scatter(
            df[colx].tolist()[0],
            df[coly].tolist()[0],
            s=5,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            df[colx].tolist()[len(df[colx]) - 1],
            df[coly].tolist()[len(df[coly]) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        [
            plt.scatter(
                xi, y_data[i], label=label1[i], marker=mp, s=sp, color=color1[i]
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[0],
                y_data[i].tolist()[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[len(xi) - 1],
                y_data[i].tolist()[len(y_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)

    plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")


def scat3d(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]
    mp = kwargs["mtype"]
    sp = kwargs["msize"]

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(colx) == list:
        if (type(coly) != list) or (type(colz) != list):
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    if type(colz) == list:
        z_data = [df[coli] for coli in colz]
    else:
        z_data = df[colz]

    if type(colx) != list:
        xmax = np.max(x_data)
        ymax = np.max(y_data)
        zmax = np.max(z_data)
        xmin = np.min(x_data)
        ymin = np.min(y_data)
        zmin = np.min(z_data)
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])
        axes.scatter(
            df[colx],
            df[coly],
            df[colz],
            label=label1,
            marker=mp,
            s=sp,
            color=color1,
            depthshade=False,
        )
        axes.scatter(
            x_data.tolist()[0],
            df[coly].tolist()[0],
            df[colz].tolist()[0],
            s=10,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            x_data.tolist()[len(x_data) - 1],
            df[coly].tolist()[len(y_data) - 1],
            df[colz].tolist()[len(z_data) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        xmax = np.max([np.max(xi) for i, xi in enumerate(x_data)])
        ymax = np.max([np.max(xi) for i, xi in enumerate(y_data)])
        zmax = np.max([np.max(xi) for i, xi in enumerate(z_data)])
        xmin = np.min([np.min(xi) for i, xi in enumerate(x_data)])
        ymin = np.min([np.min(xi) for i, xi in enumerate(y_data)])
        zmin = np.min([np.min(xi) for i, xi in enumerate(z_data)])
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])
        [
            axes.scatter(
                xi,
                y_data[i],
                z_data[i],
                label=label1[i],
                marker=mp,
                s=sp,
                color=color1[i],
                depthshade=False,
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[0],
                y_data[i].tolist()[0],
                z_data[i].tolist()[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[len(xi) - 1],
                y_data[i].tolist()[len(y_data[i]) - 1],
                z_data[i].tolist()[len(z_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)
    axes.set_xlim(lmin, lmax)
    axes.set_ylim(lmin, lmax)
    axes.set_zlim(lmin, lmax)
    axes.set_box_aspect([1, 1, 1])

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    plt.close("all")

def scat3d_pchoc(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    color1 = kwargs["color1"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]
    mp = kwargs["mtype"]
    sp = kwargs["msize"]

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(colx) == list:
        if (type(coly) != list) or (type(colz) != list):
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    if type(colz) == list:
        z_data = [df[coli] for coli in colz]
    else:
        z_data = df[colz]

    if type(colx) != list:
        xmax = np.max(x_data)
        ymax = np.max(y_data)
        zmax = np.max(z_data)
        xmin = np.min(x_data)
        ymin = np.min(y_data)
        zmin = np.min(z_data)
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])
        axes.scatter(
            df[colx],
            df[coly],
            df[colz],
            label=label1,
            marker=mp,
            s=sp,
            color=color1,
            depthshade=False,
        )
        axes.scatter(
            x_data.tolist()[0],
            df[coly].tolist()[0],
            df[colz].tolist()[0],
            s=10,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            x_data.tolist()[len(x_data) - 1],
            df[coly].tolist()[len(y_data) - 1],
            df[colz].tolist()[len(z_data) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        xmax = np.max([np.max(xi) for i, xi in enumerate(x_data)])
        ymax = np.max([np.max(xi) for i, xi in enumerate(y_data)])
        zmax = np.max([np.max(xi) for i, xi in enumerate(z_data)])
        xmin = np.min([np.min(xi) for i, xi in enumerate(x_data)])
        ymin = np.min([np.min(xi) for i, xi in enumerate(y_data)])
        zmin = np.min([np.min(xi) for i, xi in enumerate(z_data)])
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])
        [
            axes.scatter(
                xi,
                y_data[i],
                z_data[i],
                label=label1[i],
                marker=mp,
                s=sp,
                color=color1[i],
                depthshade=False,
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[0],
                y_data[i].tolist()[0],
                z_data[i].tolist()[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[len(xi) - 1],
                y_data[i].tolist()[len(y_data[i]) - 1],
                z_data[i].tolist()[len(z_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    xmax = np.max([np.abs(xi).max() for i,xi in enumerate(x_data)])
    ymax = np.max([np.abs(xi).max() for i,xi in enumerate(y_data)])
    xmax = np.max([xmax,ymax,zmax])
    axes.set_xlim([-xmax,xmax])
    axes.set_ylim([-xmax,xmax])
    axes.set_zlim([zmin,(zmin+4.*(zmax-zmin))])
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)
    # si on veut les axes egaux 
    # axes.set_xlim(lmin, lmax)
    # axes.set_ylim(lmin, lmax)
    # axes.set_zlim(lmin, lmax)
    # axes.set_box_aspect([1, 1, 1])

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    plt.close("all")

def scat3d_colordf(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    colz = kwargs["colz"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    labelz = kwargs["labelz"]
    # color1 = kwargs["color1"]
    dfcol = kwargs["dfcol"]
    view = kwargs["view"]
    loc_leg = kwargs["loc_leg"]
    mp = kwargs["mtype"]
    sp = kwargs["msize"]

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # ax = f.gca(projection='3d')
    axes = Axes3D(f)
    # View :
    axes.view_init(elev=view[0], azim=view[1], vertical_axis="z")

    axes.set_title(title1)
    axes.xaxis.set_pane_color((0, 0, 0, 0))
    axes.yaxis.set_pane_color((0, 0, 0, 0))
    axes.zaxis.set_pane_color((0, 0, 0, 0))
    if type(colx) == list:
        if (type(coly) != list) or (type(colz) != list):
            print(f"Les 3 inputs doivent etre des lists!")
            return
        x_data = [df[coli] for coli in colx]
    else:
        x_data = df[colx]

    if type(coly) == list:
        y_data = [df[coli] for coli in coly]
    else:
        y_data = df[coly]

    if type(colz) == list:
        z_data = [df[coli] for coli in colz]
    else:
        z_data = df[colz]

    if type(colx) != list:
        xmax = np.max(x_data)
        ymax = np.max(y_data)
        zmax = np.max(z_data)
        xmin = np.min(x_data)
        ymin = np.min(y_data)
        zmin = np.min(z_data)
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])

        for indi,rowi in df.iterrows(): 
            # print(rowi)
            # print(rowi[colx])
            # print(rowi[coly])
            axes.scatter(rowi[colx], rowi[coly], rowi[colz], marker=mp, s=sp, color=dfcol.loc[indi]['color'])

        axes.scatter(
            x_data.tolist()[0],
            df[coly].tolist()[0],
            df[colz].tolist()[0],
            s=10,
            c="black",
            marker="o",
            edgecolors="red",
        )
        axes.scatter(
            x_data.tolist()[len(x_data) - 1],
            df[coly].tolist()[len(y_data) - 1],
            df[colz].tolist()[len(z_data) - 1],
            s=10,
            c="yellow",
            marker="o",
            edgecolors="red",
        )
    else:
        xmax = np.max([np.max(xi) for i, xi in enumerate(x_data)])
        ymax = np.max([np.max(xi) for i, xi in enumerate(y_data)])
        zmax = np.max([np.max(xi) for i, xi in enumerate(z_data)])
        xmin = np.min([np.min(xi) for i, xi in enumerate(x_data)])
        ymin = np.min([np.min(xi) for i, xi in enumerate(y_data)])
        zmin = np.min([np.min(xi) for i, xi in enumerate(z_data)])
        lmin = np.min([xmin, ymin, zmin])
        lmax = np.max([xmax, ymax, zmax])
        [
            axes.scatter(
                xi,
                y_data[i],
                z_data[i],
                label=label1[i],
                marker=mp,
                s=sp,
                color=color1[i],
                depthshade=False,
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[0],
                y_data[i].tolist()[0],
                z_data[i].tolist()[0],
                s=10,
                c="black",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]
        [
            axes.scatter(
                xi.tolist()[len(xi) - 1],
                y_data[i].tolist()[len(y_data[i]) - 1],
                z_data[i].tolist()[len(z_data[i]) - 1],
                s=10,
                c="yellow",
                marker="o",
                edgecolors="red",
            )
            for i, xi in enumerate(x_data)
        ]

    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(True)

    plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)
    axes.set_zlabel(labelz)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.zaxis.set_major_formatter(formatter)
    axes.set_xlim(lmin, lmax)
    axes.set_ylim(lmin, lmax)
    axes.set_zlim(lmin, lmax)
    axes.set_box_aspect([1, 1, 1])

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    # code_3D = tikz.get_tikz_code(f, rep_save + title_save + ".tex")
    plt.close("all")

def scat2d_df_colorbar(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    # label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    # color1 = kwargs["color1"]
    leg = kwargs["leg"]
    loc_leg = kwargs["loc_leg"]
    mp = kwargs["mtype"]
    sp = kwargs["msize"]
    colcol = kwargs["colcol"]
    ampl = kwargs["ampl"]
    title_col = kwargs["title_colbar"]
    logcol = kwargs["logcol"]

    # creation d'un dataframe pour la couleur : 
    kcol = {'colx' : colcol, 'ampl' : ampl, 'logcol' : logcol}
    dfcol = color_from_value(df,**kcol)

    f = plt.figure(figsize=(8, 6), dpi=resolution)
    # init subplots :
    gs = gridspec.GridSpec(1,2, width_ratios=[10,0.5])
    # 1st subplot : 
    plt.subplot(gs[0])

    axes = f.gca()

    axes.set_title(title1)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)
    axes.axis("equal")
    axes.set_facecolor("white")
    axes.grid(False)

    for indi,rowi in df.iterrows(): 
        # print(rowi)
        # print(rowi[colx])
        # print(rowi[coly])
        plt.scatter(rowi[colx], rowi[coly], marker=mp, s=sp, color=cmapinf(dfcol.loc[indi]['color']))

    # start & end points :
    print(df[colx].iloc[0])
    print(df[coly].iloc[0])
    axes.scatter(
        # df[colx].head(1),
        # df[coly].head(1),
        df[colx].iloc[0],
        df[coly].iloc[0],
        s=10,
        c="black",
        marker="o",
        edgecolors="red",
    )
    axes.scatter(
        # df[colx].tail(1),
        # df[coly].tail(1),
        df[colx].iloc[-1],
        df[coly].iloc[-1],
        s=10,
        c="yellow",
        marker="o",
        edgecolors="red",
    )
    if leg:
      plt.legend(loc=loc_leg)
    axes.set_xlabel(labelx)
    axes.set_ylabel(labely)

    # 2nd subplot : 
    plt.subplot(gs[1])
    plt.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax=f.gca()
    # remove offset from axis :
        # X axis only :
    # plt.ticklabel_format(useOffset=False)
        # Y axis only :
    # plt.ticklabel_format(style='plain')
        # Both :
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    varcol = df[colcol].drop_duplicates().array.astype(float)
    norm = mpl.colors.Normalize(vmin=np.min(varcol), vmax=np.max(varcol))
    if logcol:
      norm = mpl.colors.Normalize(vmin=np.log10(1.+np.min(varcol)), vmax=np.log10(1.+np.max(varcol)))

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapp),
    #              cax=ax, orientation='vertical', label=title_leg)

    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapinf),
                 cax=ax, orientation='vertical', label=title_col) \
                 .formatter.set_useOffset(False)

    f.tight_layout(pad=0.5)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    plt.close("all")

def pltsub2d_ind(df, **kwargs):
    # si certains inputs manquent, on prend les valeurs par default :
    kwargs = defkwargs | kwargs
    # lecture des arguments :
    title1 = kwargs["tile1"]
    title_save = kwargs["tile_save"]
    ind = kwargs["ind"]
    colx = kwargs["colx"]
    coly = kwargs["coly"]
    rep_save = kwargs["rep_save"]
    label1 = kwargs["label1"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    color1 = kwargs["color1"]
    loc_leg = kwargs["loc_leg"]

    lw = 0.8  # linewidth
    f = plt.figure(figsize=(8, 6), dpi=resolution)
    axes = f.gca()
    axes.set_title(title1)
    if type(ind) == list:
        x_data = [ [df.iloc[indi][colx] for indi in ind] for i,yi in enumerate(coly) ]
        y_data = [ [df.iloc[indi][yi] for indi in ind] for i,yi in enumerate(coly) ]
    else:
        x_data = df.iloc[ind][colx]
        y_data = df.iloc[ind][coly]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    # axes.yaxis.set_major_formatter(StrMethodFormatter("{x:+.4f}"))

    # init subplots :
    gs = gridspec.GridSpec(len(coly),1, width_ratios=[1])
#   loop on subplots :
    for j in np.arange(len(coly)):
        plt.subplot(gs[j])
        [
            plt.plot(x_data[j][i], y_data[j][i], label=label1[i], linewidth=lw, color=color1[i])
            for i,xi in enumerate(x_data[j])
        ]
        if isinstance(label1, (str,list)):
            plt.legend(loc=loc_leg,fontsize=14)
        plt.xlabel(labelx[j],fontsize=14)
        plt.ylabel(labely[j],fontsize=14)
    # plt.tight_layout()
    # axes.set_facecolor('None')
    axes.set_facecolor("white")
    axes.grid(False)
    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")
    # f.savefig(rep_save + title_save + ".ps", bbox_inches="tight", format="ps")
    # tikz.save(rep_save + title_save + ".tex")
    #plt.show()()
    plt.close("all")

#%% heatmap form df
def heatmap(df,**kwargs):
    num_x = kwargs['numx']
    num_y = kwargs['numy']
    colx = kwargs['colx'] 
    coly = kwargs['coly'] 
    colval = kwargs['colval'] 
    agreg = kwargs['agreg']
    # zlim = kwargs['zlim']

    # df = df[(df[coly] <= np.mean(df[coly])+zlim) & (df[coly] >= np.mean(df[coly])-zlim)]
    # Assign each point to a specific cell
    df['Xbin'] = pd.cut(df[colx], bins=np.linspace(kwargs['xlim'][0],kwargs['xlim'][1], num_x + 1), labels=False)
    df['Ybin'] = pd.cut(df[coly], bins=np.linspace(kwargs['ylim'][0],kwargs['ylim'][1], num_y + 1), labels=False)
    
    # Create a new column 'Cell' to represent the cell for each point
    df['Cell'] = list(zip(df['Xbin'], df['Ybin']))
    
    # Calculate the mean value for each cell
    if (agreg=='mean'):
        heatmap_data = df.groupby('Cell')[colval].mean().reset_index()
    if (agreg=='sum'):
        heatmap_data = df.groupby('Cell')[colval].sum().reset_index()
    
    # Create a MultiIndex for the heatmap
    heatmap_data.set_index('Cell', inplace=True)
    
    # Create a grid with all possible cell combinations
    all_cells = pd.MultiIndex.from_product([range(num_x+1), range(num_y+1)], names=['Xbin', 'Ybin'])
    #
    complete_grid = pd.DataFrame(index=all_cells)
    
    # Merge the complete grid with the original heatmap_data
    heatmap_data.index = pd.MultiIndex.from_tuples(heatmap_data.index, names=['Xbin', 'Ybin'])
    
    heatmap_data_complete = heatmap_data.combine_first(complete_grid)

    heatmap_data_complete = heatmap_data_complete.values.reshape(num_x+1,num_y+1)
    # Merge the complete grid with the original heatmap_data
    # heatmap_data_complete = pd.merge(complete_grid, heatmap_data, how='left', left_index=True, right_index=True)
    
    # Reshape the heatmap_matrix into a 2D array : attention on rajoute un transpose.
    
    return heatmap_data_complete
    # return heatmap_data_complete[colval].values.reshape(num_x, num_y)

def plt_heat(df,**kwargs):

    # kwargs = defkwargs | kwargs

    title1 = kwargs["title1"]
    title_save = kwargs["title_save"]
    rep_save = kwargs["rep_save"]
    labelx = kwargs["labelx"]
    labely = kwargs["labely"]
    colval = kwargs["colval"]
    title_col = kwargs["title_colbar"]

    # newxticks = np.linspace(-180.,180.,int(num_th)+1)
    # newyticks = -np.linspace(-1.e-4,1.e-4,int(num_z)+1)
    xticks = np.arange(kwargs['xlim'][0],kwargs['xlim'][1]+20., step=20)
    yticks = np.arange(kwargs['ylim'][0],kwargs['ylim'][1], step=1.e-5)
    # Plot the heatmap with rectangles
    # f = plt.figure(figsize=(10, 8),dpi=600)
    f = plt.figure(figsize=(8, 4),dpi=600)
    gs = gridspec.GridSpec(1,2, width_ratios=[10,0.5])
    # 1st subplot : 
    plt.subplot(gs[0])
    axes = f.gca()
    axes.set_title(title1 + "\n")
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    axes.yaxis.set_major_formatter(formatter)

    axes.set_xlabel(labelx,fontsize=12)
    axes.set_ylabel(labely,fontsize=12)
    cmapp = kwargs["cmap"]
    axes.imshow(np.transpose(kwargs['heatmap']),
    # axes.imshow((heatmap_matrix_2d),
                cmap = cmapp,
                interpolation='nearest', 
                aspect='auto', 
                origin='lower', 
                extent=[1.*min(xticks),
                        1.*max(xticks),
                        1.*min(yticks),
                        1.*max(yticks)])
    axes.set_xticks(xticks)

    plt.subplot(gs[1])

    plt.ticklabel_format(useOffset=False, style='plain', axis='both')

    axes=f.gca()

    axes.ticklabel_format(axis='y', style='plain', useOffset=False)

    varcol = df[colval].drop_duplicates().array.astype(float)
    norm = mpl.colors.Normalize(vmin=np.min(varcol), vmax=np.max(varcol))
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmapp),
                 cax=axes, orientation='vertical') 
    cbar.set_label(label=title_col,size=12) 
    cbar.formatter.set_useOffset(False)

    f.tight_layout(pad=0.5)

    f.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")

    plt.close('all')

def plt_heat_circ(df,**kwargs):

    kwargs = defkwargs | kwargs

    colval = kwargs['colval']
    angle = kwargs['angle']
    num_sectors = kwargs['nbsect']  # 360 degrees / 5 degrees per sector
    title1 = kwargs["title1"]
    title_save = kwargs["title_save"]
    rep_save = kwargs["rep_save"]
    colval = kwargs["colval"]
    title_col = kwargs["title_colbar"]
    agreg = kwargs['agreg']
    # cmapp = kwargs["cmap"]

    df[angle] = (df[angle] + 360) % 360

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},dpi=600)

    ax.set_title(title1 + "\n")

    # Calculate mean values for each sector
    mean_values = []
    for i in range(num_sectors):
        lower_bound = i * (360 / num_sectors)
        upper_bound = (i + 1) * (360 / num_sectors)
        sector_points = df[(df[angle] >= lower_bound) & (df[angle] < upper_bound)]
        if not sector_points.empty:
            if (agreg=='mean'):
                mean_value = sector_points[colval].mean()
            if (agreg=='sum'):
                mean_value = sector_points[colval].sum()
            mean_values.append(mean_value)
        else:
            mean_values.append(np.nan)

    # Plot the colored sectors
    theta = np.linspace(0, 2*np.pi, num_sectors, endpoint=False)
    colors = plt.cm.inferno(mean_values/np.max(mean_values))  # You can use other colormaps
    bars = ax.bar(theta, mean_values, width=(2*np.pi)/num_sectors, align="center", color=colors)

    # Remove legend inside the circle
    ax.legend().set_visible(False)
    ax.set_yticklabels([])

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect("equal")

    # sm = ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=np.nanmin(mean_values), vmax=np.nanmax(mean_values)))
    sm = ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=np.min(mean_values), vmax=np.max(mean_values)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(label=title_col,size=12) 
    cbar.formatter.set_useOffset(False)

    fig.savefig(rep_save + title_save + ".png", bbox_inches="tight", format="png")

    plt.close('all')

# %% spectrogramme :
def spectro(df,**kwargs):
    f = plt.figure(figsize=(8, 6), dpi=600)
    axes = f.gca()

    axes.set_title(kwargs["title1"])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['xpower'], kwargs['xpower']))
    axes.xaxis.set_major_formatter(formatter)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-kwargs['ypower'], kwargs['ypower']))
    axes.yaxis.set_major_formatter(formatter)


    spec, freqs, t, im  = axes.specgram(df[kwargs['x']],NFFT=kwargs['nfft'], Fs=kwargs['fs'],noverlap=kwargs['noverlap'])
    # scipy alternative : 
    # freqs, t, Sxx = signal.spectrogram(df[kwargs['x']],kwargs['fs'], nperseg=kwargs['nfft'])
    # axes.pcolormesh(t,f,Sxx,shading='gouraud')

    # print(f"spectro : len(t) = {len(t)}")
    # axes.set_xlim(xmax=128.)
    axes.set_ylim(ymax=kwargs['ymax'])
    # xmax1=len(df[kwargs['x']])/kwargs['fs']
    # print(f"spectro : xmax1 = {xmax1}")
    # nframes = int((Lsignal - Loverlap) / (Lsegement - Loverlap))
    nframes = int((len(df[kwargs['x']])-kwargs['noverlap'])/(kwargs['nfft']-kwargs['noverlap']))
    axes.set_xlim(xmax=nframes)
    # plt.margins(0,0)
    desired_ticks = np.arange(kwargs['f1'], kwargs['f2']+2, 2)
    # Create a new set of ticks with higher resolution
    new_ticks = np.linspace(0, len(t), len(desired_ticks) * 10)
    # Interpolate the corresponding time values for the new ticks
    interpolated_times = np.interp(new_ticks, np.arange(len(t)), t)
    # # Set the x-axis ticks and labels
    plt.xticks(interpolated_times[::10], desired_ticks)
    # axes.set_xlim(xmin=kwargs['f1'])

    axes.set_xlabel(kwargs['labelx'])
    axes.set_ylabel(kwargs['labely'])
    # plt.show()
    f.savefig(kwargs['rep_save'] + kwargs['title_save'] + ".png", bbox_inches="tight", format="png")
    plt.close("all")