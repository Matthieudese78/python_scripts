#!/bin/python3
#%%
import numpy as np
import pandas as pd
import glob
import os
# import subprocess
defkwargs = {'name_save' : 'result'}
#%%
# rep_load = f"../data/"
# rep_save = f"./pickle/"
def csv2pickle(**kwargs):
    kwargs = defkwargs | kwargs
    rep_load = kwargs['rep_load']
    rep_save = kwargs['rep_save']
    if not os.path.exists(rep_load):
        print(f"load FOLDER : {rep_load} doesn't exists!")
    
    if not os.path.exists(rep_save):
        os.makedirs(rep_save)
        print(f"FOLDER : {rep_save} created.")
    else:
        print(f"FOLDER : {rep_save} already exists.")
    all_files = glob.glob(rep_load + "*.csv")
    print(f"all_files = {all_files}")
    # i=0
    for filename in all_files:
        print(filename)
        df = pd.read_csv(filename,delimiter=(','),header=0,low_memory=False)
        print(np.shape(df))
        df = df.replace(r'^\s*$', np.nan, regex=True)
        # df = df.replace(r'^\s*$', regex=True)
        df.columns = df.columns.str.replace(' ', '')
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        # print(df)
        df.to_pickle( f"{rep_save}{kwargs['name_save']}.pickle")

def csvs2pickle(**kwargs):
    rep_load = kwargs['rep_load']
    rep_save = kwargs['rep_save']
    if not os.path.exists(rep_load):
        print(f"load FOLDER : {rep_load} doesn't exists!")
    
    if not os.path.exists(rep_save):
        os.makedirs(rep_save)
        print(f"FOLDER : {rep_save} created.")
    else:
        print(f"FOLDER : {rep_save} already exists.")
    all_files = glob.glob(rep_load + "*.csv")
    print(f"all_files = {all_files}")
    # i=0
    dfs = []
    for filename in all_files:
        print(filename)
        df = pd.read_csv(filename,delimiter=(','),header=0)
        print(np.shape(df))
        df = df.replace(r'^\s*$', np.nan, regex=True)
        # df = df.replace(r'^\s*$', regex=True)
        df.columns = df.columns.str.replace(' ', '')
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        # print(df)
        dfs.append(df)
    combined_df = pd.concat(dfs,ignore_index=True)
    combined_df.to_pickle( f"{rep_save}result.pickle")
# %% test :
# df2 = pd.read_pickle(f"{rep_save}result.pickle")

# %%
