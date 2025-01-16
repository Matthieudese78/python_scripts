#!/bin/python3

# exb = np.array([0., -1., 0.])
# eyb = np.array([0., 0., 1.])
# ezb = np.array([-1., 0., 0.])

# name_cols = ['uxplam_h2','uyplam_h2','uzplan_h2']
# name_cols1 = ['uxplam_h','uyplam_h','uzplan_h']

# EX = 0. -1. 0. ; 
# EY = 0. 0. 1.  ;
# EZ = (-1.) 0. 0. ;
#%%
import numpy as np
import pandas as pd
#%%
def repchg(df,**kwargs):
  exa = np.array([1., 0., 0.])
  eya = np.array([0., 1., 0.])
  eza = np.array([0., 0., 1.])
  ua = np.zeros(3)
  ua[0] = df[kwargs['colx']]
  ua[1] = df[kwargs['coly']]
  ua[2] = df[kwargs['colz']]
  exb   = kwargs['exb'] 
  eyb   = kwargs['eyb'] 
  ezb   = kwargs['ezb'] 
  uxb   = ua[0]*(exa @ exb) + ua[1]*(eya @ exb) + ua[2]*(eza @ exb) 
  uyb   = ua[0]*(exa @ eyb) + ua[1]*(eya @ eyb) + ua[2]*(eza @ eyb) 
  uzb   = ua[0]*(exa @ ezb) + ua[1]*(eya @ ezb) + ua[2]*(eza @ ezb) 
  return uxb, uyb, uzb

def repchg_mat(df,**kwargs):
  exa = np.array([1., 0., 0.])
  eya = np.array([0., 1., 0.])
  eza = np.array([0., 0., 1.])
  exb   = kwargs['exb'] 
  eyb   = kwargs['eyb'] 
  ezb   = kwargs['ezb'] 
  P = np.array([[(exa @ exb) , (eya @ exb) , (eza @ exb)], 
                [(exa @ eyb) , (eya @ eyb) , (eza @ eyb)], 
                [(exa @ ezb) , (eya @ ezb) , (eza @ ezb)]])
  return np.dot(np.dot(P,df[kwargs['mat']]),np.transpose(P))

def repchgdf(df,**kwargs):
  exb = kwargs['base2'][0]
  eyb = kwargs['base2'][1]
  ezb = kwargs['base2'][2]
  name_cols  = kwargs['name_cols']
  dict = {'ub' : df.apply(repchg, colx=name_cols[0], coly=name_cols[1], colz=name_cols[2], \
                                  exb=exb, eyb=eyb, ezb=ezb, axis=1)}
  df2 = pd.DataFrame(dict)
  df2[name_cols] = pd.DataFrame(df2.ub.tolist(), index=df2.index)
  df2.drop(['ub'],inplace=True,axis=1)
  df.loc[df2.index, name_cols] = df2

def repchgdf_mat(df,**kwargs):
  exb = kwargs['base2'][0]
  eyb = kwargs['base2'][1]
  ezb = kwargs['base2'][2]
  name_cols  = kwargs['name_cols']
  mat  = kwargs['mat']
  dict = {name_cols[0] : df.apply(repchg_mat, \
                          exb=exb, eyb=eyb, ezb=ezb, mat=mat, axis=1)}
  df2 = pd.DataFrame(dict)
  # df2[name_cols] = pd.DataFrame(df2.ub.tolist(), index=df2.index)
  # df2.drop(['ub'],inplace=True,axis=1)
  df.loc[df2.index, name_cols] = df2

