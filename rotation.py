#!/bin/python3
#%%
import numpy as np
import numpy.linalg as LA
import pandas as pd 
#%%
def pv(v1,v2):
    pv1 = np.array([v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]])
    return pv1

    # produit scalaire
def scal(v1,v2):
    ps1 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    return ps1

    # produit quaternion
def pquat(x, y):
    s1 = x[0]
    a1 = x[1]
    b1 = x[2]
    c1 = x[3]
    v1 = np.array([a1, b1, c1])
    s2 = y[0]
    a2 = y[1]
    b2 = y[2]
    c2 = y[3]
    v2 = np.array([a2, b2, c2])
    pxy1 = np.array([s1*s2- scal(v1,v2) , s1*v2 + s2*v1 + np.cross(v1,v2) ] )
    return np.array([ pxy1[0], pxy1[1][0], pxy1[1][1], pxy1[1][2] ])

    # CHAPEAU
def chap(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return np.array([[0. , -x3 , x2 ], [x3 , 0. , -x1], [-x2 , x1 , 0 ]])

    # QUAT_2_MAT

def quat2mat(x):
    # d'abord normalisation:
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    if (LA.norm(x[1:])<1.e-10):
      return np.eye(3)
    else:
      x = x/LA.norm(x)
      return 2*np.array([[x0**2+x1**2-0.5, x1*x2-x3*x0, x1*x3+x2*x0],[x2*x1+x3*x0, x0**2+x2**2-0.5, x2*x3-x1*x0],[x3*x1-x2*x0,x3*x2+x1*x0,x0**2+x3**2-0.5]])
    # v = np.array([x1, x2, x3])
    # M = (2.*s**2-1.)*np.eye(3) + 2.*s*chap(v) + 2.*np.dot(v.reshape((3,1)),v.reshape((1,3))) --> formule matircielle

    # MAT_2_QUAT
def mat2quat(M): # ---> Spurrier algorithm    
    if np.abs(M.trace()) >= np.max([ np.abs(M[0][0]), np.abs(M[1][1]), np.abs(M[2][2]) ]):
        q0 = 0.5*np.sqrt(1. + M.trace() )
        B = (M - np.transpose(M))/(4*q0)
        a =  np.array( [-B[1][2] , B[0][2] , -B[0][1] ] )
        q = np.array([q0 , a[0], a[1], a[2]])
    else:
        # print("Spurrier")
        q = np.zeros(4)
        l_indice = np.array([0,1,2,0,1,2])
        l = np.array([M[0][0], M[1][1], M[2][2]])
        i = np.where(l == l.max())[0]
        j = l_indice[i + 1][0] 
        k = l_indice[i + 2][0] 
        i = i[0]
        q[i + 1] = np.sqrt( l[i]/2 + (1-M.trace())/4 )
        q[0] = (M[k][j]-M[j][k])/4*q[i+1]
        q[j + 1] = (M[j][i]+M[i][j])/4*q[i+1]
        q[k + 1] = (M[k][i]+M[i][k])/4*q[i+1]
        
    q = q/LA.norm(q)
    return q

def vect2quat(u):
    q = np.zeros(4)
    q[0] = np.cos(LA.norm(u)/2)
    # ---> on anticipe un problème de conditionnement norme(vecteur rotation) trop petite
    if LA.norm(u) < 1e-20: 
        x = LA.norm(u)/2
        # dvlpmt Taylor de sin(x)/x:
        q[1:] = 0.5*u*(1 - x**2/2 + x**4/120 - x**6/5040 + x**8/362880 - x**10/39916800 ) 
    else:
        q[1:] = np.sin(LA.norm(u)/2)*u/LA.norm(u) 
    # normalisation:
    q1 = q/LA.norm(q)
    return q1    
        
    # MAT_2_vROT
def Mrot2Vrot(M):
    theta = np.arcsin((M[0][0] + M[1][1] + M[2][2] - 1)/2)
    v = np.array( [M[2][1] - M[1][2], M[0][2] - M[2][0], M[1][0] - M[0][1] ])/(2*np.sin(theta))
    return theta*v

# formule exponentielle d'une matrice de rotation:
def exp(u):
    if LA.norm(u)<1e-20:
        M = np.eye(3)
    else: 
        theta = LA.norm(u) # angle rotation
        v = u/LA.norm(u) # vecteur unitaire
        M = np.eye(3) + np.sin(theta)*chap(v) + 2*np.sin(theta/2)**2*(LA.matrix_power(chap(v),2))
    return M

    # Matrice différenciation d'un vecteur rotation theta*u:     
def T(u):
    if LA.norm(u) < 1e-20:
            T = np.eye(3)
    else:
            e = np.array([u/LA.norm(u)]) # = vecteur unitaire directeur de la rotation incrémentale
            theta = LA.norm(u) # = angle rotation incrémentale
            x = (theta/2)
            # devlpmt en série de Taylor de x1 = tan(x)/x : 
            x1 = np.tan(x)/x
            # x1  = 1 + (1/3)*x**2 + (2/15)*x**4 + (17/315)*x**6 + (62/2835)*x**8  
            
            T = np.dot(np.transpose(e),e) + (1/x1)*(np.eye(3)-np.dot(np.transpose(e),e)) + 0.5*chap(u)   
    return T

    # quat_2_vect
def quat2vect(q):
    # d'abord normalisation:
    q = q/LA.norm(q)
    if LA.norm(q[1:]) < 1e-20:
        v = np.array([0,0,0])
    else:
        theta = 2*np.arcsin(LA.norm(np.array(q[1:]))) # angle de rotation
        v = theta*(np.array(q[1:])/LA.norm(np.array(q[1:]))) # angle*vecteur unitaire
    return v

def quat2vect2(q):
    # d'abord normalisation:
    q = q/LA.norm(q)
    if LA.norm(q[1:]) < 1e-20:
        v = np.array([0,0,0])
    else:
        # theta = 2*np.arcsin(LA.norm(np.array(q[1:]))) # angle de rotation
        theta = 2.*np.arctan2(LA.norm(np.array(q[1:])),q[0]) # angle de rotation
        v = theta*(np.array(q[1:])/LA.norm(np.array(q[1:]))) # angle*vecteur unitaire
    return v

# quat2mat : dataframe
def quat2mat_df(df,**kwargs):
#   q1 = kwargs['q1']
#   q2 = kwargs['q2']
#   q3 = kwargs['q3']
#   q4 = kwargs['q4']
#   return quat2mat(np.array([df[q1],df[q2],df[q3],df[q4]]))
  x = np.array([df[kwargs['q1']],df[kwargs['q2']],df[kwargs['q3']],df[kwargs['q4']]]).astype(float)
  x0 = x[0]
  x1 = x[1]
  x2 = x[2]
  x3 = x[3]
#   if (LA.norm(x[1:])<1.e-20):
#     return np.eye(3)
#   else:
  x = x/LA.norm(x)
  return 2.*np.array([[x0**2+x1**2-0.5, x1*x2-x3*x0, x1*x3+x2*x0],[x2*x1+x3*x0, x0**2+x2**2-0.5, x2*x3-x1*x0],[x3*x1-x2*x0,x3*x2+x1*x0,x0**2+x3**2-0.5]])
  
def q2mdf(df,**kwargs):
    colname = kwargs['colname']
    dict1 = {colname : df.apply(quat2mat_df, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df.loc[df1.index, colname] = df1
    return "Done"

def trdf(df,**kwargs):
    x = kwargs['x']
    y = kwargs['y']
    z = kwargs['z']
    rot = kwargs['mrot']
    M = df[rot]
    X = np.array([df[x],df[y],df[z]])
    return np.dot(np.transpose(M),X)

def b2a(df,**kwargs):
    colnames = kwargs['colnames']
    dict1 = {'trm' : df.apply(trdf, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df1[colnames] = pd.DataFrame(df1.trm.tolist(),index=df1.index)
    df1.drop(['trm'],inplace=True,axis=1)
    df.loc[df1.index, colnames] = df1
    return "Done"

def spinextr(df,**kwargs):
    mat = df[kwargs['mat']]
    if (np.abs(mat[2,2])>1.):
        print(f"mat_33 > 1 !! = {mat[2,2]}")
        mat = np.round(mat,5)
    theta = np.sign(mat[2,1])*np.arccos(mat[2,2])
    # print(f"sin(theta) = {np.sin(theta)}")
    crit = 1.e-2
    if (np.abs(theta)<=crit): 
        return np.arctan2(mat[1,0],mat[1,1])
    #     return np.arctan2(mat[2,1],mat[1,1])
    if (np.abs(theta)>crit):
        return np.arcsin(mat[2,0]/np.sin(theta))

def spinextrdf(df,**kwargs):
    colnames = kwargs['colnames']
    dict1 = {colnames[0] : df.apply(spinextr, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    # df1[colnames] = pd.DataFrame(df1.angle.tolist(),index=df1.index)
    # df1.drop(['angle'],inplace=True,axis=1)
    df.loc[df1.index, colnames] = df1
    return "Done"

def recopoint(df,**kwargs):
    mat = df[kwargs['mat']]
    u = np.dot(mat,kwargs['point'])
    return u[0], u[1], u[2]

def recopointdf(df,**kwargs):
    dict1 = {'col1' : df.apply(recopoint, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df1[kwargs['colnames']] = pd.DataFrame(df1.col1.tolist(), index=df1.index)
    df.loc[df1.index, kwargs['colnames']] = df1
    return "Done"
# %% energies :
def epot(df,**kwargs):
    # print(kwargs["masse"])
    # print(df[kwargs["altitude"]])
    m = kwargs["masse"]
    z = df[kwargs["altitude"]]
    # print(m) 
    # print(z)
    return m*9.81*z

def epotdf(df,**kwargs):
    dict1 = {kwargs['colname'] : df.apply(epot, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df.loc[df1.index, kwargs['colname']] = df1
    return "Done"

def etot(df,**kwargs):
    return df[kwargs["ecin"]]+df[kwargs["epot"]]

def etotdf(df,**kwargs):
    dict1 = {kwargs['colname'] : df.apply(etot, **kwargs, axis=1)}
    df1 = pd.DataFrame(dict1)
    df.loc[df1.index, kwargs['colname']] = df1
    return "Done"