#!/bin/python3
#%% 
import numpy as np
import random
#%%
def find_pivot(liste):
  xf = liste[0]
  xm = liste[int(np.floor(len(liste)/2.))]
  xl = liste[-1]
  return np.sort([xf,xm,xl])[1]
#%% definition liste :
liste = random.sample(range(20),17)
print(f"liste = {liste}")
  # find pivot :
xf = liste[0]
xm = liste[int(np.floor(len(liste)/2.))]
xl = liste[-1]
print(f"xf = {xf}")
print(f"xm = {xm}")
print(f"xl = {xl}")
pivot = find_pivot(liste)
ipiv = liste.index(pivot)
print(f"pivot = {pivot}")
print(f"ipiv = {ipiv}")
# quicksort :
  # objectif : right number < pivot < left nbr :
  # on met le pivot a la fin de la liste :
liste.append(pivot)
liste.__delitem__(ipiv)
print(f"liste pivot at the end : {liste}")

#%% on cherche le premier chiffre en partant de la gauche > pivot :
ileft = 0
iright = -2
while ileft<iright:
  for i,el in enumerate(liste):
    if el>pivot:
      larger_from_left = el   
      ileft = i
      break
  for i,el in enumerate(liste):
    ind = -i-2
    if liste[ind]<pivot:
      smaller_from_right = liste[ind]
      iright = ind 
      break
  # si ileft > iright : on change larger form left <--> pivot :
  if ileft>iright: 
    liste[-1] = larger_from_left 
    liste[ipiv] = pivot 
find_pivot(liste)

  # on check 
# puis on change de pivot
  # on intervertit le larger_form_left et le smaller from right :
liste[ileft]  = smaller_from_right  
liste[iright] = larger_from_left  
print(f"larger_from_left : {larger_from_left}")
print(f"smaller_from_right : {smaller_from_right}")
print(f"liste 1st iteration : {liste}")

# %%
