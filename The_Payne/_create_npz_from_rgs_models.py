import os
import numpy as np

def params_from_name(fn):
   arr = fn[:-4].split('_')
   return [float(arr[1]), 0.01*float(arr[2]), float(arr[6])]

fluxes = []
params = []
for fn in os.listdir('.'):
    if not fn.endswith('.rgs'): continue
    p = params_from_name(fn)
    data = np.loadtxt(fn)
    wave = data[:,0]
    fluxes.append(data[:,1])
    params.append(p)
    print(fn, 'processed')

fluxes = np.array(fluxes)
params = np.array(params)

np.savez('_GRID.npz', flux=fluxes, labels=params, wvl=wave)





