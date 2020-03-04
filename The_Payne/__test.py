import numpy as np
import matplotlib.pyplot as plt
from training import *

data = np.load('other_data\\kurucz_training_spectra.npz')

spectra = data['spectra']
labels  = data['labels']

print(spectra.shape)
print(labels.shape)

#    training_labels has the dimension of [# training spectra, # stellar labels]
#    training_spectra has the dimension of [# training spectra, # wavelength pixels]

#plt.plot(labels[0,:], labels[-1,:], '.')
#plt.show()

N = spectra.shape[0]

train_spec, train_lbl, valid_spec, valid_lbl = [],[],[],[]
for i in range(N):
    if np.random.rand()>0.1:
        train_spec.append(spectra[i,:])
        train_lbl.append(labels[:,i])
    else:
        valid_spec.append(spectra[i,:])
        valid_lbl.append(labels[:,i])

train_spec = np.array(train_spec)
train_lbl = np.array(train_lbl)
valid_spec = np.array(valid_spec)
valid_lbl = np.array(valid_lbl)

print(train_spec.shape)
print(valid_spec.shape)
print(train_lbl.shape)
print(valid_lbl.shape)

NNT = NNTrain()

NNT.train(train_lbl, train_spec, valid_lbl, valid_spec)

        
