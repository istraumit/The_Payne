import numpy as np
import matplotlib.pyplot as plt
from training import *

NNT = NNTrain()

NNT.train_on_npz('other_data\\kurucz_training_spectra.npz')

        
