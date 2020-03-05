import sys, os
import numpy as np
import matplotlib.pyplot as plt
from training import *

bs = int(sys.argv[1])

NNT = NNTrain(batch_size=bs, batch_size_valid=bs)

NNT.train_on_npz(os.path.join('other_data','kurucz_training_spectra.npz'), validation_fraction=0.5)




