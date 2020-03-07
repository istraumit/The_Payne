# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit
import utils
from network import Network

class Fit:

    def __init__(self, network:Network, tol=5.e-4):
        self.tol = tol # tolerance for when the optimizer should stop optimizing.
        self.network = network

    def run(self, wavelength, norm_spec, spec_err, mask=None, p0 = None):
        '''
        fit a single-star model to a single combined spectrum

        p0 is an initial guess for where to initialize the optimizer. Because
            this is a simple model, having a good initial guess is usually not
            important.

        labels = [Teff, Logg, Vturb [km/s],
                [C/H], [N/H], [O/H], [Na/H], [Mg/H],\
                [Al/H], [Si/H], [P/H], [S/H], [K/H],\
                [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],\
                [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H],\
                C12/C13, Vmacro [km/s], radial velocity

        returns:
            popt: the best-fit labels
            pcov: the covariance matrix, from which you can get formal fitting uncertainties
            model_spec: the model spectrum corresponding to popt
        '''

        # set infinity uncertainty to pixels that we want to omit
        if mask != None:
            spec_err[mask] = 999.

        # number of labels + radial velocity
        num_labels = self.network.num_labels() + 1

        def fit_func(dummy_variable, *labels):
            norm_spec = self.network.get_spectrum_scaled(scaled_labels = labels[:-1])
            norm_spec = utils.doppler_shift(wavelength, norm_spec, labels[-1])
            return norm_spec

        # if no initial guess is supplied, initialize with the median value
        if p0 is None:
            p0 = np.zeros(num_labels)

        # prohibit the minimimizer to go outside the range of training set
        bounds = np.zeros((2,num_labels))
        bounds[0,:] = -0.5
        bounds[1,:] = 0.5
        bounds[0,-1] = -5.
        bounds[1,-1] = 5.

        # run the optimizer
        popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = p0,
                    bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')
        model_spec = fit_func([], *popt)

        x_min = self.network.x_min
        x_max = self.network.x_max
        # rescale the result back to original unit
        popt[:-1] = (popt[:-1]+0.5)*(x_max-x_min) + x_min
        pcov[:-1,:-1] = pcov[:-1,:-1]*(x_max-x_min)
        return popt, pcov, model_spec
