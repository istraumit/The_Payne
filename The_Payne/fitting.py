# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import chebval
import utils
from network import Network

class Fit:

    def __init__(self, network:Network, Cheb_order, tol=5.e-4):
        self.tol = tol # tolerance for when the optimizer should stop optimizing.
        self.network = network
        self.Cheb_order = Cheb_order

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
        nnl = self.network.num_labels()
        num_labels = nnl + self.Cheb_order + 1

        def fit_func(dummy_variable, *labels):
            norm_spec = self.network.get_spectrum_scaled(scaled_labels = labels[:nnl])
            norm_spec = utils.doppler_shift(self.network.wave, norm_spec, labels[-1])
            Cheb_coefs = labels[nnl : nnl + self.Cheb_order]
            Cheb_x = np.linspace(-1, 1, len(norm_spec))
            Cheb_poly = chebval(Cheb_x, Cheb_coefs)
            spec_with_resp = norm_spec * Cheb_poly
            return np.interp(wavelength, self.network.wave, spec_with_resp)

        # if no initial guess is supplied, initialize with the median value
        if p0 is None:
            p0 = np.zeros(num_labels)

        # prohibit the minimimizer to go outside the range of training set
        bounds = np.zeros((2,num_labels))
        bounds[0,:nnl] = -0.5
        bounds[1,:nnl] = 0.5
        bounds[0, nnl : nnl + self.Cheb_order] = -np.inf
        bounds[1, nnl : nnl + self.Cheb_order] = np.inf
        bounds[0,-1] = -5.
        bounds[1,-1] = 5.

        # run the optimizer
        popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = p0,
                    bounds = bounds, ftol = self.tol, xtol = self.tol, absolute_sigma = True, method = 'trf')
        model_spec = fit_func([], *popt)

        x_min = self.network.x_min
        x_max = self.network.x_max
        # rescale the result back to original unit
        popt[:nnl] = (popt[:nnl]+0.5)*(x_max-x_min) + x_min
        pcov[:nnl,:nnl] = pcov[:nnl,:nnl]*(x_max-x_min)
        
        def chi2_func(labels):
            model = fit_func([], *labels)
            diff = (norm_spec - model) / spec_err
            chi2 = np.sum(diff**2)
            return chi2
        
        return popt, pcov, model_spec, chi2_func












