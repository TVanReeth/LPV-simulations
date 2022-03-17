#!/usr/bin/env python3
#
# File: gmode_pulsation.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module that reads in the necessary quantities from a GYRE g-mode pulsation

import numpy as np
import sys
from astropy import units as u

from Hough import hough_functions

class gmode_pulsation(object):
    
    def __init__(self, gyre_dir, pulsation_filename, teff_var, vel_var):
        """
            Reading in a g-mode pulsation, calculated with GYRE v5.x or v6.x, as well as
            additional parameters describing the g-mode.
            
            Parameters:
                gyre_dir:           string
                                    The GYRE 5.x or 6.x installation directory.
                pulsation_filename: string
                                    the filename of the GYRE output describing the studied pulsation
                teff_var:           float
                                    the assumed temperature amplitude of the simulated g-mode
                vel_var:            float
                                    the assumed velocity amplitude of the simulated g-mode
        """
        
        self._puls_freq = 0.
        self._xir_xih_ratio = 0.
        self._n = 0.
        self._l = 0.
        self._k = 0.
        self._m = 0.
        self._lam_fun = 0.
        
        self._omval = 0.
        self._lmdb = 0.
        self._Hr = 0.
        self._Ht = 0.
        self._Hp = 0.
        
        self.read_pulsation(pulsation_filename, gyre_dir)
        self._teff_var = teff_var
        self._vel_var = vel_var
        
        return
    
    
    @property
    def puls_freq(self):
        return self._puls_freq
    
    @property
    def xir_xih_ratio(self):
        return self._xir_xih_ratio
    
    @property
    def n(self):
        return self._n 
    
    @property
    def k(self):
        return self._k
    
    @property
    def l(self):
        return self._l
    
    @property
    def m(self):
        return self._m
    
    def lam(self, spin):
        """
            Calculate the eigenvalue Lambda of the Laplace Tidal Equation
            that corresponds to the given spin value for the mode identi-
            fication (k,m) of the studied g-mode.
            
            Parameters:
                spin:   float
                        the spin parameter value s = 2*nu_rot/nu_co, where
                        nu_rot is the stellar rotation frequency and nu_co
                        is the pulsation frequency in the corotating frame.
            
            Returns:
                lambda: float
                        the eigenvalue lambda corresponding to the given
                        spin
        """
        
        return self._lam_fun(spin)
        
    
    def read_pulsation(self, pulsation_filename, gyre_dir):
        """
            Read in the relevant information from a g-mode pulsation calculated with GYRE
            
            Parameters:
                pulsation_filename:    string
                                       the path to the g-mode pulsation file from GYRE
                gyre_dir:              string
                                       the GYRE v5.x or v6.x installation directory
        """
        
        self._puls_freq, self._xir_xih_ratio, self._n, self._k, self._l, self._m = self._read_gyre_file(pulsation_filename)
        self._lam_fun = self._retrieve_lambda(gyre_dir, self._k, self._m)
        
        return 
        
    
    def _read_gyre_file(self,filename):
        """
            Read in a g-mode pulsation, calculated with GYRE
            
            Parameters:
                filename:      string
                               name of the file containing the g-mode pulsation info
                
            Returns:
                puls_freq:     astropy quantity
                               the cyclic g-mode pulsation frequency (in the inertial reference frame)
                xir_xih_ratio: float
                               the ratio of the vertical and the horizontal displacement associated 
                               with the g-mode, measured at the stellar surface
                nval:          int
                               radial order of the g-mode
                kval:          int
                               latitudinal degree of the g-mode
                lval:          int
                               spherical degree of the g-mode
                mval:          int
                               azimuthal order of the g-mode
        """
        
        gennames = []
        genval = []
    
        ii = 0
        file = open(filename,'r')
        while 1:
            ii += 1
            line = file.readline()
            if(ii < 5):
                if(ii == 3):
                    gennames = line.strip().split()
                    dtype = {'names' : gennames, 'formats' : len(gennames)*['f8']}
                elif(ii == 4):
                    genval_str = line.strip().split()
                    if('freq_units' in gennames):
                        index_units = gennames.index('freq_units')
                        genval_str[index_units] = -1
                    genval = np.array(genval_str,dtype=float)
            else:
                break
        file.close()
    
        globaldata = np.rec.fromarrays(genval, dtype = dtype)
        profiledata = np.genfromtxt(filename, skip_header=5, names=True)
        
        assert "n_pg" in globaldata.dtype.names, "STOP: please ensure that the radial order n_pg is included in the GYRE output file for the calculated g-mode."
        assert "l" in globaldata.dtype.names, "STOP: please ensure that the spherical degree l is included in the GYRE output file for the calculated g-mode."
        assert "m" in globaldata.dtype.names, "STOP: please ensure that the azimuthal order m is included in the GYRE output file for the calculated g-mode."
        assert "Re(freq)" in globaldata.dtype.names, "STOP: please redo the GYRE computations and include the pulsation frequency (in the inertial reference frame) in the output."
        assert "Rexi_r" in profiledata.dtype.names, "STOP: please redo the GYRE computations and include the radial displacement (in the inertial reference frame) in the output."
        assert "Rexi_h" in profiledata.dtype.names, "STOP: please redo the GYRE computations and include the horizontal displacement (in the inertial reference frame) in the output."
    
        puls_freq = globaldata['Re(freq)'] / u.d
        xir = profiledata['Rexi_r']
        xih = profiledata['Rexi_h']
        xir_xih_ratio = xir[-1] / xih[-1]
        
        nval = int(globaldata['n_pg'])
        lval = int(globaldata['l'])
        mval = int(globaldata['m'])
        
        kval = lval - abs(mval)
        
        return puls_freq, xir_xih_ratio, nval, kval, lval, mval
        
        
    
    def _retrieve_lambda(self, gyre_dir, kval, mval):
        """
            Retrieving the function lambda(nu) given in GYRE v5.x or v6.x.
    
            Parameters:
                gyre_dir: string
                          The GYRE 5.x or 6.x installation directory.
                kval:     int/float
                          latitudinal degree of the g-mode
                mval:     int/float
                          azimuthal order of the g-mode
    
            Returns:
                lam_fun: function
                         A function to calculate lambda, given spin parameter
                         values as input.
        """

        if(kval >= 0):
            kstr = f'+{kval}'
        else:
            kstr = f'{kval}'
        if(mval >= 0):
            mstr = f'+{mval}'
        else:
            mstr = f'{mval}'
    
        infile = f'{gyre_dir}/data/tar/tar_fit.m{mstr}.k{kstr}.h5'
    
        sys.path.append(gyre_dir+'/src/tar/') 
        import gyre_tar_fit
        import gyre_cheb_fit
    
        tf = gyre_tar_fit.TarFit.load(infile)
        lam_fun = np.vectorize(tf.lam)
    
        return lam_fun
    
    
    
    def calculate_puls_geometry(self, star):
        """
            A wrapper around the Hough function calculation by Vincent Prat, to get
            the 2D geometry of the g-mode pulsation on the surface of this star.
    
            Parameters:
                star:       stellar model object
                            our studied stellar model

            Computes:
                omval:      float
                            the angular pulsation frequency in the corotating
                            reference frame (unit: rad/s)
                lmbd:       float
                            the corresponding eigenvalue of the Laplace Tidal
                            Equation (associated with the calculated pulsation)
                hr_mapped:  numpy array
                            the radial component of the Hough function (as a
                            function of theta)
                ht_mapped:  numpy array
                            the latitudinal component of the Hough function (as a
                            function of theta)
                hp_mapped:  numpy array
                            the azimuthal component of the Hough function (as a
                            function of theta)
        """
        
        freqval_co = abs(self._puls_freq - self._m*star.frot_u)
        
        spinval = float(2.*star.frot_u.to(1./u.d)/freqval_co.to(1./u.d))
    
        if(self._m < 0.):
            omval = -2.*np.pi * freqval_co.to(1./u.s).value  
        else:
            omval = 2.*np.pi * freqval_co.to(1./u.s).value

        lmbd_est = -self._lam_fun(spinval)
        (lmbd_min, mu, hr, ht, hp) = hough_functions(spinval, self._l, self._m, lmbd=lmbd_est,npts=1000)
        lmbd = -lmbd_min
    
        hr_mapped = np.ones(star.theta.shape)
        ht_mapped = np.ones(star.theta.shape)
        hp_mapped = np.ones(star.theta.shape)
    
        for irow in np.arange(len(star.theta[:,0])):
            hr_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],hr[::-1])
            ht_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],ht[::-1])
            hp_mapped[irow][::-1] = np.interp(np.cos(star.theta[irow,::-1]),mu[::-1],hp[::-1])
        
        self._omval = omval
        self._lmdb = lmbd
        self._Hr = hr_mapped
        self._Ht = ht_mapped
        self._Hp = hp_mapped
        
        return



    def calculate_displacement(self, star, puls_phase):
        """
            Converting the geometry of the calculated g-mode pulsation to temperature
            variations, Lagrangian displacements and the associated velocity field

            Parameters:
                star:           stellar model object
                                our studied stellar model
                puls_phase:     float
                                the current phase of the studied pulsation (as seen
                                by the observers)

            Returns:
                xi_r:           numpy array
                                radial component of the scaled g-mode displacement at
                                the stellar surface
                xi_t:           numpy array
                                latitudinal component of the scaled g-mode displacement
                                at the stellar surface
                xi_p:           numpy array
                                azimuthal component of the scaled g-mode displacement to
                                the stellar surface
                u_r:            numpy array
                                radial component of the scaled g-mode velocity field at
                                the stellar surface
                u_t:            numpy array
                                latitudinal component of the scaled g-mode velocity
                                field at the stellar surface
                u_p:            numpy array
                                azimuthal component of the scaled g-mode velocity
                                field at the stellar surface
                Tr:             numpy array
                                scaled temperature variability of the g-mode at the
                                stellar surface
        """
        
        if(self._m < 0.):
            sign = 1.
        else:
            sign = -1.

        # calculate the pulsation mode geometry
        if(type(self._Hr) == float):
            self.calculate_puls_geometry(star)
    
        Tr = self._teff_var * (self._Hr * np.cos(self._m*star.phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(self._Hr * np.cos(self._m*star.phi + 2.*np.pi*sign*puls_phase))
        xi_r = -self._xir_xih_ratio / self._omval * self._Hr * np.cos(self._m*star.phi + 2.*np.pi*sign*puls_phase)
        xi_t =  1. / self._omval * self._Ht * np.cos(self._m*star.phi + 2.*np.pi*sign*puls_phase)
        xi_p = -1. / self._omval * self._Hp * np.sin(self._m*star.phi + 2.*np.pi*sign*puls_phase)

        u_r =   self._xir_xih_ratio * self._Hr * np.sin(self._m*star.phi + 2.*np.pi*sign*puls_phase)
        u_t =  -1. * self._Ht * np.sin(self._m*star.phi + 2.*np.pi*sign*puls_phase)
        u_p =  -1. * self._Hp * np.cos(self._m*star.phi + 2.*np.pi*sign*puls_phase)

        # rescaling the displacement
        extra_scaling = np.nanmax(np.sqrt(u_r**2. + u_t**2. + u_p**2.))
        u_r  = u_r  * self._vel_var / extra_scaling   # the time step dt of both velocities drops away in the division
        u_t  = u_t  * self._vel_var / extra_scaling
        u_p  = u_p  * self._vel_var / extra_scaling
        xi_r = xi_r * self._vel_var / extra_scaling
        xi_t = xi_t * self._vel_var / extra_scaling
        xi_p = xi_p * self._vel_var / extra_scaling
        
        return xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr
    
    
