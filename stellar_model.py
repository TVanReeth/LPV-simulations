#!/usr/bin/env python3
#
# File: stellar_model.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module that reads in the necessary quantities from a MESA stellar model

import numpy as np
from astropy import units as u

class stellar_model(object):
    
    def __init__(self, starmodel_file, incl_deg, frot):
        """
            Reading in a stellar model from a GYRE input file (calculated with MESA),
            and setting other (custom) parameter values, including the inclination 
            angle (in degrees) and the cyclic rotation frequency (in d^{-1}) of the
            star.
            
            Parameters:
                starmodel_file:    string
                                   the absolute path to the stellar model file
                incl_deg:          float
                                   the inclination angle of the star in degrees
                frot:              float
                                   the rotation frequency of the star (in d^{-1})
        """
        
        # Initialsing the necessary quantities for the functions below
        # (also serves as an overview of the class variables)
        self._rad = 0.
        self._brunt2 = 0.
        self._Rstar = 0.
        self._teff = 0.
        self._incl_deg = 0.
        self._frot = 0.
        self._theta = 0.
        self._phi = 0.
        self._cell_weight = 0.
        self._theta_incl = 0. 
        self._phi_incl = 0.
        
        self._incl_deg = incl_deg
        self._frot = frot
        
        self._set_surfacegrid()
        self._set_inclined_surface()
        self.read_star(starmodel_file)
        
        return
        
    
    @property
    def rad(self):
        return self._rad
    
    @property
    def brunt2(self):
        return self._brunt2
    
    @property
    def Rstar(self):
        return self._Rstar
    
    @property
    def teff(self):
        return self._teff
    
    @property
    def incl_deg(self):
        return self._incl_deg
    
    @incl_deg.setter
    def incl_deg(self, incl_degrees):
        self._incl_deg = incl_deg
        self._set_inclined_surface(incl_deg)
    
    @property
    def incl(self):
        return self._incl_deg * np.pi / 180.
    
    @property
    def frot(self):
        return self._frot
    
    @frot.setter
    def frot(self, rot_freq):
        self._frot = rot_freq
    
    @property
    def frot_u(self):
        return self._frot / u.d
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def phi(self):
        return self._phi
    
    @property
    def cell_weight(self):
        return self._cell_weight
    
    @property
    def theta_incl(self):
        return self._theta_incl
    
    @property
    def phi_incl(self):
        return self._phi_incl
    
    
    def _set_surfacegrid(self, Ntheta=200, Nphi=400):
        """
            Setting the 2D-surface grid in (theta,phi) in which we 
            want to evaluate the stellar properties and pulsations,
            as well as the necessary cell_weights for when we need
            to calculate selected quantities by integrating over the
            surface.
            
            Parameters:
                Ntheta:    int; optional
                           the number of grid cells in the 
                           latitudinal direction (default: 200)
                Nphi:      int; optional
                           the number of grid cells in the 
                           azimuthal direction (default: 400)
        """
        
        theta = np.linspace(0, np.pi, Ntheta)
        phi = np.linspace(0., 2.*np.pi, Nphi)
        theta_weight = np.ones(len(theta))*np.pi/float(len(theta)-1)
        phi_weight = np.ones(len(phi))*np.pi/float(len(phi)-1)
        theta_weight[0] /= 2.
        theta_weight[-1] /= 2.
        phi_weight[0] /= 2.
        phi_weight[-1] /= 2.
        
        self._theta,self._phi = np.meshgrid(theta,phi)
        theta_weight,phi_weight = np.meshgrid(theta_weight,phi_weight)
        self._cell_weight = theta_weight*phi_weight*np.sin(self._theta) * 4.*np.pi / np.nansum(theta_weight*phi_weight*np.sin(self._theta))
        
        return
    
    
    
    def _set_inclined_surface(self):
        """
            Rotate the stellar model coordinates (in a spherical coordinate frame)
            over the inclination angle.

            NOTE: the rotation is always done around the y-axis. In other words, the
                angle "phi = 0" remains in the same (xz)-plane.
        """
       
        x,y,z = self._spherical_to_cartesian()
        x_rot, y_rot, z_rot = self._rotate_cart(x,y,z,self.incl)
        self._theta_incl, self._phi_incl = self._cartesian_to_spher(x_rot,y_rot,z_rot)
        
        return
    
    
    
    def _spherical_to_cartesian(self):
        """
            Convert spherical surface coordinates to cartesian coordinates.
    
            Returns:
                x:     numpy array
                       the Cartesian coordinates on the x-axis
                y:     numpy array
                       the Cartesian coordinates on the y-axis
                z:     numpy array
                       the Cartesian coordinates on the z-axis
        """
    
        x = np.sin(self._theta) * np.cos(self._phi)
        y = np.sin(self._theta) * np.sin(self._phi)
        z = np.cos(self._theta)
        
        return x,y,z
    
    
    
    def _cartesian_to_spher(self,x,y,z):
        """
            Convert cartesian coordinates to spherical coordinates.
        
            Parameters:
                x:     numpy array
                       the Cartesian coordinates on the x-axis
                y:     numpy array
                       the Cartesian coordinates on the y-axis
                z:     numpy array
                       the Cartesian coordinates on the z-axis
    
            Returns:
                theta: numpy array
                       colatitude (in rad)
                phi:   numpy array
                       azimuthal angle (in rad)
        """
        
        theta = np.arctan2(np.sqrt(x**2. + y**2.),z)
        phi = np.arctan2(y,x)
        
        return theta,phi
        
    
    
    def _rotate_cart(self,x,y,z,i_rot):
        """
            Rotate the stellar model coordinates (in a cartesian frame) over the
            angle i_rot.

            NOTE: the rotation is always done around the y-axis. In other words, the
                  angle "phi = 0" remains in the same (xz)-plane.

            Parameters:
                x:     numpy array
                       the original coordinates on the x-axis
                y:     numpy array
                       the original coordinates on the y-axis
                z:     numpy array
                       the original coordinates on the z-axis
                i_rot: float
                       the angle over which we rotate the model (in rad).
     
            Returns:
                x_rot: numpy array
                       the rotated coordinates on the x-axis
                y_rot: numpy array
                       the rotated coordinates on the y-axis
                z_rot: numpy array
                       the rotated coordinates on the z-axis
    
        """
        
        x_rot = np.cos(i_rot)*x - np.sin(i_rot)*z
        y_rot = y
        z_rot = np.sin(i_rot)*x + np.cos(i_rot)*z
    
        return x_rot, y_rot, z_rot
        
    
    
    def read_star(self, filename):
        """
            Read in a stellar model (i.e. a GYRE input file from MESA)
    
            Parameters:
                filename:    string
                             the filename of the input GYRE model (txt file, as 
                             calculated with MESA)
        """
    
        data = []
    
        ii = 0
        file = open(filename,'r')
        while 1:
            ii += 1
            line = file.readline()
            if not line:
                break
            elif(line.isspace()):
                continue
            elif(ii==1):
                head = list(np.array(line.strip().split(),dtype=float))
                head[0] = int(head[0])
                head[-1] = int(head[-1])
            else:
                line1 = line.replace('D','E')
                data1 = list(np.array(line1.strip().split(),dtype=float))
                data1[0] = int(data1[0])
                data.append(data1)
        head = np.array(head)
        data = np.array(data)
        file.close()
    
        self._rad = data[:,1] * u.cm
        self._brunt2 = data[:,8] * (u.rad / u.s)**2.
    
        self._Rstar = head[2] * u.cm
        self._teff = np.interp(head[2], data[:,1], data[:,5]) * u.K
        
        return
    
    
    
    def limb_darkening(self, mu=0.6):
        """
            Basic linear limb darkening law, with coefficient mu = 0.6 (appropriate
            for F-type stars), applied to the observed stellar surface.
    
            Parameters:
                theta_rot: numpy array
                           the colatitudinal coordinate with respect to the
                           observer's line-of-sight (in rad)
                mu:        float; optional (default = 0.6)
                           linear limb darkening coefficient
    
            Returns:
                ld:        numpy array
                           relative intensity of the observed stellar disk
        """
    
        ld = 1. - mu * (1. - np.cos(self.theta_incl))
        return ld
    
    
    

    def surface_intensity(self, xi_r=0.,xi_t=0.,xi_p=0.,Tr=0.):
        """
            Estimate the relative stellar brightness (bi) variations caused by the
            pulsations. Value = 1: no variations.
    
            NOTE: xi_r, xi_t, xi_p and Rstar must have the same units.
            Tr (Teff variations) and Teff must also have the same units.

            Parameters:
                xi_r:      float (default) or numpy array (with the same shape as
                           theta and phi); optional (default = 0)
                           the radial displacement of the studied pulsation at the
                           stellar surface
                xi_t:      float (default) or numpy array (with the same shape as
                           theta and phi); optional (default = 0)
                           the latitudinal displacement of the studied pulsation at
                           the stellar surface
                xi_p:      float (default) or numpy array (with the same shape as
                           theta and phi); optional (default = 0)
                           the azimuthal displacement of the studied pulsation at
                           the stellar surface
                Tr:        float (default) or numpy array (with the same shape as
                           theta and phi); optional (default = 0)
                           the temperature variation of the studied pulsation at the
                           stellar surface
            Returns:
                bi:        numpy array (with the same shape as theta and phi)
                           the normalised intensity variations at the stellar
                           surface, caused by the studied pulsation
        """
        
        bi = np.ones(self._theta.shape)
        bi = ((self.Rstar.to(u.km).value + xi_r)*bi/self.Rstar.to(u.km).value)**2. * ((self.teff.value + Tr)/self.teff.value)**4.
    
        sel = ~np.isfinite(bi)
        bi[sel] = 1.
    
        return bi
    
        

    def rotational_velocityfield(self):
        """
            Calculate the velocity field of the star in the inertial reference
            frame, caused by the stellar rotation
    
            NOTE 1: the rotation is always done around the y-axis. In other words,
                    the angle "phi = 0" remains in the same (xz)-plane.
        
            Returns:
                radv:  numpy array
                       the radial component of the rotational velocity field in the
                       inertial reference frame (in km/s)
                thv:   numpy array
                       the colatitudinal component of the rotational velocity field
                       in the inertial reference frame (in km/s)
                phv:   numpy array
                       the azimuthal component of the rotational velocity field in
                       the inertial reference frame (in km/s)
        """
        
        radv = np.zeros(self._theta.shape)
        thv =  np.zeros(self._theta.shape)
        phv = 2. * np.pi * self.Rstar.to(u.km).value * np.sin(self._theta) * self.frot_u.to(u.Hz).value
    
        return radv, thv, phv
    
    
    
    def project_velocityfield(self, radv, thv, phv):
        """
            Calculate the radial velocities per grid cell on the stellar surface,
            (including the rotational velocity field) projected along the line-of-sight of the observer

            Parameters:
                radv:      numpy array
                           the radial component of the velocity field of the star in
                           the inertial reference frame
                thv:       numpy array
                           the colatitudinal component of the velocity field of the
                           star in the inertial reference frame
                phv:       numpy array
                           the azimuthal component of the velocity field of the star
                           in the inertial reference frame
    
            Returns:
                proj_vel:  numpy array
                           the velocity field of the star, projected along the
                           line-of-sight of the observer
        """
        
        proj_v_rad = -radv * np.cos(self._theta_incl)
        proj_v_th  = thv * np.sin(self._theta_incl) * np.abs(np.cos(self._phi))
        proj_v_ph  = phv * np.sin(self.incl) * np.sin(self._phi)
    
        proj_vel = proj_v_rad+proj_v_th+proj_v_ph
    
        return proj_vel
    
