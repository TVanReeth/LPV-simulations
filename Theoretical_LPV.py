#!/usr/bin/env python3
#
# File: Theoretical_LPV.py
# Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
# License: GPL-3+
# Description: Module that computes a toy model for line profile variability, caused by high-order
#              g-mode pulsations. This code relies on the TAR, and uses the Hough function
#              calculations written by Vincent Prat.
#        
#              If this code is used, please reference:
#                  Prat et al. 2019
#                  (https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..64P/abstract)
#                  Van Reeth et al. (submitted)
#                  (--)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import astropy.units as u

from stellar_model import stellar_model
from gmode_pulsation import gmode_pulsation

mpl.rc('font',size=14)
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def read_inlist():
    """
        read in the input from the file 'inlist.dat'
        
        Parameters (in the inlist):
            gyre_dir:           string
                                The GYRE 5.x or 6.x installation directory.
            
            starmodel_file:     string
                                the filename of the used MESA stellar structure model
            incl_deg:           float
                                inclination angle of the star (as seen by the observer) in degrees
            frot:               float
                                cyclic (surface) rotation frequency of the star in d^{-1}
                                
            pulsation_filename: string
                                the filename of the GYRE output describing the studied pulsation
            teff_var:           float
                                the assumed temperature amplitude of the simulated g-mode
            vel_var:            float
                                the assumed velocity amplitude of the simulated g-mode
                                
            outdir:             string
                                the directory where all output of this script has to be saved
            savefilename:       string
                                the name of the file where the main output of this script has to be saved
                                
            line_depth:         float
                                depth of the spectral line for the non-rotating starting model
            vtherm:             float
                                thermal velocity broadening (microturbulence; in km/s)
            rv_min:             float
                                min. radial velocity (in km/s)
            rv_max:             float
                                max. radial velocity
            N_rv:               integer
                                number of sampled RV points
        
        Returns:
            star:               stellar_model object
                                the studied stellar model
            gmode:              gmode_pulsation object
                                the studied g-mode pulsation
            outdir:             string
                                the directory where all output of this script has to be saved
            savefilename:       string
                                the name of the file where the main output of this script has to be saved
            line_depth:         float
                                depth of the spectral line for the non-rotating starting model
            vtherm:             float
                                thermal velocity broadening (microturbulence; in km/s)
            vel_min:            float
                                min. radial velocity (in km/s)
            vel_max:            float
                                max. radial velocity
            nvel:               integer
                                number of sampled RV points
    """
    
    inlist_file = open('./LPV_inlist.dat', 'r')
    lines = inlist_file.readlines()
    inlist_file.close()
    
    gyre_dir = ''
    outdir = ''
    starmodel_file = ''
    pulsation_filename = ''
    savefilename = ''
    incl_deg = 0.
    frot = 0.
    teff_var = 0.
    vel_var = 0.
    
    line_depth = 0.15
    vtherm = 2.
    vel_min = -300.
    vel_max = 300.
    nvel = 6001
    
    for line in lines:
        if(len(line) > 0):
            var_val = line.strip().split('#')[0]
            
            if('gyre_dir' == var_val.strip()[:8]):
                gyre_dir = var_val.split('=')[1].strip()
            if('outdir' == var_val.strip()[:6]):
                outdir = var_val.split('=')[1].strip()
            
            if('starmodel_file' == var_val.strip()[:14]):
                starmodel_file = var_val.split('=')[1].strip()
            
            if('pulsation_filename' == var_val.strip()[:18]):
                pulsation_filename = var_val.split('=')[1].strip()
            
            if('savefilename' == var_val.strip()[:12]):
                savefilename  = var_val.split('=')[1].strip()
            
            if('incl_deg' == var_val.strip()[:8]):
                incl_deg  = float(var_val.split('=')[1].strip())
            
            if('frot' == var_val.strip()[:4]):
                frot  = float(var_val.split('=')[1].strip())
            
            if('teff_ampl' == var_val.strip()[:9]):
                teff_var  = float(var_val.split('=')[1].strip())
            
            if('vel_ampl' == var_val.strip()[:8]):
                vel_var  = float(var_val.split('=')[1].strip())
            
            if('line_depth' == var_val.strip()[:10]):
                line_depth  = float(var_val.split('=')[1].strip())
            
            if('vtherm' == var_val.strip()[:6]):
                vtherm  = float(var_val.split('=')[1].strip())
            
            if('rv_min' == var_val.strip()[:6]):
                vel_min  = float(var_val.split('=')[1].strip())
            
            if('rv_max' == var_val.strip()[:6]):
                vel_max  = float(var_val.split('=')[1].strip()) 
            
            if('N_rv' == var_val.strip()[:4]):
                nvel  = int(var_val.split('=')[1].strip())
    
    star = stellar_model(starmodel_file, incl_deg, frot)
    gmode = gmode_pulsation(gyre_dir, pulsation_filename, teff_var, vel_var)
    
    return star, gmode, outdir, savefilename, line_depth, vtherm, vel_min, vel_max, nvel



def base_lineprofile(depth, vtherm):
    """
        An initial (gaussian) line profile with a given depth(relative to a
        continuum value = 1) and thermal broadening

        NOTE: this line profile is "upside down", with the continuum at zero
              instead of 1 (for easier calculations later).

        Parameters:
            depth:  float
                    the depth of the gaussian line profile(relative to a
                    continuum value = 1)
            vtherm: float
                    the thermal broadening of the gaussian line profile

        Returns:
            rv_base:   numpy array
                       the Doppler velocity domain of the calculated Gaussian
                       line profile
            prof_base: numpy array
                       the gaussian line profile (as a function of rv_base)
    """

    rv_base = np.linspace(-20.,20.,401)
    prof_base = depth*np.exp(-0.5*(rv_base/vtherm)**2.)
    prof_base[0] = 0.
    prof_base[-1] = 0.

    return rv_base, prof_base



def calculate_line(star, line_depth, vtherm, vel_min, vel_max, nvel, gmode=None, puls_phase=0.):
    """
        Calculate the line profile of a rotating, pulsating star.

        Parameters:
            star:          stellar model object
                           our studied stellar model
            line_depth:    float
                           depth of the spectral line for the non-rotating starting model
            vtherm:        float
                           thermal velocity broadening (microturbulence; in km/s)
            vel_min:       float
                           min. radial velocity (in km/s)
            vel_max:       float
                           max. radial velocity (in km/s)
            nvel:          integer
                           number of sampled RV points
            gmode:         gmode_pulsation object; optional
                           our studied g-mode pulsation (default: None)
            puls_phase:    float; optional
                           The fractional pulsation phase (one full pulsation
                           cycle = 1)

        Returns:
            obs_vel:       numpy array, float
                           the velocity range in which the line profile is
                           calculated
            obs_line:      the "flux"  associated with the line profile
    """

    # calculating the rotational velocity field (uniform, perfectly symmetric rotation only - for now...)
    radv,thv,phv = star.rotational_velocityfield()

    if(gmode is None): # no pulsation info included = set everything to zero
        proj_vel = star.project_velocityfield(radv,thv,phv)
    else: # calculating the pulsation geometry properties if the info is included
        xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr = gmode.calculate_displacement(star, puls_phase)
        # project the stellar velocity field along the observer's line-of-sight
        proj_vel = star.project_velocityfield(radv + u_r,thv + u_t,phv + u_p)

    # Calculate the brightness distribution over the observed stellar surface, and integrate everything
    obs_vel = np.linspace(vel_min, vel_max, nvel)
    obs_line = np.zeros(obs_vel.shape)
    base_vel,base_line = base_lineprofile(line_depth, vtherm)
    ld = star.limb_darkening()
    
    if(gmode is None): # no pulsation info included = set everything to zero
        bi = star.surface_intensity()
    else:
        bi = star.surface_intensity(xi_r=xi_r,xi_t=xi_t,xi_p=xi_p,Tr=Tr)

    # This sum is an adapted version of the equation given by H. Saio et al. (2018), which they used to calculate the mode visibilities of r-modes.
    for ith in np.arange(len(star.theta_incl[:,0])):
        for iph in np.arange(len(star.theta_incl[0,:])):
            if(star.theta_incl[ith,iph] < np.pi/2.):
                obs_line += np.interp(obs_vel,proj_vel[ith,iph] + base_vel, 2.*np.cos(star.theta_incl[ith, iph])*star.cell_weight[ith, iph]*ld[ith,iph]*bi[ith, iph] * base_line)
    obs_line = 1. - obs_line  #Invert the line profile, so the continuum is at one
    
    return obs_vel, obs_line



def calculate_moment(vel,line,n_order=1):
    """
        Calculate the variation of the 1st-moment of a selected LSD profile

        Parameters:
            vel:     numpy array
                     radial velocities
            line:    numpy array
                     the LSD flux (as a function of the radial velocities)
            norder:  int/float; optional
                     the order of the moment that will be calculated (default: 1)
        Returns:
            moment:  float
                     the moment of order n_order
    """

    moment = np.nansum((vel**n_order)*(1.-line)) / np.nansum(1.-line)
    return moment



def calculate_amplitudes_phases(puls_phases,vel,line_profiles):
    """
        Calculate the amplitude and phase variability of the LSD profile as a
        function of the pulsation phase

        Parameters:
            puls_phases:    numpy array
                            sampled phases of the g-mode pulsation cycle
            vel:            numpy array
                            radial velocities
            line_profiles:  numpy array
                            the deformed line profiles (at the studied pulsation
                            phases) sampled at the specified radial velocities
                            "vel"

        Returns:
            amp:            numpy array
                            the amplitude profile of the g-mode, as a function
                            of the sampled radial velocities "vel"
            phase:          numpy array
                            the phase profile of the g-mode, as a function
                            of the sampled radial velocities "vel"
    """

    matrix = []
    matrix.append(np.sin(2.*np.pi*puls_phases))
    matrix.append(np.cos(2.*np.pi*puls_phases))
    matrix = np.array(matrix).T

    amp = []
    phase = []
    for ipix in np.arange(len(vel)):
        if(np.std(line_profiles.T[ipix]) > 10.**(-7.)):
            par = np.linalg.inv(matrix.T @ matrix) @ matrix.T @ line_profiles.T[ipix]
            amp.append(np.sqrt(par[0]**2. + par[1]**2.))
            phc = np.arccos(par[0]/amp[-1])
            phs = np.arcsin(par[1]/amp[-1])

            if(phc <= np.pi/2.):
                phase.append(phs)
            elif((phc > np.pi/2.) & (phs >= 0.)):
                phase.append(phc)
            else:
                phase.append(-np.pi - phs)
        else:
            amp.append(0.)
            phase.append(0.)

    amp = np.array(amp)
    phase = np.array(phase)

    sel = np.r_[(amp > 0.) & (phase != 0.)]
    phasel = phase[sel]
    for jiter in np.arange(10):
        for ipix in np.arange(1,len(vel[sel])):
            dp = np.abs(phasel[ipix]-phasel[ipix-1])
            if(np.abs(phasel[ipix]-phasel[ipix-1] - 2.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] - 2.*np.pi
            elif(np.abs(phasel[ipix]-phasel[ipix-1] + 2.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] + 2.*np.pi
            elif(np.abs(phasel[ipix]-phasel[ipix-1] - 4.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] - 4.*np.pi
            elif(np.abs(phasel[ipix]-phasel[ipix-1] + 4.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] + 4.*np.pi
            elif(np.abs(phasel[ipix]-phasel[ipix-1] - 6.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] - 6.*np.pi
            elif(np.abs(phasel[ipix]-phasel[ipix-1] + 6.*np.pi) < dp):
                phasel[ipix] = phasel[ipix] + 6.*np.pi

    phasel -= np.nanmean(phasel)

    phase[sel] = phasel

    return amp, phase




    
if __name__ == "__main__":

    # Setting the variables
    star, gmode, outdir, savefilename, line_depth, vtherm, vel_min, vel_max, nvel = read_inlist()

    # Calculating the base line profile
    base_vel, base_line = calculate_line(star, line_depth, vtherm, vel_min, vel_max, nvel)

    # Calculating the deformed line profiles
    puls_phases = np.linspace(0.,1.,21)[:-1]
    mom1 = []
    all_lines = []

    for iph,puls_phase in enumerate(puls_phases):
        obs_vel, obs_line = calculate_line(star, line_depth, vtherm, vel_min, vel_max, nvel, gmode=gmode, puls_phase=puls_phase)
        all_lines.append(obs_line-base_line)
        mom1.append(calculate_moment(obs_vel, obs_line))

    all_lines = np.array(all_lines)
    mom1 = np.array(mom1)
    amp, phase = calculate_amplitudes_phases(puls_phases,base_vel,all_lines)
    sel_line = np.r_[phase != 0.]
    
    
    ### Saving the amplitude and phase profiles of the LPV
    np.savetxt(f'{outdir}{savefilename}', np.array([base_vel[sel_line], base_line[sel_line], amp[sel_line], phase[sel_line]/(2.*np.pi) - np.nanmean(phase[sel_line]/(2.*np.pi))]).T)
    
    
    ### Plotting the calculated profiles
    fig = plt.figure(1,figsize=(6,7))
    plt.subplots_adjust(left=0.16,right=0.95,top=0.95,hspace=0.1)
    ax1 = fig.add_subplot(311)
    plt.xticks([-150.,-100.,-50.,0.,50.,100.,150.],['','','','','','',''])
    plt.xlim(-160.,160.)
    plt.ylim(0.94,1.002)

    ax2 = fig.add_subplot(312)
    plt.xticks([-150.,-100.,-50.,0.,50.,100.,150.],['','','','','','',''])
    plt.xlim(-160.,160.)
    plt.ylim(0.,0.0025)
    ax3 = fig.add_subplot(313)
    plt.xlim(-160.,160.)
    ax3.plot(base_vel[sel_line],phase[sel_line]/(2.*np.pi) - np.nanmean(phase[sel_line]/(2.*np.pi)),'k-')
    plt.ylim(-1.,1.)

    ax1.plot(base_vel[sel_line], base_line[sel_line], 'k-')
    ax2.plot(base_vel[sel_line],amp[sel_line],'k-')

    ax1.set_ylabel('mean LSD')
    ax2.set_ylabel('amplitude')
    ax3.set_xlabel(r'velocity ($\sf km\,s^{-1}$)')
    ax3.set_ylabel(r'phase (2$\sf\pi$ rad)',labelpad=15)

    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50.))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10.))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.025))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))

    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50.))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10.))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.001))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.0002))

    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(50.))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(10.))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    ### Saving the figures
    figname = f'{os.path.splitext(savefilename)[0]}_FPF'
    plt.savefig(f'{outdir}{figname}.pdf')
    plt.savefig(f'{outdir}{figname}.png')
    
    plt.show()
    
