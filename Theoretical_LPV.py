"""
	Calculating a toy model for line profile variability, caused by high-order g-mode pulsations
	This code relies on the TAR, and uses the Hough function calculations written by Vincent Prat.
	
	If this code is used, reference: Prat et al. 2019
					 https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..64P/abstract

	Author: Timothy Van Reeth
		timothy.vanreeth@kuleuven.be
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import astropy.units as u
import os

from amigo_simple.gmode_series import asymptotic
from amigo_simple.stellar_model import stellar_model
from Hough_VPrat import hough_functions

mpl.rc('font',size=14)
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def spherical_to_cartesian(theta,phi,rad=None):
    """
    	Convert spherical coordinates to cartesian coordinates.
    """
    
    if(rad is None):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
    else:
        x = rad * np.sin(theta) * np.cos(phi)
        y = rad * np.sin(theta) * np.sin(phi)
        z = rad * np.cos(theta)
    
    return x,y,z



def cartesian_to_spher(x,y,z):
    """
    	Convert cartesian coordinates to spherical coordinates.
    """
    
    rad = np.sqrt(x**2. + y**2. + z**2.)
    theta = np.arctan2(np.sqrt(x**2. + y**2.),z)    # NOT using arccos, because it was not sufficiently precise...
    phi = np.arctan2(y,x)
    
    return rad,theta,phi
    


def rotate_cart(x,y,z,incl):
    """
    	Rotate cartesian coordinates over an angle incl.
	NOTE: the rotation is always done around the y-axis. In other words, the angle "phi = 0" remains in the same (xz)-plane.
    """
    
    x_rot = np.cos(incl)*x - np.sin(incl)*z
    y_rot = y
    z_rot = np.sin(incl)*x + np.cos(incl)*z
    
    return x_rot, y_rot, z_rot



def rotate_spher(theta, phi, incl, rad=None):
    """
    	Rotate spherical coordinates over an angle incl.
	NOTE: the rotation is always done around the y-axis. In other words, the angle "phi = 0" remains in the same (xz)-plane.
    """
    
    x,y,z = spherical_to_cartesian(theta,phi,rad=rad)
    x_rot, y_rot, z_rot = rotate_cart(x,y,z,incl)
    rad_rot, theta_rot, phi_rot = cartesian_to_spher(x_rot,y_rot,z_rot)
    
    if(rad is None):
        return theta_rot, phi_rot
    else:
        return rad_rot, theta_rot, phi_rot
    


def base_lineprofile(depth=0.15):
    """
    	An initial (gaussian) line profile with amplitude 0.15 (relative to a continuum value = 1)
	NOTE: this line profile is "upside down", with the continuum at zero instead of 1 (for easier calculations later).
    """
    
    rv_base = np.linspace(-20.,20.,401)
    prof_base = depth*np.exp(-0.5*(rv_base/4.)**2.)
    prof_base[0] = 0.
    prof_base[-1] = 0.
    
    return rv_base, prof_base



def rotational_velocityfield(rad, theta, phi, f_rot):
    """
    	Rotate spherical coordinates over an angle incl.
	NOTE 1: the rotation is always done around the y-axis. In other words, the angle "phi = 0" remains in the same (xz)-plane.
	NOTE 2: the rotation frequency does not have an astropy unit here, and is assumed to be in Hertz.
    """
    
    radv = np.zeros(rad.shape)
    thv =  np.zeros(rad.shape)
    phv = 2. * np.pi * rad * np.sin(theta) * f_rot
    
    return radv,thv,phv



def project_velocityfield(radv,thv,phv,theta_rot,phi_rot,theta,phi,incl):
    """
        Calculate the radial velocities per cell on the stellar surface, projected along the line-of-sight of the observer
    """
    
    proj_v_rad = -radv * np.cos(theta_rot)
    proj_v_th  = thv * np.sin(theta_rot) * np.abs(np.cos(phi))
    proj_v_ph  = phv * np.sin(incl) * np.sin(phi)
    
    return proj_v_rad+proj_v_th+proj_v_ph



def limb_darkening(theta_rot, mu=0.6):
    """
        Basic linear limb darkening law, with mu = 0.6 (appropriate for F-type stars).
    """
    ld = 1. - mu * (1. - np.cos(theta_rot))
    return ld



def puls_intensity(star,theta, phi, xi_r=0.,xi_t=0.,xi_p=0.,Tr=0.,Teff=7000.):
    """
        Roughly estimate the relative stellar brightness (bi) variations caused by the pulsations. Value = 1: no variations.
        NOTE: xi_r, xi_t, xi_p are assumed to be in km. Tr (Teff variations) and Teff are in Kelvin.
    """
    bi = np.ones(theta.shape)
    bi =  ((star.radius.to(u.km)[-1].value + xi_r)*bi/star.radius.to(u.km)[-1].value)**2. * ((Teff+Tr)/Teff)**4.
    sel = ~np.isfinite(bi)
    bi[sel] = 1.
    
    return bi



def calculate_line(star, incl, frot_u, asym=None, azim_view=0.,puls_phase=0.,nval=50, teff_var=10.,vel_var=5.,xir_xih_ratio=-1, saveplot=False):
    """
        Calculate the line profile of a rotating/pulsating star.
	
	Parameters:
		star:		stellar model file (from the amigo package)
		incl:		float
				The inclination angle of the star (in rad)
		frot_u: 	float (with astropy unit)
				the cyclic stellar rotation frequency

		asym:		stellar g-mode series object (from the amigo package)
				Optional; if asym is None, the line profile of a non-pulsating star is calculated.
		azim_view:	float
				The azimuthal viewing angle of the observer (in rad)
		puls_phase:	float
				The fractional pulsation phase (one full pulsation cycle = 1)
		nval:		int
				the radial order of the studied pulsation
		teff_var:	float
				The scaling factor for the Teff-variations caused by the pulsations (in Kelvin)
		vel_var:	float
				The scaling factor for the velocity variations on the stellar surface caused by the pulsations (in km/s)
		xir_xih_ratio:	scaling the ratio between the radial and horizontal components of the Lagrangian displacement xi
				When xir_xih_ratio = -1, the theoretical value omega_co/N is used 
				(with omega_co the angular pulsation frequency in the co-rotating frame, and N the Brunt-Vaisala frequency)
		saveplot:	boolean (default: False)
				Do you want to save a figure of the calculated line profile and of the pulsation (as seen by the observr)
				NOTE: there are bugs in the matplotlib quiver command, which means that this often fails... 
				      To assist, it is assumed 
	
	Returns:
		fpuls:		float
				the cyclic pulsation frequency in the inertiel reference frame (unit: cycle_per_day)
		omval:		float
				the angular pulsation frquency in the corotating reference frame
		lmbd:		float
				the corresponding eigenvalue of the Lapalce Tidal Equation (associated with the calculated pulsation)
		obs_vel:	numpy array, float
				the velocity range in which the line profile is calculated
		obs_line:	the "flux"  associated with the line profile
    """
    
    # setting up base quanities for the 3D stellar model
    Rstarkm = star.radius.to(u.km).value[-1]
    f_rot = frot_u.to(1/u.s).value
    
    theta = np.linspace(0, np.pi, 200)
    phi = np.linspace(azim_view, azim_view+2.*np.pi, 400)
    theta_weight = np.ones(len(theta))*np.pi/float(len(theta)-1)
    phi_weight = np.ones(len(phi))*np.pi/float(len(phi)-1)
    theta_weight[0] /= 2.
    theta_weight[-1] /= 2.
    phi_weight[0] /= 2.
    phi_weight[-1] /= 2.
    
    theta,phi = np.meshgrid(theta,phi)
    theta_weight,phi_weight = np.meshgrid(theta_weight,phi_weight)
    cell_weight = theta_weight*phi_weight*np.sin(theta) * 4.*np.pi / np.nansum(theta_weight*phi_weight*np.sin(theta))
    
    rad = Rstarkm * np.ones(theta.shape)
    
    # calculating the rotational velocity field (uniform, perfectly symmetric rotation only - for now...)
    radv,thv,phv = rotational_velocityfield(rad, theta, phi, f_rot)
    
    if(asym is None): # no pulsation info included = set everything to zero
        xi_r = 0.
        xi_t = 0.
        xi_p = 0.
        u_r = 0.
        u_t = 0.
        u_p = 0.
        Tr = 0.
        fpuls = 0.
        omval = 0.
        lmbd = 0.
    else: # calculating the pulsation geometry properties if the info is included
        fpuls, omval, lmbd, xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr = calculate_pulsations(star, frot_u, theta, phi, teff_var, vel_var,xir_xih_ratio,asym, nval, puls_phase)
    
    # rotate the stellar model so the rotation axis no longer coincides with the line-of-sight of the observer
    rad_rot, theta_rot, phi_rot = rotate_spher(theta, phi, incl, rad=rad)
    
    # project the stellar velocity field along the observer's line-of-sight 
    proj_vel = project_velocityfield(radv + u_r,thv + u_t,phv + u_p,theta_rot,phi_rot,theta,phi,incl)
    
    # Calculate the brightness distribution over the observed stellar surface, and integrate everything
    obs_vel = np.linspace(-300.,300.,6001)
    obs_line = np.zeros(obs_vel.shape)
    base_vel,base_line = base_lineprofile()
    ld = limb_darkening(theta_rot)
    bi = puls_intensity(star,theta,phi,xi_r=xi_r,xi_t=xi_t,xi_p=xi_p,Tr=Tr)
    
    # This sum is an adapted version of the equation given by H. Saio et al. (2018), to calculate the mode visibilities of r-modes.
    for ith in np.arange(len(theta_rot[:,0])):
        for iph in np.arange(len(theta_rot[0,:])):
            if(theta_rot[ith,iph] < np.pi/2.):
                obs_line += np.interp(obs_vel,proj_vel[ith,iph] + base_vel, 2.*np.cos(theta_rot[ith, iph])*cell_weight[ith, iph]*ld[ith,iph]*bi[ith, iph] * base_line)
    obs_line = 1. - obs_line  #Invert the line profile, so the continuum is at one
    
    if((not asym is None) & saveplot):
        fig = plt.figure(figsize=(10,5))
        plt.subplots_adjust(left=0.0,right=0.95,wspace=0.25,bottom=0.13,top=0.92)
        plot_pulsation(star, frot_u, incl, theta, phi, Tr, teff_var, vel_var, xir_xih_ratio, asym, nval, puls_phase, fig, 121)
        plt.subplot(122)
        plt.plot(obs_vel, obs_line, 'k-',lw=1.5)
        plt.xlim(-160.,160.)
        plt.ylim(0.945,1.005)
        plt.ylabel('LSD profile')
        plt.xlabel(r'velocity ($\sf km\,s^{-1}$)')
        if not os.path.exists('./LPV_snapshots/'):
            os.makedirs('./LPV_snapshots/')
        figname = f'n{nval}k{int(asym.kval)}m{int(asym.mval)}_frot{round(frot_u.value*1000)}_pulsphase{round(1000*puls_phase)}'
        plt.savefig(f'./LPV_snapshots/{figname}.png')
        
    return fpuls, omval, lmbd, obs_vel, obs_line
    


def calculate_puls_geometry(asym,nval,frot,Pi0,theta,phi):
    """
        A wrapper around the Hough function calculation by Vincent Prat, to get the 2D geometry of the g-mode pulsation
    """
    
    lval = abs(asym.mval) + abs(asym.kval)
    patt_freq = asym.uniform_pattern(frot,Pi0,alpha_g=0.5,unit='cycle_per_day')
    
    ind = np.argmin((asym.nvals - float(nval))**2.)
    fpuls = patt_freq.value[ind]
    if(asym.kval < 0):
        fpuls *= -1.
        freqval_co = abs(fpuls + asym.mval*frot.value)
    else:
        freqval_co = abs(fpuls - asym.mval*frot.value)
    spinval = 2.*frot.value/freqval_co
    if(asym.mval < 0.):
        omval = -2.*np.pi*freqval_co/86400.
    else:
        omval = 2.*np.pi*freqval_co/86400.
    
    lmbd_est = -asym.lam_fun(spinval)
    
    (lmbd, mu, hr, ht, hp) = hough_functions(spinval, lval, asym.mval, lmbd=lmbd_est,npts=1000)
    
    hr_mapped = np.ones(theta.shape)
    ht_mapped = np.ones(theta.shape)
    hp_mapped = np.ones(theta.shape)
    
    for irow in np.arange(len(theta[:,0])):
        hr_mapped[irow][::-1] = np.interp(np.cos(theta[irow,::-1]),mu[::-1],hr[::-1])
        ht_mapped[irow][::-1] = np.interp(np.cos(theta[irow,::-1]),mu[::-1],ht[::-1])
        hp_mapped[irow][::-1] = np.interp(np.cos(theta[irow,::-1]),mu[::-1],hp[::-1])
        
    return fpuls, omval, -lmbd, hr_mapped, ht_mapped, hp_mapped



def calculate_pulsations(star, frot, theta, phi, teff_var, vel_var, xir_xih_ratio, asym, nval, puls_phase):
    """
        Converting the geomety of the g-mode pulsation to temperature variations, Lagrangian displacements and the associated velocity field
    """
    
    if(asym.mval < 0.):
        sign = 1.
    else:
        sign = -1.
    # calculate Pi0
    Pi0 = star.buoyancy_radius()
    
    # calculate the pulsation mode geometry
    fpuls, omval, lmbd, hr, ht, hp = calculate_puls_geometry(asym,nval,frot,Pi0,theta,phi)
    non_convective = star.brunt.value > 0.
    
    if(xir_xih_ratio == -1):
        xir_xih_ratio = np.abs(np.sqrt(lmbd) * omval / star.brunt[non_convective][-1].value)
    
    if(asym.kval < 0.):
        Tr = teff_var * (hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase))
        
        xi_r = -teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_t =  teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * ht * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_p =  teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hp * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        
        u_r =   teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval)     * hr * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_t =   teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval)     * ht * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        u_p =  -teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval)     * hp * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
    
    
    else:
        Tr = teff_var * (hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase))
        
        xi_r = -teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_t =  teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * ht * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_p = -teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hp * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        
        u_r =   teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval)     * hr * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_t =  -teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval)     * ht * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_p =  -teff_var                 / (star.radius.to(u.km)[non_convective][-1].value * omval)     * hp * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        
    
    # rudimentary scaling?
    extra_scaling = np.nanmax(np.sqrt(u_r**2. + u_t**2. + u_p**2.))
    if(extra_scaling > 0.):
        u_r  = u_r  * vel_var / extra_scaling   # the time step dt of both velocities drops away in the division
        u_t  = u_t  * vel_var / extra_scaling
        u_p  = u_p  * vel_var / extra_scaling
        xi_r = xi_r * vel_var / extra_scaling
        xi_t = xi_t * vel_var / extra_scaling
        xi_p = xi_p * vel_var / extra_scaling
        
    
    return fpuls, omval, lmbd, xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr



def plot_pulsation(star, frot, incl, theta, phi, Tr, teff_var, vel_var, xir_xih_ratio, asym, nval, puls_phase, fig, axisid):
    """
        A fairly crude routine to plot the 2D-view of the stellar pulsation (with the temperature indicating colours, and the arrowds indicating the Lagrangian displacements)
    """
    
    azim = 0.
    degshift = 90. - (incl * 180./np.pi)
    vminval = -1.  # keep for now
    vmaxval = 2.   # keep for now
    
    x,y,z = spherical_to_cartesian(theta, phi)
            
    fcolors_r = Tr
    fmax_r, fmin_r = fcolors_r.max(), fcolors_r.min()
    fcolors_ra = (fcolors_r - fmin_r)/(fmax_r - fmin_r)
    
    ax = fig.add_subplot(axisid, projection='3d')
    ax.set_aspect('equal')
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic_r(fcolors_ra),vmin=vminval, vmax=vmaxval)
    ax.set_axis_off()
    
    theta_rot0 = np.linspace(0.,np.pi,20)     
    phi_rot0 = np.linspace(-0.5*np.pi,0.5*np.pi,15)
    theta0,phi0 = np.meshgrid(theta_rot0,phi_rot0)
    
    fpuls0, omval0, lmbd0, xi_r0,xi_t0,xi_p0,u_r0,u_t0,u_p0,Tr0 = calculate_pulsations(star, frot, theta0, phi0, teff_var, vel_var, xir_xih_ratio, asym, nval, puls_phase)
    
    """
    xih_len = np.sqrt(xi_t0**2. + xi_p0**2.)
    xih_max, xih_min = xih_len.max(), xih_len.min()
    xih_norm = (xih_len - xih_min)/(xih_max - xih_min)
        
    xit_sc = xi_t0 * xih_norm / xih_len
    xip_sc = xi_p0 * xih_norm / xih_len
        
    x0,y0,z0 = spherical_to_cartesian(theta0,phi0)
    ur = 0.1 * ( xit_sc * np.cos(theta0) * np.cos(phi0) - xip_sc * np.sin(phi0) )
    vr = 0.1 * ( xit_sc * np.cos(theta0) * np.sin(phi0) + xip_sc * np.cos(phi0) )
    wr = 0.1 * ( -xit_sc * np.sin(theta0) )
    """
    
    xi_len = np.sqrt(xi_r0**2. + xi_t0**2. + xi_p0**2.)
    xi_max, xi_min = xi_len.max(), xi_len.min()
    xi_norm = (xi_len - xi_min)/(xi_max - xi_min)
        
    xit_sc = xi_t0 * xi_norm / xi_len
    xip_sc = xi_p0 * xi_norm / xi_len
    xir_sc = xi_r0 * xi_norm / xi_len
        
    x0,y0,z0 = spherical_to_cartesian(theta0,phi0)
    ur = 0.15 * ( xit_sc * np.cos(theta0) * np.cos(phi0) - xip_sc * np.sin(phi0)  + xir_sc * np.sin(theta0)*np.cos(phi0))
    vr = 0.15 * ( xit_sc * np.cos(theta0) * np.sin(phi0) + xip_sc * np.cos(phi0)  + xir_sc * np.sin(theta0)*np.sin(phi0))
    wr = 0.15 * ( -xit_sc * np.sin(theta0) + xir_sc * np.cos(theta0))
    
    ax.quiver(x0, y0, z0,  1.1*ur, 1.1*vr, 1.1*wr, color='k',linewidth=3.)
    ax.quiver(1.002*x0, 1.002*y0, 1.002*z0,  ur, vr, wr, color='w',linewidth=1.)#,headlength=8., headwidth=5.)
            
    ax.view_init(azim, degshift)
    ax.dist = 7
    return
    
   


def read_star(filename):
    """
        Read in a stellar model (i.e. a GYRE input file from MESA)
        The data that are read in, are the radial coordinates and the N^2-profile.
    """
    data = np.loadtxt(filename, skiprows=1)
    rad = data[:,1] * u.cm
    brunt2 = data[:,8] * (u.rad / u.s)**2.
    
    return rad, brunt2


def calculate_first_moment(vel,line):
    """
        Calculate the variation of the 1st-moment over the pulsational phase cycle
    """
    
    mom1 = np.nansum(vel*(1.-line)) / np.nansum(1.-line)
    return mom1


def calculate_amplitudes_phases(puls_phases,vel,line_profiles):
    """
        Calculate the amplitude and phase variability over the stellar line profile
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
    
    incl = 80.*np.pi/180.	# inclination angle in rad (between 0 and pi/2)
    frot_u = 1.535/ u.day   	# rotation frequency of the star, with astropy units 1/u.day
    kval = -2			# mode identification (n,k,m)
    mval = -1
    nval = 100
    
    teff_var = 10.   		# Scaling of the temperature variations caused by the pulsation; in Kelvin; wrt Teff
    vel_var = 10.     		# in km/s
    xir_xih_ratio = -1  	# Scaling the ratio between the vertical and horizontal component of the Lagrangian displacement; -1 = use the theroetical ratio
    
    starmodel_file = '/lhome/timothyv/PeterDeCat/HD112429/models/M1p5_Z0014_X50_fov15_Dmix1.GYRE'
    gyre_dir = '/lhome/timothyv/Bin/mesa/mesa-11701/gyre/gyre/'
    saveplot=True
    
    rad,brunt2 = read_star(starmodel_file)
    star = stellar_model(rad,brunt2)
    asym = asymptotic(gyre_dir,kval=kval,mval=mval,nmin=1,nmax=100)
    
    fpuls, omval, lmbd, base_vel, base_line = calculate_line(star, incl, frot_u)
    
    puls_phases = np.linspace(0.,1.,81)[:-1]
    mom1 = []
    all_lines = []
    
    fig = plt.figure(1,figsize=(6,7))
    plt.subplots_adjust(left=0.16,right=0.95,top=0.95,hspace=0.1)
    ax1 = fig.add_subplot(311)
    plt.xticks([-150.,-100.,-50.,0.,50.,100.,150.],['','','','','','',''])
    plt.xlim(-160.,160.)
    plt.ylim(0.95,1.002)

    for iph,puls_phase in enumerate(puls_phases):
        fpuls, omval, lmbd, obs_vel, obs_line = calculate_line(star, incl, frot_u, asym=asym, puls_phase=puls_phase,nval=nval,teff_var=teff_var,vel_var=vel_var,xir_xih_ratio=xir_xih_ratio, saveplot=saveplot)
        all_lines.append(obs_line-base_line)
        mom1.append(calculate_first_moment(obs_vel, obs_line))
        ax1.plot(obs_vel, obs_line)
    
    all_lines = np.array(all_lines)
    mom1 = np.array(mom1)
    amp, phase = calculate_amplitudes_phases(puls_phases,base_vel,all_lines)
    sel_line = np.r_[phase != 0.]
    

    ax2 = fig.add_subplot(312)
    plt.xticks([-150.,-100.,-50.,0.,50.,100.,150.],['','','','','','',''])
    plt.xlim(-160.,160.)
    plt.ylim(0.,0.0065)
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
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.003))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.0006))
    
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(50.))
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(10.))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
    if(saveplot):
        figname = f'n{nval}k{int(asym.kval)}m{int(asym.mval)}_frot{round(frot_u.value*1000)}_FPF'
        plt.savefig(f'./LPV_snapshots/{figname}.png')
    
    fig2 = plt.figure(2,figsize=(6,3))
    plt.subplots_adjust(left=0.16,right=0.95,top=0.95,hspace=0.1,bottom=0.2)
    ax4 = fig2.add_subplot(111)
    comb_phases = np.array(list(puls_phases-1.)+list(puls_phases)+list(puls_phases+1.))
    comb_mom1 = np.array(list(mom1)*3)
    zp = np.interp(0.,comb_mom1[np.r_[(comb_phases > 0.7) & (comb_phases < 1.1)]], comb_phases[np.r_[(comb_phases > 0.7) & (comb_phases < 1.1)]]) - 1
    ax4.plot(comb_phases-zp,comb_mom1,'k-')
    plt.ylim(-3.,3.)
    plt.xlim(0.,1.)
    ax4.set_ylabel(r'$\sf 1^{st}$ moment ($\sf km\,s^{-1}$)',labelpad=10)
    ax4.set_xlabel('pulsation phase')
    
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax4.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(3.))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.6))
    
    if(saveplot):
        figname = f'n{nval}k{int(asym.kval)}m{int(asym.mval)}_frot{round(frot_u.value*1000)}_moment1'
        plt.savefig(f'./LPV_snapshots/{figname}.png')
    else:
        plt.show()
    













