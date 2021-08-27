import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import astropy.units as u

from amigo_simple.gmode_series import asymptotic
from amigo_simple.stellar_model import stellar_model
from Hough_VPrat import hough_functions

def spherical_to_cartesian(theta,phi,rad=None):
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
    rad = np.sqrt(x**2. + y**2. + z**2.)
    theta = np.arccos(z/rad)
    
    phi = np.arcsin(x/(rad*np.sin(theta)))
    phi_nansel = ~np.isfinite(phi)
    phi[phi_nansel] = 0.
    
    phi2 = np.arccos(y/(rad*np.sin(theta)))
    phi_nansel = ~np.isfinite(phi2)
    phi2[phi_nansel] = 0.
    
    phi_replace = np.r_[(phi2 > np.pi/2.) & (phi > 0.)]
    phi[phi_replace] = phi2[phi_replace]
    
    phi_replace = np.r_[(phi2 > np.pi/2.) & (phi < 0.)]
    phi[phi_replace] = 2.*np.pi - phi2[phi_replace]
    
    return rad,theta,phi
    


def rotate_cart(x,y,z,incl):
    
    x_rot = np.cos(incl)*x - np.sin(incl)*z
    y_rot = y
    z_rot = np.sin(incl)*x + np.cos(incl)*z
    
    return x_rot, y_rot, z_rot


def rotate_spher(theta, phi, incl, rad=None):
    x,y,z = spherical_to_cartesian(theta,phi,rad=rad)
    x_rot, y_rot, z_rot = rotate_cart(x,y,z,incl)
    rad_rot, theta_rot, phi_rot = cartesian_to_spher(x_rot,y_rot,z_rot)
    
    if(rad is None):
        return theta_rot, phi_rot
    else:
        return rad_rot, theta_rot, phi_rot
    

def base_lineprofile(depth=0.15):
    rv_base = np.linspace(-20.,20.,401)
    prof_base = depth*np.exp(-0.5*(rv_base/4.)**2.)
    prof_base[0] = 0.
    prof_base[-1] = 0.
    
    return rv_base, prof_base


def rotational_velocityfield(rad, theta, phi, f_rot):
    radv = np.zeros(rad.shape)
    thv =  np.zeros(rad.shape)
    phv = 2. * np.pi * rad * np.sin(theta) * f_rot
    
    return radv,thv,phv


def project_velocityfield(radv,thv,phv,theta_rot,theta,phi,incl):
    proj_v_rad = radv * np.cos(theta_rot)
    proj_v_th = thv * np.sin(theta_rot) * np.cos(phi)
    proj_v_ph = phv * np.sin(incl) * np.sin(phi)
    
    return proj_v_rad+proj_v_th+proj_v_ph


def limb_darkening(theta_rot, mu=0.6):
    ld = 1. - mu * (1. - np.cos(theta_rot))
    return ld


def puls_intensity(star,theta, phi, xi_r=0.,xi_t=0.,xi_p=0.,Tr=0.,Teff=7000.):
    bi = np.ones(theta.shape)
    bi =  ((star.radius.to(u.km)[-1].value + xi_r)*bi/star.radius.to(u.km)[-1].value)**2. * ((Teff+Tr)/Teff)**4.
    sel = ~np.isfinite(bi)
    bi[sel] = 1.
    
    return bi



def calculate_line(star, incl, frot_u, asym=None, azim_view=0.,puls_phase=0.,nval=50, teff_var=10.,vel_var=5.,xir_xih_ratio=0.001):
    Rstarkm = star.radius.to(u.km).value[-1]
    f_rot = frot_u.value / 86400.
    
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
    
    radv,thv,phv = rotational_velocityfield(rad, theta, phi, f_rot)
    # TODO: add the pulsational velocity fields here...
    if(asym is None):
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
    else:
        fpuls, omval, lmbd, xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr = calculate_pulsations(star, frot_u, theta, phi, teff_var, vel_var,xir_xih_ratio,asym, nval, puls_phase)
    
    rad_rot, theta_rot, phi_rot = rotate_spher(theta, phi, incl, rad=rad)
    proj_vel = project_velocityfield(radv + u_r,thv + u_t,phv + u_p,theta_rot,theta,phi,incl)
    
    obs_vel = np.linspace(-300.,300.,6001)
    obs_line = np.zeros(obs_vel.shape)
    base_vel,base_line = base_lineprofile()
    ld = limb_darkening(theta_rot)
    bi = puls_intensity(star,theta,phi,xi_r=xi_r,xi_t=xi_t,xi_p=xi_p,Tr=Tr)
    
    for ith in np.arange(len(theta_rot[:,0])):
        for iph in np.arange(len(theta_rot[0,:])):
            if(theta_rot[ith,iph] < np.pi/2.):
                obs_line += np.interp(obs_vel,proj_vel[ith,iph] + base_vel, 2.*np.cos(theta_rot[ith, iph])*cell_weight[ith, iph]*ld[ith,iph]*bi[ith, iph] * base_line)
    obs_line = 1. - obs_line
    
    return fpuls, omval, lmbd, obs_vel, obs_line
    


def calculate_puls_geometry(asym,nval,frot,Pi0,theta,phi):
    
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
    
    if(asym.mval < 0.):
        sign = -1.
    else:
        sign = 1.
    # calculate Pi0
    Pi0 = star.buoyancy_radius()
    
    # calculate the pulsation mode geometry
    fpuls, omval, lmbd, hr, ht, hp = calculate_puls_geometry(asym,nval,frot,Pi0,theta,phi)
    non_convective = star.brunt.value > 0.
    
    if(asym.kval < 0.):
        Tr = teff_var * (hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase))
        
        xi_r = np.sqrt(lmbd) * teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval) * hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_t = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * ht * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_p = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hp * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        
        u_r = -np.sqrt(lmbd) * teff_var * xir_xih_ratio  / (star.radius.to(u.km)[non_convective][-1].value) * hr * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_t = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * ht * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        u_p = teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * hp * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
    
    
    else:
        Tr = teff_var * (hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)) / np.nanmax(hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase))
        
        xi_r = -np.sqrt(lmbd) * teff_var * xir_xih_ratio / (star.radius.to(u.km)[non_convective][-1].value * omval) * hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_t = teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * ht * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        xi_p = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hp * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        
        u_r = np.sqrt(lmbd) * teff_var * xir_xih_ratio  / (star.radius.to(u.km)[non_convective][-1].value) * hr * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_t = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * ht * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
        u_p = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * hp * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
        
       ## xi_r = -np.sqrt(lmbd) * teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval * star.brunt[non_convective][-1].value) * hr * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
       ## xi_t = teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * ht * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
       ## xi_p = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval**2.) * hp * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
    
       ## u_r = np.sqrt(lmbd) * teff_var / (star.radius.to(u.km)[non_convective][-1].value * star.brunt[non_convective][-1].value) * hr * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
       ## u_t = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * ht * np.sin(mval*phi + 2.*np.pi*sign*puls_phase)
       ## u_p = -teff_var / (star.radius.to(u.km)[non_convective][-1].value * omval) * hp * np.cos(mval*phi + 2.*np.pi*sign*puls_phase)
    
    # rudimentary scaling?
    extra_scaling = np.nanmax(np.sqrt(u_r**2. + u_t**2. + u_p**2.))
    if(extra_scaling > 0.):
        u_r = u_r * vel_var/extra_scaling
        u_t = u_t * vel_var/extra_scaling
        u_p = u_p * vel_var/extra_scaling
        xi_r = xi_r * vel_var/extra_scaling
        xi_t = xi_t * vel_var/extra_scaling
        xi_p = xi_p * vel_var/extra_scaling
    """
    fig2 = plt.figure()
    azim = 20.
    degshift=30.
    alphaval = 1.
    vminval = -1.  # keep for now
    vmaxval = 2.   # keep for now
    
    x,y,z = spherical_to_cartesian(theta,phi)
            
    fcolors_r = Tr
    fmax_r, fmin_r = fcolors_r.max(), fcolors_r.min()
    fcolors_ra = (fcolors_r - fmin_r)/(fmax_r - fmin_r)
            
    fcolors_r = 2.*(fcolors_r - fmin_r)/(fmax_r - fmin_r) - 1.
    fcolors_r = 0.5*(fcolors_r/np.sqrt(np.abs(fcolors_r)) + 1.)
            
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig2.clf()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors_ra),vmin=vminval, vmax=vmaxval)
    # Turn off the axis planes
    ax2.set_axis_off()
            
            
    xi_t *= 0.2
    xi_p *= 0.2        
    
    if 1:
            fcolors_r = xi_t     #sph_harm(m, l, theta, phi).real
            fmax_r, fmin_r = fcolors_r.max(), fcolors_r.min()
            
            fcolors_r2 = xi_p     #sph_harm(m, l, theta, phi).real
            flen = np.sqrt(fcolors_r**2. + fcolors_r2**2.)
            fmax_r, fmin_r = flen.max(), flen.min()
            normr = (flen - fmin_r)/(fmax_r - fmin_r)
        
            fcolors_ra1 = fcolors_r * normr / flen
            fcolors_ra2 = fcolors_r2 * normr / flen
        
        
            ur = 0.1 * ( fcolors_ra1 * np.cos(theta) * np.cos(phi) - fcolors_ra2 * np.sin(phi) )
            vr = 0.1 * ( fcolors_ra1 * np.cos(theta) * np.sin(phi) + fcolors_ra2 * np.cos(phi) )
            wr = 0.1 * ( -fcolors_ra1 * np.sin(theta) )
            
            #ax2.quiver(x, y, z,  1.1*ur, 1.1*vr, 1.1*wr, color='k',linewidth=3.)
            ax2.quiver(1.002*x, 1.002*y, 1.002*z,  ur, vr, wr, color='w',linewidth=1.)#,headlength=8., headwidth=5.)
            
            ax2.view_init(azim, degshift)
    plt.show()
    """
    
    return fpuls, omval, lmbd, xi_r,xi_t,xi_p,u_r,u_t,u_p,Tr



def read_star(filename):
    data = np.loadtxt(filename, skiprows=1)
    rad = data[:,1] * u.cm
    brunt2 = data[:,8] * (u.rad / u.s)**2.
    
    return rad, brunt2


def calculate_first_moment(vel,line):
    mom1 = np.nansum(vel*(1.-line)) / np.nansum(1.-line)
    return mom1


def calculate_amplitudes_phases(puls_phases,vel,line_profiles):
    
    matrix = []
    matrix.append(np.sin(puls_phases))
    matrix.append(np.cos(puls_phases))
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
    
    incl = 0.5*np.pi   #83.*np.pi/180.
    frot_u = 1.535 / u.day
    kval = -2
    mval = -1
    nval = 30
    
    teff_var = 20.   # in Kelvin
    vel_var = 5.     # in km/s
    xir_xih_ratio = 0.01 # ratio between the vertical and horizontal component of the Lagrangian displacement
    
    starmodel_file = '/lhome/timothyv/PeterDeCat/HD112429/M1p5_Z0014_X50_fov15_Dmix1.GYRE'
    gyre_dir = '/lhome/timothyv/Bin/mesa/mesa-11701/gyre/gyre/'
    
    rad,brunt2 = read_star(starmodel_file)
    star = stellar_model(rad,brunt2)
    asym = asymptotic(gyre_dir,kval=kval,mval=mval,nmin=1,nmax=100)
    
    fpuls, omval, lmbd, base_vel, base_line = calculate_line(star, incl, frot_u)
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312,sharex=ax1)
    ax3 = fig.add_subplot(313,sharex=ax1)
    
    ax1.plot(base_vel, base_line, 'k-')
    
    puls_phases = np.linspace(0.,1.,51)[:-1]
    mom1 = []
    all_lines = []
    for iph,puls_phase in enumerate(puls_phases):
        print(iph)
        fpuls, omval, lmbd, obs_vel, obs_line = calculate_line(star, incl, frot_u, asym=asym, puls_phase=puls_phase,nval=nval,teff_var=teff_var,vel_var=vel_var,xir_xih_ratio=xir_xih_ratio)
        all_lines.append(obs_line-base_line)
        mom1.append(calculate_first_moment(obs_vel, obs_line))
        ax1.plot(obs_vel, obs_line)
    
    all_lines = np.array(all_lines)
    amp, phase = calculate_amplitudes_phases(puls_phases,base_vel,all_lines)
    ax2.plot(base_vel,amp,'k-')
    ax3.plot(base_vel,phase,'k-')
    plt.show()
    













