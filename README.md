# LPV-simulations

This code allows you to build a toy model of spectral line profile variations (LPV) caused by (high-order) gravity-mode pulsations, as explained in the Appendix of Van Reeth et al. (2022, submitted). We refer the interested reader to this publication for a detailed mathematical description of this model. It relies on the Traditional Approximation of Rotation (TAR) module of GYRE, and uses the Hough function calculations presented by dr. Vincent Prat in Prat et al. (2019).
    
Python packages required to run the code, are:
- astropy
- matplotlib
- numpy

Other required software packages are:
- the stellar evolution code MESA (https://docs.mesastar.org/)
- the stellar pulsation code GYRE (https://gyre.readthedocs.io/)

If the LPV-simulations software package is used, please reference:
- Prat et al. (2019) (https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..64P/abstract)
- Van Reeth et al. (submitted)
- We refer the user to the websites of MESA, GYRE and the various python packages for the most up-to-date acknowledgement requirements for these codes.
    

To use the LPV-simulations code, adapt the contents of the inlist 'LPV_inlist.dat' as needed, and execute the command:

    $ python Theoretical_LPV.py
    
    
For questions, contact: timothy.vanreeth at kuleuven.be
