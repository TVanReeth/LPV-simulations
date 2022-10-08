# LPV-simulations

This code allows you to build a toy model of spectral line profile variations (LPV) caused by (high-order) gravity-mode pulsations, as explained in the Appendix of Van Reeth et al. (2022, A&A, accepted). We refer the interested reader to this publication for a detailed mathematical description of this model. It relies on the Traditional Approximation of Rotation (TAR) module of GYRE, and uses the Hough function calculations presented by dr. Vincent Prat in Prat et al. (2019).
    
Python packages required to run the code, are:
- astropy
- matplotlib
- numpy

Other required software packages are:
- the stellar evolution code MESA (https://docs.mesastar.org/; for the calculation of the input models)
- the stellar pulsation code GYRE (https://gyre.readthedocs.io/; both for the calculation of the input models and to run the LPV-simulations)

If the LPV-simulations software package is used, please reference:
- Prat et al. (2019) (https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..64P/abstract)
- Van Reeth et al. (2022) (https://ui.adsabs.harvard.edu/abs/2022A%26A...662A..58V/abstract)
- We refer the user to the websites of MESA, GYRE and the various python packages for the most up-to-date acknowledgement requirements for these codes.
    

To use the LPV-simulations code, adapt the contents of the inlist 'LPV_inlist.dat' as needed, and execute the command:

    $ python Theoretical_LPV.py
    
    
For questions and feedback, please contact: timothy.vanreeth at kuleuven.be
