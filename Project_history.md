# Project history

## Up until 12 April (what we did and what we still need to do)
*Already done*:
- implemented python code for the laser filamentation, including all the terms:
    - transverse diffraction
    - GVD
    - absorption and defocusing due to electron plasma
    - MPI (multi-photon absorption)
    - nonlinear SF (self focusing)
- opened github project
- ran code on school computers

*To do*:
- meet with the professor to discuss future directions
- abstractize the code (for ease of use and efficiency)
- run A BUNCH of sims, with different parameters, to see how it reacts to different conditions
- find a way to run the sims on more powerful computers (ask professor about access to computer clusters)
- look into possible applications and uses of this effect (the PRL of Rares, acceleration of electron beams in cavities, PWA paper to read most important)

## 14 April 2026, Meeting with professor

*Ideas from professor*:
- increase resolution in r, resolution in z is more than the necessary
- run diagnostics on small scales, to check validity
- are we actually doing multiple laser filamentation? For now we have just used the cylindrical symmetry. Change to cartesian
- to ease the memory usage, try to save periodically the data into a file (once every few steps). Then use the information from there
- Would it be possible to change the resolution in r once filamentation occurs?
- Can we combine Finite difference method with Spectral method? Use the Fourier transform in terms of $z$ with no time term.

* TO DO *
- Problems that need to be solved rigth now: what we have at the moment is a grid $z\times r\times t$ of the form 100_002 x 102 x 102. In this form, because of the small resolution along $r$, once the beam gets to the non-linear regime, it starts oscillating too much. To solve this, we need to increase the number of points in the radial vector (say by at least a factor of 10). To do this, we need to store the envelope of the electric field for diagnostics in a more memory-efficient way. 
