# multiple_laser_filamentation
Electrodynamics project: Simulations of multiple laser filamentation for the PHY_2S004_EP electodynamics course.

The structure of the data is the following:

    - Rares: the main simulation codes for all the numerical
    methods and the bash scripts for running concurrent sims
    on distributed computers through SSH
    
    - Eren: the addition of the plasma term in the simulations
    
    - Ilja: the addition of the MPA term in the simulations
    
    - Teofil: the plotting functions based on the simulation
    results (which were stored locally), a few simulation runs
    and input beam profile codes
    
    - Results: all the plots that we produced, based on the
    search tree in the project report. For each simulation,
    dynamic contains the plots on dynamic scales, static on a
    static scale and a_manual contains manual plotting of graphs
    for comparisons to be made.
