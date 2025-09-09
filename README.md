# Quasiperiodic-OFL

A collection of Python scripts for studying quasiperiodic optical flux lattices.

## Physical Background
Quasicrystals have spatial order, but are not periodic. Quasiperiodic systems with non-trivial topology have been theoretically predicted to host various exotic phenomena, but experimental research is lacking. In this work, we propose a scheme whereby a topological quasicrystalline system may be implemented in a cold-atom experiment, through the means of a so-called `quasiperiodic optical flux lattice' (OFL). These files contain Python code for analysing the model Hamiltonian generated in our scheme, to show that it is both quasiperiodic and topological.

## Contents
### Approximant
This folder contains code used to generate and analyse periodic approximants to the quasiperiodic system.
- Approximant_Bandstructure.py: code for diagonalising the approximant Hamiltonians, calculating the density of states (DoS), etc.
- Approximant_Curvature.py: code for calculating the Berry curvature and Chern number of approximant systems.
- Approximant_Vectors.py: code for generating and visualising the approximant systems.
- Generate_Combined_Data.py: a single script which can be run to fully analyse a single approximant system with specified parameters. The bandstructure, density of states, Berry curvature and Chern number are all calculated and saved.
- Plot_Approximant_Bandstructure.py: code for plotting bandstructures, DoS, etc.
- Plot_Approximant_Curvature.py: code for plotting Berry curvature.
- Plot_Phase_Diagram.py: code for plotting the topological phase diagram of the approximant system (i.e., the Chern number over parameter space).
- Trends.py: code for generating and plotting data of the trends in system properties for increasingly accurate approximants.

### Updated Geometry
This folder contains code used in analysing the quasiperiodic system directly. (The name 'updated' is with reference to a previous model, which was discarded as it did not become topological anywhere in parameter space).
- Calc_Bandstructure.py: code for calculating the system bandstructure, DoS, etc., using a basis of plane-wave states.
- Calc_Curvature.py: code for calculating the Berry curvature and Chern number.
- Generate_Data.py: script for performing a full analysis of a quasiperiodic system with specified parameters, performed at a specified order of expansion in the plane-wave basis.
- Plot_Bandstructure.py: code for plotting bandstructures, DoS, etc.
- Plot_Curvature.py: code for plotting Berry curvature.
- Plot_N_bands.py: code for plotting the number of bands below the topological gap in the system, against order of plane-wave basis set.
- QBZ_Bandstructure.py: code for calculating and plotting the bandstructure and Berry curvature over the 'quasi-Brillouin zone' (QBZ) (valid in the weak-coupling limit) using a minimal basis of plane-waves required at first-order in the coupling strength.
- Symmetries.py: code for investigating the symmetry properties of the system in the weak-coupling limit. Representations of different symmetry operators are constructed in the minimal basis of states needed to span the QBZ, and their commutation relations with each other and different terms in the Hamiltonian are calculated.
- B_eff.py: code for calculating, plotting and analysing the effective magnetic field B_eff for a particle adiabatically following the low-energy spinor of the system.