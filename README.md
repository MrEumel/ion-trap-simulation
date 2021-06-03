# Simulation of ions in a linear ion trap

This program may be used to simulate and visualize ion trajectories inside a linear ion trap with many different operation modes and analytic features. In contrast to commercial ion trajectory simulation software, this program solves the equations of motion using a very simple integration method (the Euler-method), allowing one to simulate the behaviour of complex ion clouds consisting of up to thousands of ions with relatively little computing time.

## Getting started

# Setup

Open the "Simulation.py" file using your preferred Python IDE. Before you run the program, please first make sure that you have the required packages installed.

The file is separated into two main functions:

1. Simulation: The equations of motion are solved for all ions, their trajectories as well as the total energy of the system may be plotted, and results are saved in a text file.
2. Evaluation: This reads the text file containing the relevant simulation results and offers several commonly used phase diagram plots, which allows the evaluation and comparison of different parameters' impact on the stability of the ion cloud.

These two functions can be turned on or off individually in lines X and Y. The evaluation function checks for an existing results text file for the combination of parameters specified in the settings. As such, evaluation will fail when the simulation function is skipped **and** no relevant results from previous simulations exist. 

The two main functions shall be discussed in more detail in the following sections and many of the most commonly used operating modes illustrated with example results.

# Folder structure

The results of any given simulation and/or evaluation are automatically stored in a file path that specifies the operating mode and parameters used. If the file path already exists, new results will be added and existing simulations with identical settings will be overwritten. The text file "results_complete.txt" will never be overwritten and new simulation results are automatically appended to this file. 

## The Simulation

# Settings

At the beginning of the code, the sections titled "Simulation settings" offers various options to specify the geometric properties of the ion trap, operating modes and their parameters. Settings may be activated by setting to "1" and deactivated by setting to "0". Please note that activating more than one RF amplitude mode at a time may cause unreasonable results, same applies to activating more than one beta (friction coefficient) mode. 

A selection of core parameters takes arrays as input type, which allows one to specify multiple values as input. In this case, the program will iterate through and simulate all possible combinations of given values. For each combination of values, selected plots of the individual simulation are saved in a separate folder, while the key properties of all simulations performed are saved as individual columns in one results text file named "results.txt". In addition to that, a "results_complete.txt" file is created, which contains **all** simulation results with the chosen combination of settings. New simulation results are always automatically appended as columns in the "results_complete.txt" file. 

# Output

Running the simulation will always yield a "results.txt" file, which is saved according to the folder structure mentioned above, and these results are always added to the "results_complete.txt" contained in the same folder. Each individual simulation is saved as one column in the text file, where values are stored in rows as follows:

0 = Index
1 = U (amplitude of RF field)
2 = N (total number of ions)
3 = f (frequency of RF field)
4 = q-value
5 = U_eff (effective potential of RF field)
6 = delta_E (energy resolution)
7 = ions_escaped
8 = beta (friction coefficient used to simulate collision cooling effect)
9 = n (ion amount in beam)
10 = ion percent in beam
11 = pulse duration (In case of U_rect: only valid for cycleList with one dipole pulse (0, ..., 0, +1, -1, 0, ..., 0))
12 = pause duration
13 = ion cloud max radius
14 = duty cycle of pulse mode (defined as duty_cycle = (pause_duration - 1e-6) / (pause_duration + pulse_duration))

In addition to that, many more output options are available and can be activated through the simulation settings. This includes plot options, such as ion trajectories within the trap geometry or the kinetic, potential and total energy of the system. The exact position coordinates of all ions may also be saved as a text file with x/y/z coordinates saved to separate text files, however, please note that these text files can easily become very large for many ions and may take a long time to save.

# Examples

* 2D/3D ion clouds (quad/oct?) (stable/unstable)

* 1D oscillation gedämpft/ungedämpft

* RF heating / energy



## The Evaluation

# Settings

The section titled "Evaluation settings" offers a choice of phase diagrams and plots to evaluate all results that match the operating settings and parameters specified when running the code, with further options to filter the results plotted by various parameters. Since often times there are three parameters that influence the behaviour of the ion cloud, this filter function is useful and necessary to accurately view the stability of each configuration. 

It is possible to choose between evaluating only the results of the most recently performed simulation ("results.txt") or evaluating the results of all previous simulations with the chosen operating modes and parameters ("results_complete.txt"). It is also possible to activate both at the same time.

# Output

All output files generated will be saved automatically within the aforementioned folder structure. The evaluation results are stored in a separate folder called "phase diagrams". Please note that diagrams are overwritten each time you re-run the evaluation and select different filters.

# Examples

* N-q / N-dutycycle stability phase diagrams (comparing different modes)
* cos/rechteck/puls -> duty cycle

## Work in progress: Electron simulation

It is possible to use the existing simulation code to simulate electron trajectories instead of ion trajectories by changing the parameters accordingly. To do this, one needs to change the particle mass to 0.00054858 amu and the friction coefficient beta is set to zero. Random starting velocities for the particles should also be activated. Most importantly, the number of integration steps per period needs to be significantly higher than it is for ions. 

The practical purpose of such a simulation is to determine reasonable parameters that allow for the extraction of electrons created within the trap through the use of a divergent magnetic field. This divergent magnetic field can be activated and modelled as needed within the program settings.

This is currently still a work in progress and reasonable working parameters still need to be explored.
