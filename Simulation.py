import numpy as np
import matplotlib.pyplot as plt
import os
import numba
import time
import itertools
import math
import random

begin = time.time() # to show runtime after program is finished
np.seterr('raise')

"""SIMULATION SETTINGS"""

"""Constants"""
r_0 = 0.003 # [m] - radius of the linear ion trap in x-y-plane
r_beam = 0.0005 # [m] - radius of the beam used to free electrons from ions within the trap
e = 1.602 * 10**(-19) # [C]
amu = 550 # [amu] - mass of simulated ions (for electrons: 0.00054858)
m = amu * 1.66054 * 10 ** (-27) # [kg]
epsilon = 8.854187 * 10**(-12) # [As/Vm]
dz = 0.001 # [m] - length of the linear ion trap in z-direction for 3D simulations

"""Select parameter values as arrays and simulate all possible combinations"""
U_0 = np.array([1.8]) # [V] - amplitude of the applied RF trapping field
f_0 = np.array([188534, 133314, 108850,  94267,  84315,  76969]) # [Hz] - frequency of the applied RF trapping field
beta_0 = np.array([110000, 90000, 70000, 50000, 30000, 10000, 0]) # friction coefficient used to approximate collision cooling
ion_amount = np.array([1500]) # number of ions to simulate

"""Steps per period"""
spp = 25 # number of integration steps per RF period (this is ignored for pulse mode)

"""Periods to simulate"""
periods = 100 # number of RF periods to simulate

"""Select integration parameters"""
dimensions = 2 # choose between 2D and 3D

"""Select trapping potential(s)"""
activate_quad = 1 # activates quadrupole trapping field
activate_oct = 0 # activates octopole trapping field with amplitude of U_oct = U_0 * r_0 ** 2
activate_big_coul = 0 # activates imaginary charge in trap center, also allows scaling for Q = n * (number_of_ions * e)

activate_static_B_field = 0 # activates the static magnetic field
activate_divergent_B_field = 0 # activates a divergent magnetic field similar to electric field of a point charge

activate_starting_velocity = 0 # activates random starting velocity for ions
starting_velocity_factor = 5000 # [m/s] - this factor is multiplied by a random number in interval [0,1] for x,y,z component separately

"""Define the magnetic field if needed"""
B_0 = 0.15 # [T] - magnitude of static magnetic field

B_x = 11.5 # [T] - magnitude of divergent magnetic field in x-direction
B_y = 11.5 # [T] - magnitude of divergent magnetic field in y-direction
B_z = 11.5 # [T] - magnitude of divergent magnetic field in z-direction
B_origin = 0.24 # [m] - point of origin of the divergent magnetic field

"""Select RF amplitude mode"""
activate_U_cosine = 1 # amplitude modulation by cosine function

activate_U_amp_mod = 0 # amplitude modulation by factor 1 + cos(OMEGA * t / U_amp_freq) * U_amp_factor
U_amp_freq = 20
U_amp_factor = 0.8

activate_U_rectangle = 0 # amplitude modulation by rectangular function (periods are divided into length of cycleList)
cycleList = [1,-1,0,0,0,0,0,0] # factors to cycle through, multiplies U_0 with the integer in the list

activate_U_pulses = 0 # amplitude modulation by pulses of defined length (specify timestep "dt" and pulse duration below)
dt = 3e-7 # [s] - time per integration step
asym_pulse_steps = 6 # in units of [dt] - yields total duration of amplitude pulse
U_asym_cycle = [3, -3] # asym_pulse_steps divided by cycle length (= number of elements in cycle) needs to be an integer

activate_U_pause = 0 # turn off trapping field at 1/2 duration of first amplitude factor in cycleList for periods selected below
U_pause_start = 10 # period to turn the field off
U_pause_stop = 15 # period to turn the field back on

"""Select beta modes"""
activate_beta_cosine = 0 # beta modulation by cosine function
beta_cosine_freq = 5e5 # beta cosine modulation frequency in [Hz]

activate_beta_factor = 0 # original beta value (beta_0) times beta_factor is subtracted from current beta value every beta switch period
beta_start_switch = 10  # in periods, when to begin beta modulation
beta_end_switch = 30 # in periods, when to stop beta modulation
beta_factor = 0.05 # factor times beta_0 to be subtracted per beta switch
switch_beta = 5 # beta factor times beta_0 is subtracted after this many periods

"""Select plots to save"""
save_default_plots = 1 # save default view of ion cloud in x-y-plane (2D) or x-y-z-volume (3D)
save_energy_plots = 1 # save plots for kinetic, potential and total energy of the system
save_separate_plane_time_plots = 0 # save plots of trajectories over time in x/y/z planes separately
view_periods_start = periods - periods # plots for the last X periods are saved, set X to "periods" to plot all periods
default_view_show = 0 # show plots after integration (plots have to be manually closed before next integration can run)
activate_trajectory_saving = 0 # save the trajectory of each ion over time in a text file

"""EVALUATION SETTINGS"""

"""General"""
skip_simulation = 0 # skip simulation and only evaluate already existing results
evaluate_simulation_results = 0 # evaluate results from current simulation only
evaluate_complete_results = 0 # evaluate results from all existing results with identical simulation mode settings
show_phase_diagrams = 0 # show selected phase diagrams (plots have to be manually closed for program to finish)

"""Select plots to show"""
radius_diagram = 1 # activate to plot ion cloud radius vs. q value and ion cloud radius vs. beta value
n_q_diagram = 1 # activate to plot ion amount in beam area (n) vs. q value
n_beta_diagram = 1 # activate to plot ion amount in beam area (n) vs. beta value
N_q_diagram = 1 # activate to plot total ion amount (N) vs. q stability diagram
beta_q_diagram = 1 # activate to plot beta vs. q stability diagram
beta_duty_cycle_diagram = 1 # activate to plot beta vs. duty_cycle stability diagram
duty_cycle_density_diagram = 1 # activate to plot cloud density vs. duty_cycle stability diagram

"""Filter radius diagrams"""
"""Filter q vs. radius by beta"""
beta_density_filter = 0
beta_density_filter_min = 0
beta_density_filter_max = 110000
"""Filter beta vs. radius by q"""
q_density_filter = 0
q_density_filter_min = 0.1
q_density_filter_max = 0.1

"""Filter n vs. q diagram by beta"""
beta_n_beam_filter = 0
beta_n_beam_filter_min = 110000
beta_n_beam_filter_max = 110000

"""Filter n vs. beta diagram by q"""
q_n_beam_filter = 0
q_n_beam_filter_min = 0.60
q_n_beam_filter_max = 0.60

"""Filter N vs. q diagram"""
"""Filter N vs. q diagram by beta"""
Nq_beta_filter = 0
Nq_beta_filter_min = 1e-25
Nq_beta_filter_max = 1e-15
"""Select only ONE of the two filters at a time - ONLY works when beta filter above is activated"""
Nq_N_filter = 0 # when activated N_q diagram filters for N_filter_min =< N =< N_filter_max
Nq_N_filter_min = 400
Nq_N_filter_max = 400
Nq_q_filter = 0 # when activated N_q diagram filters for q_filter_min =< q =< q_filter_max
Nq_q_filter_max = 0.55
Nq_q_filter_min = 0.45

"""Filter beta vs. q diagram"""
"""Filter beta vs. q diagram by N"""
betaq_N_filter = 0 # when activated beta_q diagram filters for N_filter_min =< N =< N_filter_max
betaq_N_filter_min = 1500
betaq_N_filter_max = 1500
"""Select only ONE of the two filters at a time - ONLY works when N filter above is activated"""
betaq_beta_filter = 0 # when activated beta_q diagram filters for beta_filter_min =< beta =< beta_filter_max
betaq_beta_filter_min = 1e-25
betaq_beta_filter_max = 1e-15
betaq_q_filter = 0 # when activated beta_q diagram filters for q_filter_min =< q =< q_filter_max
betaq_q_filter_max = 0.55
betaq_q_filter_min = 0.45

"""Filter beta vs. duty_cycle diagram"""
"""Filter beta vs. duty_cycle diagram by N"""
betacycle_N_filter = 0 # when activated beta_q diagram filters for N_filter_min =< N =< N_filter_max
betacycle_N_filter_min = 1500
betacycle_N_filter_max = 1500
"""Select only ONE of the two filters at a time - ONLY works when N filter above is activated"""
betacycle_beta_filter = 0 # when activated beta_pulse diagram filters for beta_filter_min =< beta =< beta_filter_max
betacycle_beta_filter_min = 1e-25
betacycle_beta_filter_max = 1e-15
betacycle_pulse_filter = 0 # when activated beta_pulse diagram filters for q_filter_min =< q =< q_filter_max
betacycle_cycle_filter_max = 0.55
betacycle_cycle_filter_min = 0.45

"""Filter duty_cycle vs. density diagram"""
"""Filter duty_cycle vs. density diagram by N"""
cycle_density_N_filter = 0
cycle_density_N_filter_min = 1
cycle_density_N_filter_max = 1500
"""Filter duty_cycle vs. density diagram by beta"""
cycle_density_beta_filter = 0
cycle_density_beta_filter_min = 0
cycle_density_beta_filter_max = 110000

"""Legend for results text files - meaning of each row:

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
"""

if activate_U_rectangle == 1:
    spp_check = spp/len(cycleList)
    if spp_check.is_integer() == False:
        raise ValueError('For U_rectangle spp divided by length of cycleList has to be an integer')

if activate_U_pulses == 1:
    spp_check = asym_pulse_steps/len(U_asym_cycle)
    if spp_check.is_integer() == False:
        raise ValueError('Pulse steps divided by cycle list has to be an integer.')

if skip_simulation == 0:

    """Initiate result arrays for all possible parameter combinations to simulate"""
    index = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    U_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    amount_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    f_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    q_factor_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    V_eff_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    delta_E_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    ions_escaped_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    beta_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    ions_centered_res = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    percent_in_laser = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    dipole_pulse_duration = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    dipole_pause_duration = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    ion_cloud_radius = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    duty_cycle = np.zeros(len(U_0) * len(ion_amount) * len(f_0) * len(beta_0))
    counter = -1 # this provides an index integer for each simulation

    """Select a combination of parameters from arrays"""
    for voltage in range(len(U_0)):
        for amount in range(len(ion_amount)):
            for frequ in range(len(f_0)):
                for fric in range(len(beta_0)):
                    counter += 1
                    index[counter] = counter

                    """Load initial parameters"""
                    print("Load initial parameters")
                    number_of_ions = ion_amount[amount]
                    f = f_0[frequ]
                    OMEGA = f * 2 * np.pi
                    q_factor = (4 * e * U_0[voltage]) / (m * (r_0 ** 2) * (OMEGA ** 2))
                    V_eff = q_factor * U_0[voltage] / 4
                    delta_E = 2 * U_0[voltage] * ((r_beam / r_0) ** 2)
                    Pot_Const = activate_quad * 2 * (e / m) * (1 / (r_0 ** 2))
                    Oct_Const = activate_oct * 2 * (e / m) * (1 / (r_0 ** 4))
                    Coul_Const = (e ** 2) / (4 * np.pi * epsilon * m)
                    Big_Coul = activate_big_coul * e * e * number_of_ions / (4 * np.pi * epsilon * m)

                    """Integration parameters"""
                    T = 1 / f # period duration

                    if activate_U_pulses == 1: # for pulse function, dt is an input parameter and spp calculated accordingly
                        asym_pause_steps = int(np.round((T / dt))) - asym_pulse_steps
                        spp = asym_pulse_steps + asym_pause_steps
                        pulse_duration = asym_pulse_steps * dt
                        pause_duration = asym_pause_steps * dt
                    else:
                        dt = T / spp # for any other mode, spp is an input parameter and dt calculated accordingly

                    steps = spp * periods # total number of integration steps

                    if activate_U_rectangle == 1:
                        U_cycle_duration = np.round((T / len(cycleList)) / dt)  # U_rectangle cycle duration coupled to frequency
                        pulse_duration = (len(cycleList) - 2) * U_cycle_duration * dt

                    """Set up beta and U arrays and check beta value"""
                    beta = np.zeros(steps + 2)
                    U = np.zeros(steps + 2)
                    U_oct = np.zeros(steps + 2)
                    for i in range(steps + 2):
                        beta[i] = beta_0[fric]
                        U[i] = U_0[voltage]
                        if activate_oct == 1:
                            U_oct[i] = U_0[voltage] * r_0 ** 2

                    if beta[0] * dt > 1:
                        raise ValueError('Beta is too large for this time step.')

                    if activate_U_rectangle == 1:
                        U_rect_cycle = np.repeat(cycleList, U_cycle_duration)
                        U_rect_array = np.array([])
                        for i in range(periods + 1):
                            U_rect_array = np.append(U_rect_array, U_rect_cycle)
                        U = np.multiply(U_rect_array[:steps + 2], U)
                        if activate_oct == 1:
                            U_oct = np.multiply(U_rect_array[:steps + 2], U_oct)

                    if activate_U_pulses == 1:
                        U_pulse_array = np.array([])
                        U_correction = np.array([1])
                        U_pulse_array = np.append(U_pulse_array, U_correction)
                        U_pulse_period = np.zeros(asym_pulse_steps + asym_pause_steps)
                        steps_per_pulse = asym_pulse_steps / len(U_asym_cycle)
                        U_pulse_cycle = np.repeat(U_asym_cycle, steps_per_pulse)

                        for i in range(len(U_pulse_cycle)):
                            U_pulse_period[i] = U_pulse_cycle[i]
                        for i in range(periods+1):
                            U_pulse_array = np.append(U_pulse_array, U_pulse_period)
                        U = np.multiply(U_pulse_array[:steps + 2], U)

                    if activate_U_pause == 1:
                        U_pause_start1 = spp * U_pause_start + int(np.round(U_cycle_duration / 2))
                        U_pause_stop1 = spp * U_pause_stop + int(np.round(U_cycle_duration / 2))
                        for i in range(U_pause_start1, U_pause_stop1):
                            U[i] = 0

                    """Save outputs to results array"""
                    U_res[counter] = U_0[voltage]
                    amount_res[counter] = number_of_ions
                    f_res[counter] = f
                    beta_res[counter] = beta[0]
                    V_eff_res[counter] = V_eff
                    delta_E_res[counter] = delta_E
                    q_factor_res[counter] = q_factor
                    if activate_U_pulses == 1:
                        dipole_pulse_duration[counter] = pulse_duration
                        dipole_pause_duration[counter] = pause_duration
                        duty_cycle[counter] = (pause_duration - 1e-6) / (pause_duration + pulse_duration)
                    if activate_U_rectangle == 1:
                        dipole_pulse_duration[counter] = pulse_duration

                    """Set up ion system"""
                    t = np.zeros(steps + 2)
                    x = np.zeros((number_of_ions, steps + 2))
                    y = np.zeros((number_of_ions, steps + 2))
                    z = np.zeros((number_of_ions, steps + 2))
                    vx = np.zeros((number_of_ions, steps + 2))
                    vy = np.zeros((number_of_ions, steps + 2))
                    vz = np.zeros((number_of_ions, steps + 2))
                    ax = np.zeros((number_of_ions, steps + 2))
                    ay = np.zeros((number_of_ions, steps + 2))
                    az = np.zeros((number_of_ions, steps + 2))
                    E_kin = np.zeros(steps + 2)
                    E_pot = np.zeros(steps + 2)

                    """Create results directories"""
                    script_dir = os.path.dirname(__file__)
                    results_dir = os.path.join(script_dir,
                                               str(dimensions) + 'D/' + 'amu' + str(amu) + '/' + 'Quad' + str(activate_quad) + '_Okt' + str(
                                                   activate_oct) + '_statB' + str(activate_static_B_field) + '_divB' + str(activate_divergent_B_field) + '/')
                    if activate_U_rectangle == 1:
                        results_dir = os.path.join(results_dir,
                                                   'Urect' + str(activate_U_rectangle) + 'Ucyc' + str(cycleList) + '/')
                        if activate_U_pause == 1:
                            results_dir = os.path.join(results_dir, 'U_pause_start_period' + str(
                                U_pause_start) + '_U_pause_stop_period' + str(U_pause_stop) + '/')
                    if activate_U_pulses == 1:
                        results_dir = os.path.join(results_dir, 'Uasym_pulse_steps' + str(asym_pulse_steps) + '_Ucyc' + str(
                            U_asym_cycle) + '_dt' + str(dt) + '/')
                    if activate_U_cosine == 1:
                        results_dir = os.path.join(results_dir, 'Ucos' + str(activate_U_cosine) + '/')
                        if activate_U_amp_mod == 1:
                            results_dir = os.path.join(results_dir,
                                                       'Uampmod' + str(activate_U_amp_mod) + '_Uampfreq' + str(
                                                           U_amp_freq) + '_Uampfac' + str(U_amp_factor) + '/')
                    if activate_beta_cosine == 1:
                        results_dir = os.path.join(results_dir,
                                                   'betacos' + str(activate_beta_cosine) + '_betafreq' + str(
                                                       beta_cosine_freq) + '_betastart' + str(
                                                       beta_start_switch) + '_betastop' + str(beta_end_switch) + '/')
                    if activate_beta_factor == 1:
                        results_dir = os.path.join(results_dir, 'betafac' + str(beta_factor) + '_betaswitch' + str(
                            switch_beta) + '_betastart' + str(beta_start_switch) + '_betastop' + str(
                            beta_end_switch) + '/')
                    if activate_U_pulses == 0:
                        results_dir = os.path.join(results_dir, 'Periods' + str(periods) + '_SPP' + str(spp) + '/')
                    else:
                        results_dir = os.path.join(results_dir, 'Periods' + str(periods) + '/')
                    results_dir = os.path.join(results_dir, 'U' + str(U_0[0]) + '/N' + str(number_of_ions) + '/f' + str(
                        f) + '/beta' + str(beta_0[fric]) + '/')
                    if not os.path.isdir(results_dir):
                        os.makedirs(results_dir)

                    gen_dir = os.path.join(script_dir,
                                           str(dimensions) + 'D/' + 'amu' + str(amu) + '/' + 'Quad' + str(activate_quad) + '_Okt' + str(
                                               activate_oct) + '_statB' + str(activate_static_B_field) + '_divB' + str(activate_divergent_B_field) + '/')
                    if activate_U_rectangle == 1:
                        gen_dir = os.path.join(gen_dir,
                                               'Urect' + str(activate_U_rectangle) + 'Ucyc' + str(cycleList) + '/')
                        if activate_U_pause == 1:
                            gen_dir = os.path.join(gen_dir, 'U_pause_start_period' + str(
                                U_pause_start) + '_U_pause_stop_period' + str(U_pause_stop) + '/')
                    if activate_U_pulses == 1:
                        gen_dir = os.path.join(gen_dir, 'Uasym_pulse_steps' + str(asym_pulse_steps) + '_Ucyc' + str(
                            U_asym_cycle) + '_dt' + str(dt) + '/')
                    if activate_U_cosine == 1:
                        gen_dir = os.path.join(gen_dir, 'Ucos' + str(activate_U_cosine) + '/')
                        if activate_U_amp_mod == 1:
                            gen_dir = os.path.join(gen_dir, 'Uampmod' + str(activate_U_amp_mod) + '_Uampfreq' + str(
                                U_amp_freq) + '_Uampfac' + str(U_amp_factor) + '/')
                    if activate_beta_cosine == 1:
                        gen_dir = os.path.join(gen_dir, 'betacos' + str(activate_beta_cosine) + '_betafreq' + str(
                            beta_cosine_freq) + '_betastart' + str(beta_start_switch) + '_betastop' + str(
                            beta_end_switch) + '/')
                    if activate_beta_factor == 1:
                        gen_dir = os.path.join(gen_dir, 'betafac' + str(beta_factor) + '_betaswitch' + str(
                            switch_beta) + '_betastart' + str(beta_start_switch) + '_betastop' + str(
                            beta_end_switch) + '/')
                    if activate_U_pulses == 0:
                        gen_dir = os.path.join(gen_dir, 'Periods' + str(periods) + '_SPP' + str(spp) + '/')
                    else:
                        gen_dir = os.path.join(gen_dir, 'Periods' + str(periods) + '/')

                    """Check existing similar results for expected cloud radius"""
                    r_cloud_known = 0
                    if os.path.isfile(gen_dir + 'results_complete.txt') == True:
                        check_results = np.loadtxt(gen_dir + 'results_complete.txt')
                        check_results = check_results[:, check_results[7] <= 0]
                        check_results = check_results[:, check_results[2] <= number_of_ions+10]
                        check_results = check_results[:, check_results[2] >= number_of_ions-10]
                        check_results = check_results[:, check_results[4] <= q_factor + 0.01]
                        check_results = check_results[:, check_results[4] >= q_factor - 0.01]
                        if len(check_results[0, :]) == 1:
                            r_cloud_known = 1
                            r_cloud = check_results[13]
                        if len(check_results[0, :]) > 1:
                            r_cloud_known = 1
                            r_cloud = np.mean(check_results[13])

                    """Generate random equidistant starting positions"""
                    print("Generate starting positions")
                    # source: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
                    n = ion_amount[amount]
                    # Default setting is typically seed=0
                    # But seed = 0.5 is generally better.
                    seed = 0.5
                    g = 1.32471795724474602596090885447809
                    alpha = np.zeros(dimensions)

                    for j in range(dimensions):
                        alpha[j] = pow(1 / g, j + 1) % 1
                    initial_pos = np.zeros((n, dimensions))

                    for i in range(n):
                        initial_pos[i] = (seed + alpha * (i + 1)) % 1

                    P = np.zeros((n, dimensions)) # yields quasi-random positions in a unit-square

                    if dimensions == 3:
                        dz_half = dz / 2
                        if r_cloud_known == 1:
                            d1 = r_cloud * np.sqrt(2)
                            d2 = float(d1 / 2)
                        else:
                            d1 = np.sqrt(n / (10 ** 13 * dz)) # if no cloud radius can be estimated from existing results, approximate like this
                            d2 = float(d1 / 2)

                        Center = np.array([d2, d2, dz_half]) # shift center of the square to coordinate system origin
                        # scale coordinates in unit-square by actual length of approximate starting square
                        C = np.array([d2 * 2, 0, 0])
                        B = np.array([0, d2 * 2, 0])
                        D = np.array([0, 0, dz_half * 2])

                        #z_fix = np.random.choice(initial_pos[:, 2], size=n, replace=False) # z-values need to be randomly picked (otherwise ions start in a tilted plane)

                        for i in range(0, n):
                            P[i, :] += initial_pos[i, 0] * C
                            P[i, :] += initial_pos[i, 1] * B
                            P[i, :] += initial_pos[i, 2] * D #z_fix[i] * D

                        P_1 = P - Center # yields quasi-random starting coordinates for all ions

                    if dimensions == 2:
                        if r_cloud_known == 1:
                            d1 = r_cloud * np.sqrt(2)
                            d2 = float(d1 / 2)
                        else:
                            d1 = np.sqrt(n) * 2 * 10 ** (-5) # if no cloud radius can be estimated from existing results, approximate like this
                            d2 = float(d1 / 2)

                        Center = np.array([d2, d2]) # shift center of the square to coordinate system origin
                        C = np.array([d2 * 2, 0]) # scale coordinates in unit-square by actual length of approximate starting square
                        B = np.array([0, d2 * 2])

                        for i in range(0, n):
                            P[i, :] += initial_pos[i, 0] * C
                            P[i, :] += initial_pos[i, 1] * B

                        P_1 = P - Center # yields quasi-random starting coordinates for all ions

                    # fill first entry of ion coordinate arrays with the quasi-random coordinate calculated
                    if number_of_ions == 1:
                        x[0, 0] = P_1[0, 0]
                        y[0, 0] = P_1[0, 1]
                        if dimensions == 3:
                            z[0, 0] = P_1[0, 2]
                    else:
                        for i in range(number_of_ions):
                            x[i, 0] = P_1[i, 0]
                            y[i, 0] = P_1[i, 1]
                            if dimensions == 3:
                                z[i, 0] = P_1[i, 2]

                    if activate_starting_velocity == 1:
                        for i in range(number_of_ions):
                            vx[i, 0] = starting_velocity_factor * random.random()
                            vy[i, 0] = starting_velocity_factor * random.random()
                            if dimensions == 3:
                                vz[i, 0] = starting_velocity_factor * random.random()

                    """Integration"""
                    print("Integration")

                    @numba.jit
                    def run(t, x, y, z, vx, vy, vz, ax, ay, az, E_kin, E_pot,
                            spp, U, U_oct, B_0, B_x, B_y, B_z, activate_beta_cosine, activate_beta_factor, beta,
                            beta_start_switch, switch_beta, beta_factor, activate_big_coul, activate_U_cosine, activate_U_amp_mod, beta_end_switch, activate_static_B_field, activate_divergent_B_field):

                        for i in range(0, steps + 1):
                            t[i + 1] = i * dt # calculate total time

                            """Voltage variation code"""
                            if activate_U_cosine == 1:
                                if activate_U_amp_mod == 1:
                                    U[i+1] = U[0] * np.cos(OMEGA * t[i + 1]) * (1 + np.cos((OMEGA/U_amp_freq) * t[i + 1]) * U_amp_factor)
                                    U_oct[i+1] = U[0] * r_0 ** 2 * np.cos(OMEGA * t[i + 1]) * (1 + np.cos((OMEGA/20) * t[i + 1]))
                                else:
                                    U[i + 1] = U[0] * np.cos(OMEGA * t[i + 1])
                                    U_oct[i + 1] = U[0] * r_0 ** 2 * np.cos(OMEGA * t[i + 1])

                            """Beta variation by cosine function"""
                            if activate_beta_cosine == 1:
                                if spp * beta_end_switch > i > spp * beta_start_switch:
                                    beta[i+1] = beta[0] * np.cos(OMEGA * t[i + 1])

                            """Beta variation by beta_factor every period after start switch"""
                            if activate_beta_factor == 1:
                                if spp * beta_end_switch > i > spp * beta_start_switch:
                                    if (i / spp) % switch_beta == 0:
                                            beta[i + 1] = beta[i] - (beta_factor * beta[0])
                                    else:
                                        beta[i + 1] = beta[i]
                                else:
                                    beta[i + 1] = beta[i]

                            """Solving differential equations"""
                            for k in range(0, number_of_ions): # update velocity and position
                                r_k = np.sqrt(x[k, i] ** 2 + y[k, i] ** 2) # checks if ion escaped trap boundary and skips that ion if true
                                if r_k > r_0:
                                    x[k, i + 1] = x[k, i]
                                    y[k, i + 1] = y[k, i]
                                    if dimensions ==3:
                                        z[k, i + 1] = z[k, i]
                                else:
                                    vx[k, i + 1] = vx[k, i] + ax[k, i] * dt
                                    vy[k, i + 1] = vy[k, i] + ay[k, i] * dt
                                    if dimensions == 3:
                                        vz[k, i + 1] = vz[k, i] + az[k, i] * dt

                                    x[k, i + 1] = x[k, i] + vx[k, i + 1] * dt
                                    y[k, i + 1] = y[k, i] + vy[k, i + 1] * dt

                                    if dimensions == 2:
                                        E_kin[i + 1] += ((np.sqrt(vx[k, i + 1] ** 2 + vy[k, i + 1] ** 2)) ** 2) * m / 2

                                    if dimensions == 3:
                                        z[k, i + 1] = z[k, i] + vz[k, i + 1] * dt
                                        E_kin[i + 1] += ((np.sqrt(vx[k, i + 1] ** 2 + vy[k, i + 1] ** 2 + vz[k, i + 1] ** 2)) ** 2) * m / 2
                                        if z[k, i + 1] > dz / 2:
                                            z[k, i + 1] = z[k, i + 1] - dz
                                        if z[k, i + 1] < - dz / 2:
                                            z[k, i + 1] = z[k, i + 1] + dz

                            for k in range(0, number_of_ions): # calculate acceleration from trap potential modes and collision cooling
                                r_k = np.sqrt(x[k, i] ** 2 + y[k, i] ** 2)
                                if r_k > r_0:
                                    continue
                                else:
                                    ax[k, i + 1] += - Pot_Const * U[i + 1] * x[k, i + 1]
                                    ay[k, i + 1] += + Pot_Const * U[i + 1] * y[k, i + 1]
                                    ax[k, i + 1] += - Oct_Const * U_oct[i + 1] * (2 * x[k, i + 1] ** 3 - 6 * x[k, i + 1] * y[k, i + 1] ** 2)
                                    ay[k, i + 1] += - Oct_Const * U_oct[i + 1] * (2 * y[k, i + 1] ** 3 - 6 * y[k, i + 1] * x[k, i + 1] ** 2)
                                    ax[k, i + 1] += - beta[i +1] * vx[k, i + 1]
                                    ay[k, i + 1] += - beta[i +1] * vy[k, i + 1]
                                    if dimensions == 3:
                                        az[k, i + 1] += - beta[i + 1] * vz[k, i + 1]
                                    if activate_static_B_field == 1:
                                        ax[k, i + 1] += + (e / m) * B_0 * vy[k, i + 1]
                                        ay[k, i + 1] += - (e / m) * B_0 * vx[k, i + 1]
                                    if activate_divergent_B_field == 1:
                                        r_square = np.sqrt(x[k, i + 1] ** 2 + y[k, i + 1] ** 2 + (z[k, i + 1] + B_origin) ** 2) ** 2
                                        ax[k, i + 1] += (e / m) * (vy[k, i + 1] * B_z / r_square - vz[k, i + 1] * B_y / r_square)
                                        ay[k, i + 1] += (e / m) * (vz[k, i + 1] * B_x / r_square - vx[k, i + 1] * B_z / r_square)
                                        if dimensions == 3:
                                            az[k, i + 1] += (e / m) * (vx[k, i + 1] * B_y / r_square - vy[k, i + 1] * B_x / r_square)
                                    if activate_big_coul == 1:
                                        ax[k, i + 1] += activate_big_coul * x[k, i + 1] / r_k ** 3
                                        ay[k, i + 1] += activate_big_coul * y[k, i + 1] / r_k ** 3
                                        if dimensions == 3:
                                            az[k, i + 1] += activate_big_coul * z[k, i + 1] / r_k ** 3

                            for k in range(0, number_of_ions): # calculate acceleration from coulomb interaction between ions
                                for j in range(k + 1, number_of_ions):
                                    r_j = np.sqrt(x[j, i] ** 2 + y[j, i] ** 2)
                                    if r_j > r_0:
                                        continue
                                    else:
                                        if dimensions == 3:
                                            dxy = np.sqrt((x[k, i + 1] - x[j, i + 1]) ** 2 + (y[k, i + 1] - y[j, i + 1]) ** 2 + (
                                                        z[k, i + 1] - z[j, i + 1]) ** 2) ** 3
                                        if dimensions == 2:
                                            dxy = np.sqrt((x[k, i + 1] - x[j, i + 1]) ** 2 + (y[k, i + 1] - y[j, i + 1]) ** 2) ** 3

                                        ax[k, i + 1] += (Coul_Const * (x[k, i + 1] - x[j, i + 1]) / dxy)
                                        ay[k, i + 1] += (Coul_Const * (y[k, i + 1] - y[j, i + 1]) / dxy)
                                        if dimensions == 3:
                                            az[k, i + 1] += (Coul_Const * (z[k, i + 1] - z[j, i + 1]) / dxy)
                                        ax[j, i + 1] += - (Coul_Const * (x[k, i + 1] - x[j, i + 1]) / dxy)
                                        ay[j, i + 1] += - (Coul_Const * (y[k, i + 1] - y[j, i + 1]) / dxy)
                                        if dimensions == 3:
                                            az[j, i + 1] += - (Coul_Const * (z[k, i + 1] - z[j, i + 1]) / dxy)
                                        E_pot[i + 1] += 2 * Coul_Const * (m / dxy)

                            if dimensions == 3: # calculate acceleration from coulomb interaction with fictional mirror-ions in neighbouring, identical trap volumes (allows simulating one "slice" of longer ion trap)
                                for k in range(0, number_of_ions):
                                    for j in range(0, number_of_ions):
                                        r_j = np.sqrt(x[j, i] ** 2 + y[j, i] ** 2)
                                        if r_j > r_0:
                                            continue
                                        else:
                                            if k == j:
                                                continue
                                            else:
                                                dxy1 = np.sqrt((x[k, i + 1] - x[j, i + 1]) ** 2 + (y[k, i + 1] - y[j, i + 1]) ** 2 + (z[k, i + 1] - (z[j, i + 1] - dz)) ** 2) ** 3
                                                ax[k, i + 1] += (Coul_Const * (x[k, i + 1] - x[j, i + 1]) / dxy1)
                                                ay[k, i + 1] += (Coul_Const * (y[k, i + 1] - y[j, i + 1]) / dxy1)
                                                az[k, i + 1] += (Coul_Const * (z[k, i + 1] - (z[j, i + 1] - dz)) / dxy1)

                                                dxy2 = np.sqrt((x[k, i + 1] - x[j, i + 1]) ** 2 + (y[k, i + 1] - y[j, i + 1]) ** 2 + (z[k, i + 1] - (z[j, i + 1] + dz)) ** 2) ** 3
                                                ax[k, i + 1] += (Coul_Const * (x[k, i + 1] - x[j, i + 1]) / dxy2)
                                                ay[k, i + 1] += (Coul_Const * (y[k, i + 1] - y[j, i + 1]) / dxy2)
                                                az[k, i + 1] += (Coul_Const * (z[k, i + 1] - (z[j, i + 1] + dz)) / dxy2)

                        return t, x, y, z, E_kin, E_pot, beta, U

                    t, x, y, z, E_kin, E_pot, beta, U = run(t, x, y, z, vx, vy, vz, ax, ay, az, E_kin, E_pot, spp,
                                                            U, U_oct, B_0, B_x, B_y, B_z, activate_beta_cosine, activate_beta_factor, beta,
                                                            beta_start_switch, switch_beta, beta_factor, activate_big_coul, activate_U_cosine, activate_U_amp_mod, beta_end_switch, activate_static_B_field, activate_divergent_B_field)

                    """Count ions that escape boundaries of trap (radially)"""
                    print("Count ions that escape boundaries of trap (radially)")
                    for k in range(0, number_of_ions):
                        for i in range(1, steps + 2):
                            r = np.sqrt(x[k, i]**2 + y[k, i]**2)
                            if r > r_0:
                                ions_escaped_res[counter] += 1
                                break

                    """Save the largest radius of any ion at last time step to use for ion cloud dimension approximation in further simulations"""
                    r_max_array = np.zeros(number_of_ions)
                    for k in range(0, number_of_ions):
                        r = np.sqrt(x[k, steps+1]**2 + y[k, steps+1]**2)
                        r_max_array[k] = r
                    ion_cloud_radius[counter] = np.amax(r_max_array)

                    """Calculate percentage of ions in beam radius"""
                    print("Calculate percentage of ions in beam radius")
                    for k in range(0, number_of_ions):
                        r = np.zeros(spp+1)
                        for i in range(0, spp+1):
                            r[i] = np.sqrt(x[k, steps+1-i]**2 + y[k, steps+1-i]**2)
                        if all(i <= r_beam for i in r):
                            ions_centered_res[counter] += 1
                    percent_in_laser[counter] = (ions_centered_res[counter] / number_of_ions) * 100

                    """Calculate average energy per period"""
                    print("Calculate average energy per period")
                    E_kin_avg = np.zeros(periods)
                    E_pot_avg = np.zeros(periods)
                    E_tot_avg = np.zeros(periods)
                    t_period = np.zeros(periods)
                    for i in range(periods):
                        E_kin_temp = np.array([0.0])
                        E_pot_temp = np.array([0.0])
                        t_temp = np.array([0.0])
                        for j in range(0, spp+1):
                            E_kin_temp += E_kin[i * spp + j + 1]
                            E_pot_temp += E_pot[i * spp + j + 1]
                            t_temp += t[i * spp + j + 1]
                        E_kin_avg[i] = E_kin_temp / spp
                        E_pot_avg[i] = E_pot_temp / spp
                        E_tot_avg[i] = E_kin_avg[i] + E_pot_avg[i]
                        t_period[i] = t_temp

                    """Plot trajectories"""
                    print("Save selected plots")

                    """Plots for 2D"""
                    plt.clf()
                    if dimensions == 2:
                        if save_default_plots == 1:
                            """Ion cloud plot default"""
                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(x[k, i * spp + 1:(i + 1) * spp + 2], y[k, i * spp + 1:(i + 1) * spp + 2],
                                             linewidth=1)
                                plt.grid()
                                plt.axis('scaled')
                                circle1 = plt.Circle((0, 0), r_0, fill=False)
                                circle2 = plt.Circle((0, 0), r_beam, fill=False)
                                ax = plt.gca()
                                # ax.set_xlim((- 0.0005, 0.0005))
                                # ax.set_ylim((- 0.0005, 0.0005))
                                ax.add_artist(circle1)
                                ax.add_artist(circle2)
                                plt.title(str(i + 1) + '. Periode')
                                plt.xlabel('x')
                                plt.ylabel('y')
                                plt.savefig(results_dir + 'xy_' + str(i + 1) + '_period.png')
                                if i == periods - 1:
                                    if default_view_show == 1:
                                        plt.show()
                                # plt.show()
                                plt.clf()

                        """Ion trajectory in x-t plotted per period"""
                        # for i in range(0, periods):
                        #     for k in range(0, number_of_ions):
                        #         plt.plot(t[i*spp+1:(i+1)*spp+2], x[k, i*spp+1:(i+1)*spp+2])
                        #     plt.grid()
                        #     plt.title(str(i + 1) + '. Periode')
                        #     plt.xlabel('t')
                        #     plt.ylabel('x')
                        #     #plt.savefig(results_dir + 'tx_' + str(i + 1) + '_period.png')
                        #     #plt.show()
                        #     plt.clf()

                        if save_separate_plane_time_plots == 1:
                            """Ion trajectories in both planes separately and combined"""
                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], x[k, i * spp + 1:(i + 1) * spp + 2])
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], y[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.xlabel('t')
                                plt.ylabel('x/y')
                                plt.savefig(results_dir + 'txy.png')
                                # plt.show()
                                plt.clf()

                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], y[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.xlabel('t')
                                plt.ylabel('y')
                                plt.savefig(results_dir + 'ty.png')
                                # plt.show()
                                plt.clf()

                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], x[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.xlabel('t')
                                plt.ylabel('x')
                                plt.savefig(results_dir + 'tx.png')
                                # plt.show()
                                plt.clf()

                        plt.clf()

                        if save_energy_plots == 1:
                            """Kinetic/potential/total energy plots"""
                            for i in range(periods):
                                plt.plot(t_period[:], E_kin_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_kin')
                            plt.savefig(results_dir + 'E_kin_avg.png')
                            # plt.show()
                            plt.clf()

                            for i in range(periods):
                                plt.plot(t_period[:], E_pot_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_pot')
                            plt.savefig(results_dir + 'E_pot_avg.png')
                            # plt.show()
                            plt.clf()

                            for i in range(periods):
                                plt.plot(t_period[:], E_tot_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_tot')
                            plt.savefig(results_dir + 'E_tot_avg.png')
                            # plt.show()

                    """Plots for 3D"""
                    plt.clf()
                    if dimensions == 3:
                        if save_default_plots == 1:
                            """Ion cloud plot default"""
                            for i in range(view_periods_start, periods):
                                fig = plt.figure()
                                ax = fig.gca(projection='3d')
                                #ax.set_axis_off()
                                for k in range(0, number_of_ions):
                                    ax.scatter3D(x[k, i*spp+1:(i+1)*spp+2], z[k, i*spp+1:(i+1)*spp+2], y[k, i*spp+1:(i+1)*spp+2], marker='.')
                                #ax.axes.set_xlim3d(left=-0.0001, right=0.0001)
                                #ax.axes.set_zlim3d(bottom=-0.0001, top=0.0001)
                                ax.axes.set_ylim3d(bottom=-dz/2, top=dz/2)
                                plt.title(str(i + 1) + '. Periode')
                                plt.savefig(results_dir + 'default_view_' + str(i + 1) + '_period.png')
                                if i == periods-1:
                                    if default_view_show == 1:
                                        plt.show()
                                plt.close(fig)

                            """Ion cloud plot view on x-y-plane"""
                            for i in range(view_periods_start, periods):
                                fig = plt.figure()
                                ax = fig.gca(projection='3d')
                                #ax.set_axis_off()
                                for k in range(0, number_of_ions):
                                    ax.scatter3D(x[k, i*spp+1:(i+1)*spp+2], z[k, i*spp+1:(i+1)*spp+2], y[k, i*spp+1:(i+1)*spp+2], marker='.')
                                #ax.axes.set_xlim3d(left=-0.0001, right=0.0001)
                                #ax.axes.set_zlim3d(bottom=-0.0001, top=0.0001)
                                ax.axes.set_ylim3d(bottom=-dz/2, top=dz/2)
                                plt.title(str(i + 1) + '. Periode')
                                ax.view_init(elev=0, azim=270)
                                plt.savefig(results_dir + 'xy_view_' + str(i + 1) + '_period.png')
                                if i == periods-1:
                                    if default_view_show == 1:
                                        plt.show()
                                plt.close(fig)

                            #plt.clf()

                        if save_separate_plane_time_plots == 1:
                            """Ion trajectories in all 3 planes separately"""
                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], z[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.grid()
                                plt.xlabel('t')
                                plt.ylabel('z')
                                plt.savefig(results_dir + 'tz.png')
                                #plt.show()
                                plt.clf()

                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], x[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.grid()
                                plt.xlabel('t')
                                plt.ylabel('x')
                                plt.savefig(results_dir + 'tx.png')
                                #plt.show()
                                plt.clf()

                            for i in range(view_periods_start, periods):
                                for k in range(0, number_of_ions):
                                    plt.plot(t[i * spp + 1:(i + 1) * spp + 2], y[k, i * spp + 1:(i + 1) * spp + 2])
                                plt.grid()
                                plt.xlabel('t')
                                plt.ylabel('y')
                                plt.savefig(results_dir + 'ty.png')
                                #plt.show()
                                plt.clf()

                        if save_energy_plots == 1:
                            """Kinetic/potential/total energy plots"""
                            for i in range(periods):
                                plt.plot(t_period[:], E_kin_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_kin')
                            plt.savefig(results_dir + 'E_kin_avg.png')
                            #plt.show()
                            plt.clf()

                            for i in range(periods):
                                plt.plot(t_period[:], E_pot_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_pot')
                            plt.savefig(results_dir + 'E_pot_avg.png')
                            #plt.show()
                            plt.clf()

                            for i in range(periods):
                                plt.plot(t_period[:], E_tot_avg[:])
                            plt.grid()
                            plt.xlabel('t')
                            plt.ylabel('E_tot')
                            plt.savefig(results_dir + 'E_tot_avg.png')
                            plt.close()

                    """Save trajectories and parameter variations"""
                    print("Save trajectories and parameter variations")

                    if activate_trajectory_saving == 1:
                        np.savetxt(results_dir + 'x' + str(counter) + '_trajectories.txt', x)
                        np.savetxt(results_dir + 'y' + str(counter) + '_trajectories.txt', y)
                        if dimensions == 3:
                            np.savetxt(results_dir + 'z' + str(counter) + '_trajectories.txt', z)

                    if activate_U_rectangle == 1 or activate_U_amp_mod == 1 or activate_U_pulses == 1:
                        np.savetxt(results_dir + 'U.txt', U)
                    if activate_beta_factor == 1 or activate_beta_cosine == 1:
                        np.savetxt(results_dir + 'beta.txt', beta)

    """Save all result arrays to text file"""
    print("Save all results to text file")

    np.savetxt(results_dir + 'results.txt', (index, U_res, amount_res, f_res, q_factor_res, V_eff_res, delta_E_res, ions_escaped_res, beta_res, ions_centered_res, percent_in_laser, dipole_pulse_duration, dipole_pause_duration, ion_cloud_radius, duty_cycle))
    np.savetxt(gen_dir + 'results.txt', (index, U_res, amount_res, f_res, q_factor_res, V_eff_res, delta_E_res, ions_escaped_res, beta_res, ions_centered_res, percent_in_laser, dipole_pulse_duration, dipole_pause_duration, ion_cloud_radius, duty_cycle))

    """Concatenate results to complete results text file"""
    print("Save results to complete results file")

    t1 = np.loadtxt(gen_dir + 'results.txt')
    if not os.path.isfile(gen_dir + 'results_complete.txt'):
        np.savetxt(gen_dir + 'results_complete.txt', t1)
    else:
        if len(index) == 1:
            t2 = np.loadtxt(gen_dir + 'results_complete.txt')
            if t2.ndim == 1:
                t3 = np.concatenate((t1.reshape(-1, 1), t2.reshape(-1, 1)), axis=1)
                np.savetxt(gen_dir + 'results_complete.txt', t3)
            else:
                t3 = np.concatenate((t1.reshape(-1, 1), t2), axis=1)
                np.savetxt(gen_dir + 'results_complete.txt', t3)
        else:
            t2 = np.loadtxt(gen_dir + 'results_complete.txt')
            if t2.ndim == 1:
                t3 = np.concatenate((t1, t2.reshape(-1, 1)), axis=1)
                np.savetxt(gen_dir + 'results_complete.txt', t3)
            else:
                t3 = np.concatenate((t1, t2), axis=1)
                np.savetxt(gen_dir + 'results_complete.txt', t3)

"""EVALUATION"""

"""Phase diagrams based on simulation results"""
print('Creating phase diagrams')
script_dir = os.path.dirname(__file__)
gen_dir = os.path.join(script_dir, str(dimensions) + 'D/' + 'amu' + str(amu) + '/' + 'Quad' + str(activate_quad) + '_Okt' + str(activate_oct) + '_statB' + str(activate_static_B_field) + '_divB' + str(activate_divergent_B_field) + '/')
if activate_U_rectangle == 1:
    gen_dir = os.path.join(gen_dir, 'Urect' + str(activate_U_rectangle) + 'Ucyc' + str(cycleList) + '/')
    if activate_U_pause == 1:
        gen_dir = os.path.join(gen_dir, 'U_pause_start_period' + str(U_pause_start) + '_U_pause_stop_period' + str(U_pause_stop) + '/')
if activate_U_pulses == 1:
    gen_dir = os.path.join(gen_dir, 'Uasym_pulse_steps' + str(asym_pulse_steps) + '_Ucyc' + str(U_asym_cycle) + '_dt' + str(dt) + '/')
if activate_U_cosine == 1:
    gen_dir = os.path.join(gen_dir, 'Ucos' + str(activate_U_cosine) + '/')
    if activate_U_amp_mod == 1:
        gen_dir = os.path.join(gen_dir, 'Uampmod' + str(activate_U_amp_mod) + '_Uampfreq' + str(U_amp_freq) + '_Uampfac' + str(U_amp_factor) + '/')
if activate_beta_cosine == 1:
    gen_dir = os.path.join(gen_dir, 'betacos' + str(activate_beta_cosine) + '_betafreq' + str(beta_cosine_freq) + '_betastart' + str(beta_start_switch) + '_betastop' + str(beta_end_switch) + '/')
if activate_beta_factor == 1:
    gen_dir = os.path.join(gen_dir, 'betafac' + str(beta_factor) + '_betaswitch' + str(switch_beta) + '_betastart' + str(beta_start_switch) + '_betastop' + str(beta_end_switch) + '/')
if activate_U_pulses == 0:
    gen_dir = os.path.join(gen_dir, 'Periods' + str(periods) + '_SPP' + str(spp) + '/')
else:
    gen_dir = os.path.join(gen_dir, 'Periods' + str(periods) + '/')
if not os.path.isdir(gen_dir):
    os.makedirs(gen_dir)

evaluation_dir = os.path.join(gen_dir, 'phase diagrams/')
if not os.path.isdir(evaluation_dir):
    os.makedirs(evaluation_dir)

if evaluate_simulation_results == 1:

    """Density diagram"""
    if radius_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')
        if beta_density_filter == 1:
            results = results[:, results[8] <= beta_density_filter_max]
            results = results[:, results[8] >= beta_density_filter_min]
            plt.title('Filtered for ' + str(beta_density_filter_min) + ' =< beta =< ' + str(beta_density_filter_max))
        results_stable = results[:, results[7] <= 0]
        plt.plot(results_stable[4, :], results_stable[12, :], 'bo')
        plt.grid()
        #plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('q value')
        plt.ylabel('cloud radius')
        plt.savefig(evaluation_dir + 'R_q.png')
        if show_phase_diagrams == 1:
            plt.show()
        plt.clf()

        results = np.loadtxt(gen_dir + 'results.txt')
        if q_density_filter == 1:
            results = results[:, results[4] <= q_density_filter_max]
            results = results[:, results[4] >= q_density_filter_min]
            plt.title('Filtered for ' + str(q_density_filter_min) + ' =< q =< ' + str(q_density_filter_max))
        results_stable = results[:, results[7] <= 0]
        plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
        plt.grid()
        # plt.ylim(0, 1)
        #plt.xlim(0, 1)
        plt.xlabel('beta value')
        plt.ylabel('cloud radius')
        plt.savefig(evaluation_dir + 'R_beta.png')
        if show_phase_diagrams == 1:
            plt.show()
        plt.clf()

    """ n_beam vs. q"""
    if n_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')
        if beta_n_beam_filter == 1:
            results = results[:, results[8] <= beta_n_beam_filter_max]
            results = results[:, results[8] >= beta_n_beam_filter_min]
            plt.title('Filtered for ' + str(beta_n_beam_filter_min) + ' =< beta =< ' + str(beta_n_beam_filter_max))
        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            plt.plot(results_stable[4, :], results_stable[9, :], 'bo', markersize=4)
            plt.plot(results_unstable[4, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            #plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            plt.plot(results_unstable[4, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            plt.plot(results_stable[4, :], results_stable[9, :], 'bo', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ n_beam vs. beta"""
    if n_beta_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')
        if q_n_beam_filter == 1:
            results = results[:, results[4] <= q_n_beam_filter_max+0.01]
            results = results[:, results[4] >= q_n_beam_filter_min-0.01]
            plt.title('Filtered for ' + str(q_n_beam_filter_min) + ' =< q =< ' + str(q_n_beam_filter_max))
        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            plt.plot(results_stable[8, :], results_stable[9, :], 'bo', markersize=4)
            plt.plot(results_unstable[8, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            plt.plot(results_unstable[8, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            plt.plot(results_stable[8, :], results_stable[9, :], 'bo', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ N vs. q """
    if N_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')

        if Nq_beta_filter == 1:
            results = results[:, results[8] <= Nq_beta_filter_max]
            results = results[:, results[8] >= Nq_beta_filter_min]

        if Nq_q_filter == 1:
            results = results[:, results[4] <= Nq_q_filter_max]
            results = results[:, results[4] >= Nq_q_filter_min]

        if Nq_N_filter == 1:
            results = results[:, results[2] >= Nq_N_filter_min]
            results = results[:, results[2] <= Nq_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            if results_unstable.ndim == 1:
                min_unstable_q = results_unstable[4]
                min_unstable_N = results_unstable[2]
            else:
                min_unstable_q = results_unstable[4, :]
                min_unstable_N = results_unstable[2, :]
            if results_stable.ndim == 1:
                max_stable_q = results_stable[4]
                max_stable_N = results_stable[2]
            else:
                max_stable_q = results_stable[4, :]
                max_stable_N = results_stable[2, :]
            min_results_N = results[2, :]
            min_results_N = np.amin(min_results_N)
            max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            max_stable_q = np.amax(max_stable_q)
            min_unstable_q = np.amin(min_unstable_q)
            max_stable_N = np.amax(max_stable_N)
            min_unstable_N = np.amin(min_unstable_N)
            plt.plot(results_stable[2, :], results_stable[4, :], 'bo', markersize=4)
            plt.plot(results_unstable[2, :], results_unstable[4, :], 'ro', markersize=4)
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_N - 100, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                if Nq_N_filter == 1:
                    q_crit = (max_stable_q + min_unstable_q) / 2
                    print('Critical q value between: ' + str(max_stable_q) + ' and ' + str(min_unstable_q))
                    plt.hlines(q_crit, 0, Nq_N_filter_max)
                if Nq_q_filter == 1:
                    N_crit = (max_stable_N + min_unstable_N) / 2
                    print('Critical N value between: ' + str(max_stable_N) + ' and ' + str(min_unstable_N))
                    plt.vlines(N_crit, 0, 1)
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                max_results_N = results[2]
            else:
                max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            plt.plot(results_unstable[2, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(0, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                max_results_N = results[2]
            else:
                max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            plt.plot(results_stable[2, :], results_stable[4, :], 'bo')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(0, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ beta vs. q """
    if beta_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')

        if betaq_beta_filter == 1:
            results = results[:, results[8] <= betaq_beta_filter_max]
            results = results[:, results[8] >= betaq_beta_filter_min]

        if betaq_q_filter == 1:
            results = results[:, results[4] <= betaq_q_filter_max]
            results = results[:, results[4] >= betaq_q_filter_min]

        if betaq_N_filter == 1:
            results = results[:, results[2] >= betaq_N_filter_min]
            results = results[:, results[2] <= betaq_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            min_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = results[8, :]
            max_results_beta = np.amax(max_results_beta)
            if results_unstable.ndim == 1:
                min_unstable_q = results_unstable[4]
                max_unstable_beta = results_unstable[8]
            else:
                min_unstable_q = results_unstable[4, :]
                max_unstable_beta = results_unstable[8, :]
            if results_stable.ndim == 1:
                max_stable_q = results_stable[4]
                min_stable_beta = results_stable[8]
            else:
                max_stable_q = results_stable[4, :]
                min_stable_beta = results_stable[8, :]
            max_stable_q = np.amax(max_stable_q)
            min_unstable_q = np.amin(min_unstable_q)
            min_stable_beta = np.amin(min_stable_beta)
            max_unstable_beta = np.amax(max_unstable_beta)
            plt.plot(results_stable[8, :], results_stable[4, :], 'bo')
            plt.plot(results_unstable[8, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                if betaq_q_filter == 1:
                    beta_crit = (min_stable_beta + max_unstable_beta) / 2
                    print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
                    plt.vlines(beta_crit, 0, 1)
                if betaq_beta_filter == 1:
                    q_crit = (max_stable_q + min_unstable_q) / 2
                    print('Critical q value between: ' + str(max_stable_q) + ' and ' + str(min_unstable_q))
                    plt.hlines(q_crit, betaq_beta_filter_min, betaq_beta_filter_max)
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_unstable[8, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_stable[8, :], results_stable[4, :], 'bo')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    # """ beta vs. pulse pause duration """
    # if beta_pulse_diagram == 1:
    #     results = np.loadtxt(gen_dir + 'results.txt')
    #
    #     if betapulse_beta_filter == 1:
    #         results = results[:, results[8] <= betapulse_beta_filter_max]
    #         results = results[:, results[8] >= betapulse_beta_filter_min]
    #
    #     if betapulse_pulse_filter == 1:
    #         results = results[:, results[12] <= betapulse_pulse_filter_max]
    #         results = results[:, results[12] >= betapulse_pulse_filter_min]
    #
    #     if betapulse_N_filter == 1:
    #         results = results[:, results[2] >= betapulse_N_filter_min]
    #         results = results[:, results[2] <= betapulse_N_filter_max]
    #
    #     results_stable = results[:, results[7] <= 0]
    #     results_unstable = results[:, results[7] > 0]
    #
    #     if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
    #         min_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = results[8, :]
    #         max_results_beta = np.amax(max_results_beta)
    #         if results_unstable.ndim == 1:
    #             min_unstable_pulse = results_unstable[12]
    #             max_unstable_beta = results_unstable[8]
    #         else:
    #             min_unstable_pulse = results_unstable[12, :]
    #             max_unstable_beta = results_unstable[8, :]
    #         if results_stable.ndim == 1:
    #             max_stable_pulse = results_stable[12]
    #             min_stable_beta = results_stable[8]
    #         else:
    #             max_stable_pulse = results_stable[12, :]
    #             min_stable_beta = results_stable[8, :]
    #         max_stable_pulse = np.amax(max_stable_pulse)
    #         min_unstable_pulse = np.amin(min_unstable_pulse)
    #         min_stable_beta = np.amin(min_stable_beta)
    #         max_unstable_beta = np.amax(max_unstable_beta)
    #         plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
    #         plt.plot(results_unstable[8, :], results_unstable[12, :], 'ro')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             if betaq_pulse_filter == 1:
    #                 beta_crit = (min_stable_beta + max_unstable_beta) / 2
    #                 print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
    #                 plt.vlines(beta_crit)
    #             if betapulse_beta_filter == 1:
    #                 pulse_crit = (max_stable_pulse + min_unstable_pulse) / 2
    #                 print('Critical pulse value between: ' + str(max_stable_pulse) + ' and ' + str(min_unstable_pulse))
    #                 plt.hlines(pulse_crit, betapulse_beta_filter_min, betapulse_beta_filter_max)
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_stable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_beta = results[8]
    #             max_results_beta = results[8]
    #         else:
    #             min_results_beta = results[8, :]
    #             max_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = np.amax(max_results_beta)
    #         plt.plot(results_unstable[8, :], results_unstable[12, :], 'ro')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_unstable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_beta = results[8]
    #             max_results_beta = results[8]
    #         else:
    #             min_results_beta = results[8, :]
    #             max_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = np.amax(max_results_beta)
    #         plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     plt.clf()

    #     """ pause duration vs. cloud density """
    # if pause_density_diagram == 1:
    #     results = np.loadtxt(gen_dir + 'results.txt')
    #
    #     if pause_density_beta_filter == 1:
    #         results = results[:, results[8] <= pause_density_beta_filter_max]
    #         results = results[:, results[8] >= pause_density_beta_filter_min]
    #
    #     if pause_density_N_filter == 1:
    #         results = results[:, results[2] >= pause_density_N_filter_min]
    #         results = results[:, results[2] <= pause_density_N_filter_max]
    #
    #     results_stable = results[:, results[7] <= 0]
    #     results_unstable = results[:, results[7] > 0]
    #
    #     if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
    #         min_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = results[9, :]
    #         max_results_density = np.amax(max_results_density)
    #         if results_unstable.ndim == 1:
    #             min_unstable_pulse = results_unstable[12]
    #             max_unstable_density = results_unstable[9]
    #         else:
    #             min_unstable_pulse = results_unstable[12, :]
    #             max_unstable_density = results_unstable[9, :]
    #         if results_stable.ndim == 1:
    #             max_stable_pulse = results_stable[12]
    #             min_stable_density = results_stable[9]
    #         else:
    #             max_stable_pulse = results_stable[12, :]
    #             min_stable_density = results_stable[8, :]
    #         max_stable_pulse = np.amax(max_stable_pulse)
    #         min_unstable_pulse = np.amin(min_unstable_pulse)
    #         min_stable_density = np.amin(min_stable_density)
    #         max_unstable_density = np.amax(max_unstable_density)
    #         plt.plot(results_stable[9, :], results_stable[12, :], 'bo')
    #         plt.plot(results_unstable[9, :], results_unstable[12, :], 'ro')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(pause_density_N_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                 pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_stable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_density = results[9]
    #             max_results_density = results[9]
    #         else:
    #             min_results_density = results[9, :]
    #             max_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = np.amax(max_results_density)
    #         plt.plot(results_unstable[9, :], results_unstable[12, :], 'ro')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(
    #                 pause_density_N_filter_max))
    #             plt.savefig(
    #                 evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                     pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(
    #                 pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(
    #                 pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_unstable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_density = results[9]
    #             max_results_density = results[9]
    #         else:
    #             min_results_density = results[9, :]
    #             max_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = np.amax(max_results_density)
    #         plt.plot(results_stable[9, :], results_stable[12, :], 'bo')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(
    #                 pause_density_N_filter_max))
    #             plt.savefig(
    #                 evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                     pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(
    #                 pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(
    #                 pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #     plt.clf()

    """ beta vs. duty cycle """
    if beta_duty_cycle_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')

        if betacycle_beta_filter == 1:
            results = results[:, results[8] <= betacycle_beta_filter_max]
            results = results[:, results[8] >= betacycle_beta_filter_min]

        if betacycle_pulse_filter == 1:
            results = results[:, results[14] <= betacycle_cycle_filter_max]
            results = results[:, results[14] >= betacycle_cycle_filter_min]

        if betacycle_N_filter == 1:
            results = results[:, results[2] >= betacycle_N_filter_min]
            results = results[:, results[2] <= betacycle_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            min_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = results[8, :]
            max_results_beta = np.amax(max_results_beta)
            if results_unstable.ndim == 1:
                min_unstable_cycle = results_unstable[14]
                max_unstable_beta = results_unstable[8]
            else:
                min_unstable_cycle = results_unstable[14, :]
                max_unstable_beta = results_unstable[8, :]
            if results_stable.ndim == 1:
                max_stable_cycle = results_stable[14]
                min_stable_beta = results_stable[8]
            else:
                max_stable_cycle = results_stable[14, :]
                min_stable_beta = results_stable[8, :]
            max_stable_cycle = np.amax(max_stable_cycle)
            min_unstable_cycle = np.amin(min_unstable_cycle)
            min_stable_beta = np.amin(min_stable_beta)
            max_unstable_beta = np.amax(max_unstable_beta)
            plt.plot(results_stable[8, :], results_stable[14, :], 'bo')
            plt.plot(results_unstable[8, :], results_unstable[14, :], 'ro')
            plt.grid()
            #plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                if betacycle_pulse_filter == 1:
                    beta_crit = (min_stable_beta + max_unstable_beta) / 2
                    print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
                    plt.vlines(beta_crit)
                if betacycle_beta_filter == 1:
                    pulse_crit = (max_stable_cycle + min_unstable_cycle) / 2
                    print('Critical duty cycle between: ' + str(max_stable_cycle) + ' and ' + str(min_unstable_cycle))
                    plt.hlines(pulse_crit, betacycle_beta_filter_min, betacycle_beta_filter_max)
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_unstable[8, :], results_unstable[14, :], 'ro')
            plt.grid()
            #plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_stable[8, :], results_stable[14, :], 'bo')
            plt.grid()
            #plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ duty cycle vs. cloud density """
    if duty_cycle_density_diagram == 1:
        results = np.loadtxt(gen_dir + 'results.txt')

        if cycle_density_beta_filter == 1:
            results = results[:, results[8] <= cycle_density_beta_filter_max]
            results = results[:, results[8] >= cycle_density_beta_filter_min]

        if cycle_density_N_filter == 1:
            results = results[:, results[2] >= cycle_density_N_filter_min]
            results = results[:, results[2] <= cycle_density_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            min_results_density = results[9, :]
            min_results_density = np.amin(min_results_density)
            max_results_density = results[9, :]
            max_results_density = np.amax(max_results_density)
            if results_unstable.ndim == 1:
                min_unstable_cycle = results_unstable[14]
                max_unstable_density = results_unstable[9]
            else:
                min_unstable_cycle = results_unstable[14, :]
                max_unstable_density = results_unstable[9, :]
            if results_stable.ndim == 1:
                max_stable_cycle = results_stable[14]
                min_stable_density = results_stable[9]
            else:
                max_stable_cycle = results_stable[14, :]
                min_stable_density = results_stable[8, :]
            max_stable_cycle = np.amax(max_stable_cycle)
            min_unstable_cycle = np.amin(min_unstable_cycle)
            min_stable_density = np.amin(min_stable_density)
            max_unstable_density = np.amax(max_unstable_density)
            plt.plot(results_stable[9, :], results_stable[14, :], 'bo')
            plt.plot(results_unstable[9, :], results_unstable[14, :], 'ro')
            # plt.hlines(1e-6, min_results_density, max_results_density)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_density, max_results_density)
            plt.xlabel('ion amount in beam radius')
            plt.ylabel('duty cycle')
            if cycle_density_N_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                    cycle_density_N_filter_max))
                plt.savefig(
                    evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                        cycle_density_N_filter_max) + '.png')
            if cycle_density_beta_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                    cycle_density_beta_filter_max))
                plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                    cycle_density_beta_filter_min) + '_betamax' + str(
                    cycle_density_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'duty_cycle_density.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                min_results_density = results[9]
                max_results_density = results[9]
            else:
                min_results_density = results[9, :]
                max_results_density = results[9, :]
            min_results_density = np.amin(min_results_density)
            max_results_density = np.amax(max_results_density)
            plt.plot(results_unstable[9, :], results_unstable[14, :], 'ro')
            # plt.hlines(1e-6, min_results_density, max_results_density)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_density, max_results_density)
            plt.xlabel('ion amount in beam radius')
            plt.ylabel('duty cycle')
            if cycle_density_N_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                    cycle_density_N_filter_max))
                plt.savefig(
                    evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                        cycle_density_N_filter_max) + '.png')
            if cycle_density_beta_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                    cycle_density_beta_filter_max))
                plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                    cycle_density_beta_filter_min) + '_betamax' + str(
                    cycle_density_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'duty_cycle_density.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                min_results_density = results[9]
                max_results_density = results[9]
            else:
                min_results_density = results[9, :]
                max_results_density = results[9, :]
            min_results_density = np.amin(min_results_density)
            max_results_density = np.amax(max_results_density)
            plt.plot(results_stable[9, :], results_stable[14, :], 'bo')
            # plt.hlines(1e-6, min_results_density, max_results_density)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_density, max_results_density)
            plt.xlabel('ion amount in beam radius')
            plt.ylabel('duty cycle')
            if cycle_density_N_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                    cycle_density_N_filter_max))
                plt.savefig(
                    evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                        cycle_density_N_filter_max) + '.png')
            if cycle_density_beta_filter == 1:
                plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                    cycle_density_beta_filter_max))
                plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                    cycle_density_beta_filter_min) + '_betamax' + str(
                    cycle_density_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'duty_cycle_density.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

if evaluate_complete_results == 1:

    """Density diagram"""
    if radius_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')
        if beta_density_filter == 1:
            results = results[:, results[8] <= beta_density_filter_max]
            results = results[:, results[8] >= beta_density_filter_min]
            plt.title('Filtered for ' + str(beta_density_filter_min) + ' =< beta =< ' + str(beta_density_filter_max))
        results_stable = results[:, results[7] <= 0]
        plt.plot(results_stable[4, :], results_stable[12, :], 'bo')
        plt.grid()
        #plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('q value')
        plt.ylabel('cloud radius')
        plt.savefig(evaluation_dir + 'R_q.png')
        if show_phase_diagrams == 1:
            plt.show()
        plt.clf()

        results = np.loadtxt(gen_dir + 'results.txt')
        if q_density_filter == 1:
            results = results[:, results[4] <= q_density_filter_max]
            results = results[:, results[4] >= q_density_filter_min]
            plt.title('Filtered for ' + str(q_density_filter_min) + ' =< q =< ' + str(q_density_filter_max))
        results_stable = results[:, results[7] <= 0]
        plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
        plt.grid()
        # plt.ylim(0, 1)
        #plt.xlim(0, 1)
        plt.xlabel('beta value')
        plt.ylabel('cloud radius')
        plt.savefig(evaluation_dir + 'R_beta.png')
        if show_phase_diagrams == 1:
            plt.show()
        plt.clf()

    """ n_beam vs. q"""
    if n_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')
        if beta_n_beam_filter == 1:
            results = results[:, results[8] <= beta_n_beam_filter_max]
            results = results[:, results[8] >= beta_n_beam_filter_min]
            plt.title('Filtered for ' + str(beta_n_beam_filter_min) + ' =< beta =< ' + str(beta_n_beam_filter_max))
        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            plt.plot(results_stable[4, :], results_stable[9, :], 'bo', markersize=4)
            plt.plot(results_unstable[4, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            #plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            plt.plot(results_unstable[4, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            plt.plot(results_stable[4, :], results_stable[9, :], 'bo', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(0, 1)
            plt.xlabel('q value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ n_beam vs. beta"""
    if n_beta_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')
        if q_n_beam_filter == 1:
            results = results[:, results[4] <= q_n_beam_filter_max+0.01]
            results = results[:, results[4] >= q_n_beam_filter_min-0.01]
            plt.title('Filtered for ' + str(q_n_beam_filter_min) + ' =< q =< ' + str(q_n_beam_filter_max))
        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            plt.plot(results_stable[8, :], results_stable[9, :], 'bo', markersize=4)
            plt.plot(results_unstable[8, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            plt.plot(results_unstable[8, :], results_unstable[9, :], 'ro', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            plt.plot(results_stable[8, :], results_stable[9, :], 'bo', markersize=4)
            plt.grid()
            # plt.ylim(0, 1)
            #plt.xlim(0, 1)
            plt.xlabel('beta value')
            plt.ylabel('ion amount in beam area')
            plt.savefig(evaluation_dir + 'n_beam_beta.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ N vs. q """
    if N_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')

        if Nq_beta_filter == 1:
            results = results[:, results[8] <= Nq_beta_filter_max]
            results = results[:, results[8] >= Nq_beta_filter_min]

        if Nq_q_filter == 1:
            results = results[:, results[4] <= Nq_q_filter_max]
            results = results[:, results[4] >= Nq_q_filter_min]

        if Nq_N_filter == 1:
            results = results[:, results[2] >= Nq_N_filter_min]
            results = results[:, results[2] <= Nq_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            if results_unstable.ndim == 1:
                min_unstable_q = results_unstable[4]
                min_unstable_N = results_unstable[2]
            else:
                min_unstable_q = results_unstable[4, :]
                min_unstable_N = results_unstable[2, :]
            if results_stable.ndim == 1:
                max_stable_q = results_stable[4]
                max_stable_N = results_stable[2]
            else:
                max_stable_q = results_stable[4, :]
                max_stable_N = results_stable[2, :]
            min_results_N = results[2, :]
            min_results_N = np.amin(min_results_N)
            max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            max_stable_q = np.amax(max_stable_q)
            min_unstable_q = np.amin(min_unstable_q)
            max_stable_N = np.amax(max_stable_N)
            min_unstable_N = np.amin(min_unstable_N)
            plt.plot(results_stable[2, :], results_stable[4, :], 'bo', markersize=4)
            plt.plot(results_unstable[2, :], results_unstable[4, :], 'ro', markersize=4)
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_N - 100, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                if Nq_N_filter == 1:
                    q_crit = (max_stable_q + min_unstable_q) / 2
                    print('Critical q value between: ' + str(max_stable_q) + ' and ' + str(min_unstable_q))
                    plt.hlines(q_crit, 0, Nq_N_filter_max)
                if Nq_q_filter == 1:
                    N_crit = (max_stable_N + min_unstable_N) / 2
                    print('Critical N value between: ' + str(max_stable_N) + ' and ' + str(min_unstable_N))
                    plt.vlines(N_crit, 0, 1)
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                max_results_N = results[2]
            else:
                max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            plt.plot(results_unstable[2, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(0, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                max_results_N = results[2]
            else:
                max_results_N = results[2, :]
            max_results_N = np.amax(max_results_N)
            plt.plot(results_stable[2, :], results_stable[4, :], 'bo')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(0, max_results_N + 100)
            plt.xlabel('Ion count N')
            plt.ylabel('q value')
            if Nq_beta_filter == 1:
                plt.title('Filtered for ' + str(Nq_beta_filter_min) + ' =< beta =< ' + str(Nq_beta_filter_max))
                plt.savefig(evaluation_dir + 'N_q_betamin' + str(Nq_beta_filter_min) + '_betamax' + str(Nq_beta_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'N_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    """ beta vs. q """
    if beta_q_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')

        if betaq_beta_filter == 1:
            results = results[:, results[8] <= betaq_beta_filter_max]
            results = results[:, results[8] >= betaq_beta_filter_min]

        if betaq_q_filter == 1:
            results = results[:, results[4] <= betaq_q_filter_max]
            results = results[:, results[4] >= betaq_q_filter_min]

        if betaq_N_filter == 1:
            results = results[:, results[2] >= betaq_N_filter_min]
            results = results[:, results[2] <= betaq_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            min_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = results[8, :]
            max_results_beta = np.amax(max_results_beta)
            if results_unstable.ndim == 1:
                min_unstable_q = results_unstable[4]
                max_unstable_beta = results_unstable[8]
            else:
                min_unstable_q = results_unstable[4, :]
                max_unstable_beta = results_unstable[8, :]
            if results_stable.ndim == 1:
                max_stable_q = results_stable[4]
                min_stable_beta = results_stable[8]
            else:
                max_stable_q = results_stable[4, :]
                min_stable_beta = results_stable[8, :]
            max_stable_q = np.amax(max_stable_q)
            min_unstable_q = np.amin(min_unstable_q)
            min_stable_beta = np.amin(min_stable_beta)
            max_unstable_beta = np.amax(max_unstable_beta)
            plt.plot(results_stable[8, :], results_stable[4, :], 'bo')
            plt.plot(results_unstable[8, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                if betaq_q_filter == 1:
                    beta_crit = (min_stable_beta + max_unstable_beta) / 2
                    print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
                    plt.vlines(beta_crit, 0, 1)
                if betaq_beta_filter == 1:
                    q_crit = (max_stable_q + min_unstable_q) / 2
                    print('Critical q value between: ' + str(max_stable_q) + ' and ' + str(min_unstable_q))
                    plt.hlines(q_crit, betaq_beta_filter_min, betaq_beta_filter_max)
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_unstable[8, :], results_unstable[4, :], 'ro')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_stable[8, :], results_stable[4, :], 'bo')
            plt.grid()
            plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('q value')
            if betaq_N_filter == 1:
                plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_q_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_q.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    # """ beta vs. pulse pause duration """
    # if beta_cycle_diagram == 1:
    #     results = np.loadtxt(gen_dir + 'results_complete.txt')
    #
    #     if betacycle_beta_filter == 1:
    #         results = results[:, results[8] <= betacycle_beta_filter_max]
    #         results = results[:, results[8] >= betacycle_beta_filter_min]
    #
    #     if betacycle_pulse_filter == 1:
    #         results = results[:, results[12] <= betacycle_cycle_filter_max]
    #         results = results[:, results[12] >= betacycle_cycle_filter_min]
    #
    #     if betacycle_N_filter == 1:
    #         results = results[:, results[2] >= betacycle_N_filter_min]
    #         results = results[:, results[2] <= betacycle_N_filter_max]
    #
    #     results_stable = results[:, results[7] <= 0]
    #     results_unstable = results[:, results[7] > 0]
    #
    #     if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
    #         min_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = results[8, :]
    #         max_results_beta = np.amax(max_results_beta)
    #         if results_unstable.ndim == 1:
    #             min_unstable_cycle = results_unstable[12]
    #             max_unstable_beta = results_unstable[8]
    #         else:
    #             min_unstable_cycle = results_unstable[12, :]
    #             max_unstable_beta = results_unstable[8, :]
    #         if results_stable.ndim == 1:
    #             max_stable_cycle = results_stable[12]
    #             min_stable_beta = results_stable[8]
    #         else:
    #             max_stable_cycle = results_stable[12, :]
    #             min_stable_beta = results_stable[8, :]
    #         max_stable_cycle = np.amax(max_stable_cycle)
    #         min_unstable_cycle = np.amin(min_unstable_cycle)
    #         min_stable_beta = np.amin(min_stable_beta)
    #         max_unstable_beta = np.amax(max_unstable_beta)
    #         plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
    #         plt.plot(results_unstable[8, :], results_unstable[12, :], 'ro')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             if betaq_pulse_filter == 1:
    #                 beta_crit = (min_stable_beta + max_unstable_beta) / 2
    #                 print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
    #                 plt.vlines(beta_crit)
    #             if betacycle_beta_filter == 1:
    #                 pulse_crit = (max_stable_cycle + min_unstable_cycle) / 2
    #                 print('Critical pulse value between: ' + str(max_stable_cycle) + ' and ' + str(min_unstable_cycle))
    #                 plt.hlines(pulse_crit, betacycle_beta_filter_min, betacycle_beta_filter_max)
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_stable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_beta = results[8]
    #             max_results_beta = results[8]
    #         else:
    #             min_results_beta = results[8, :]
    #             max_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = np.amax(max_results_beta)
    #         plt.plot(results_unstable[8, :], results_unstable[12, :], 'ro')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_unstable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_beta = results[8]
    #             max_results_beta = results[8]
    #         else:
    #             min_results_beta = results[8, :]
    #             max_results_beta = results[8, :]
    #         min_results_beta = np.amin(min_results_beta)
    #         max_results_beta = np.amax(max_results_beta)
    #         plt.plot(results_stable[8, :], results_stable[12, :], 'bo')
    #         plt.grid()
    #         #plt.ylim(0, 1)
    #         plt.xlim(min_results_beta, max_results_beta)
    #         plt.xlabel('beta')
    #         plt.ylabel('pulse pause duration')
    #         if betaq_N_filter == 1:
    #             plt.title('Filtered for ' + str(betaq_N_filter_min) + ' =< N =< ' + str(betaq_N_filter_max))
    #             plt.savefig(evaluation_dir + 'beta_pulse_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(betaq_N_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'beta_pulse_pause.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     plt.clf()

    """ beta vs. duty cycle """
    if beta_duty_cycle_diagram == 1:
        results = np.loadtxt(gen_dir + 'results_complete.txt')

        if betacycle_beta_filter == 1:
            results = results[:, results[8] <= betacycle_beta_filter_max]
            results = results[:, results[8] >= betacycle_beta_filter_min]

        if betacycle_pulse_filter == 1:
            results = results[:, results[14] <= betacycle_cycle_filter_max]
            results = results[:, results[14] >= betacycle_cycle_filter_min]

        if betacycle_N_filter == 1:
            results = results[:, results[2] >= betacycle_N_filter_min]
            results = results[:, results[2] <= betacycle_N_filter_max]

        results_stable = results[:, results[7] <= 0]
        results_unstable = results[:, results[7] > 0]

        if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
            min_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = results[8, :]
            max_results_beta = np.amax(max_results_beta)
            if results_unstable.ndim == 1:
                min_unstable_cycle = results_unstable[14]
                max_unstable_beta = results_unstable[8]
            else:
                min_unstable_cycle = results_unstable[14, :]
                max_unstable_beta = results_unstable[8, :]
            if results_stable.ndim == 1:
                max_stable_cycle = results_stable[14]
                min_stable_beta = results_stable[8]
            else:
                max_stable_cycle = results_stable[14, :]
                min_stable_beta = results_stable[8, :]
            max_stable_cycle = np.amax(max_stable_cycle)
            min_unstable_cycle = np.amin(min_unstable_cycle)
            min_stable_beta = np.amin(min_stable_beta)
            max_unstable_beta = np.amax(max_unstable_beta)
            plt.plot(results_stable[8, :], results_stable[14, :], 'bo')
            plt.plot(results_unstable[8, :], results_unstable[14, :], 'ro')
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                if betacycle_pulse_filter == 1:
                    beta_crit = (min_stable_beta + max_unstable_beta) / 2
                    print('Critical beta value between: ' + str(min_stable_beta) + ' and ' + str(max_unstable_beta))
                    plt.vlines(beta_crit)
                if betacycle_beta_filter == 1:
                    pulse_crit = (max_stable_cycle + min_unstable_cycle) / 2
                    print(
                        'Critical duty cycle between: ' + str(max_stable_cycle) + ' and ' + str(min_unstable_cycle))
                    plt.hlines(pulse_crit, betacycle_beta_filter_min, betacycle_beta_filter_max)
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(
                    betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_stable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_unstable[8, :], results_unstable[14, :], 'ro')
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(
                    betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()

        if len(results_unstable[0, :]) == 0:
            if results.ndim == 1:
                min_results_beta = results[8]
                max_results_beta = results[8]
            else:
                min_results_beta = results[8, :]
                max_results_beta = results[8, :]
            min_results_beta = np.amin(min_results_beta)
            max_results_beta = np.amax(max_results_beta)
            plt.plot(results_stable[8, :], results_stable[14, :], 'bo')
            plt.grid()
            # plt.ylim(0, 1)
            plt.xlim(min_results_beta, max_results_beta)
            plt.xlabel('beta')
            plt.ylabel('duty cycle')
            if betacycle_N_filter == 1:
                plt.title('Filtered for ' + str(betacycle_N_filter_min) + ' =< N =< ' + str(betacycle_N_filter_max))
                plt.savefig(evaluation_dir + 'beta_duty_cycle_Nmin' + str(betaq_N_filter_min) + '_Nmax' + str(
                    betaq_N_filter_max) + '.png')
            else:
                plt.savefig(evaluation_dir + 'beta_duty_cycle.png')
            if show_phase_diagrams == 1:
                plt.show()
        plt.clf()

    #     """ pause duration vs. cloud density """
    # if pause_density_diagram == 1:
    #     results = np.loadtxt(gen_dir + 'results_complete.txt')
    #
    #     if pause_density_beta_filter == 1:
    #         results = results[:, results[8] <= pause_density_beta_filter_max]
    #         results = results[:, results[8] >= pause_density_beta_filter_min]
    #
    #     if pause_density_N_filter == 1:
    #         results = results[:, results[2] >= pause_density_N_filter_min]
    #         results = results[:, results[2] <= pause_density_N_filter_max]
    #
    #     results_stable = results[:, results[7] <= 0]
    #     results_unstable = results[:, results[7] > 0]
    #
    #     if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
    #         min_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = results[9, :]
    #         max_results_density = np.amax(max_results_density)
    #         if results_unstable.ndim == 1:
    #             min_unstable_pulse = results_unstable[12]
    #             max_unstable_density = results_unstable[9]
    #         else:
    #             min_unstable_pulse = results_unstable[12, :]
    #             max_unstable_density = results_unstable[9, :]
    #         if results_stable.ndim == 1:
    #             max_stable_pulse = results_stable[12]
    #             min_stable_density = results_stable[9]
    #         else:
    #             max_stable_pulse = results_stable[12, :]
    #             min_stable_density = results_stable[8, :]
    #         max_stable_pulse = np.amax(max_stable_pulse)
    #         min_unstable_pulse = np.amin(min_unstable_pulse)
    #         min_stable_density = np.amin(min_stable_density)
    #         max_unstable_density = np.amax(max_unstable_density)
    #         plt.plot(results_stable[9, :], results_stable[12, :], 'bo')
    #         plt.plot(results_unstable[9, :], results_unstable[12, :], 'ro')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(pause_density_N_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                 pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_stable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_density = results[9]
    #             max_results_density = results[9]
    #         else:
    #             min_results_density = results[9, :]
    #             max_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = np.amax(max_results_density)
    #         plt.plot(results_unstable[9, :], results_unstable[12, :], 'ro')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(
    #                 pause_density_N_filter_max))
    #             plt.savefig(
    #                 evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                     pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(
    #                 pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(
    #                 pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #
    #     if len(results_unstable[0, :]) == 0:
    #         if results.ndim == 1:
    #             min_results_density = results[9]
    #             max_results_density = results[9]
    #         else:
    #             min_results_density = results[9, :]
    #             max_results_density = results[9, :]
    #         min_results_density = np.amin(min_results_density)
    #         max_results_density = np.amax(max_results_density)
    #         plt.plot(results_stable[9, :], results_stable[12, :], 'bo')
    #         plt.hlines(1e-6, min_results_density, max_results_density)
    #         plt.grid()
    #         # plt.ylim(0, 1)
    #         plt.xlim(min_results_density, max_results_density)
    #         plt.xlabel('ion amount in beam radius')
    #         plt.ylabel('pulse pause duration')
    #         if pause_density_N_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_N_filter_min) + ' =< N =< ' + str(
    #                 pause_density_N_filter_max))
    #             plt.savefig(
    #                 evaluation_dir + 'pulse_pause_density_Nmin' + str(pause_density_N_filter_min) + '_Nmax' + str(
    #                     pause_density_N_filter_max) + '.png')
    #         if pause_density_beta_filter == 1:
    #             plt.title('Filtered for ' + str(pause_density_beta_filter_min) + ' =< beta =< ' + str(
    #                 pause_density_beta_filter_max))
    #             plt.savefig(evaluation_dir + 'pulse_pause_density_betamin' + str(
    #                 pause_density_beta_filter_min) + '_betamax' + str(
    #                 pause_density_beta_filter_max) + '.png')
    #         else:
    #             plt.savefig(evaluation_dir + 'pulse_pause_density.png')
    #         if show_phase_diagrams == 1:
    #             plt.show()
    #     plt.clf()

        """ duty cycle vs. cloud density """
        if duty_cycle_density_diagram == 1:
            results = np.loadtxt(gen_dir + 'results_complete.txt')

            if cycle_density_beta_filter == 1:
                results = results[:, results[8] <= cycle_density_beta_filter_max]
                results = results[:, results[8] >= cycle_density_beta_filter_min]

            if cycle_density_N_filter == 1:
                results = results[:, results[2] >= cycle_density_N_filter_min]
                results = results[:, results[2] <= cycle_density_N_filter_max]

            results_stable = results[:, results[7] <= 0]
            results_unstable = results[:, results[7] > 0]

            if len(results_stable[0, :]) and len(results_unstable[0, :]) > 0:
                min_results_density = results[9, :]
                min_results_density = np.amin(min_results_density)
                max_results_density = results[9, :]
                max_results_density = np.amax(max_results_density)
                if results_unstable.ndim == 1:
                    min_unstable_cycle = results_unstable[14]
                    max_unstable_density = results_unstable[9]
                else:
                    min_unstable_cycle = results_unstable[14, :]
                    max_unstable_density = results_unstable[9, :]
                if results_stable.ndim == 1:
                    max_stable_cycle = results_stable[14]
                    min_stable_density = results_stable[9]
                else:
                    max_stable_cycle = results_stable[14, :]
                    min_stable_density = results_stable[8, :]
                max_stable_cycle = np.amax(max_stable_cycle)
                min_unstable_cycle = np.amin(min_unstable_cycle)
                min_stable_density = np.amin(min_stable_density)
                max_unstable_density = np.amax(max_unstable_density)
                plt.plot(results_stable[9, :], results_stable[14, :], 'bo')
                plt.plot(results_unstable[9, :], results_unstable[14, :], 'ro')
                #plt.hlines(1e-6, min_results_density, max_results_density)
                plt.grid()
                # plt.ylim(0, 1)
                plt.xlim(min_results_density, max_results_density)
                plt.xlabel('ion amount in beam radius')
                plt.ylabel('duty cycle')
                if cycle_density_N_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                        cycle_density_N_filter_max))
                    plt.savefig(
                        evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                            cycle_density_N_filter_max) + '.png')
                if cycle_density_beta_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                        cycle_density_beta_filter_max))
                    plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                        cycle_density_beta_filter_min) + '_betamax' + str(
                        cycle_density_beta_filter_max) + '.png')
                else:
                    plt.savefig(evaluation_dir + 'duty_cycle_density.png')
                if show_phase_diagrams == 1:
                    plt.show()

            if len(results_stable[0, :]) == 0:
                if results.ndim == 1:
                    min_results_density = results[9]
                    max_results_density = results[9]
                else:
                    min_results_density = results[9, :]
                    max_results_density = results[9, :]
                min_results_density = np.amin(min_results_density)
                max_results_density = np.amax(max_results_density)
                plt.plot(results_unstable[9, :], results_unstable[14, :], 'ro')
                #plt.hlines(1e-6, min_results_density, max_results_density)
                plt.grid()
                # plt.ylim(0, 1)
                plt.xlim(min_results_density, max_results_density)
                plt.xlabel('ion amount in beam radius')
                plt.ylabel('duty cycle')
                if cycle_density_N_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                        cycle_density_N_filter_max))
                    plt.savefig(
                        evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                            cycle_density_N_filter_max) + '.png')
                if cycle_density_beta_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                        cycle_density_beta_filter_max))
                    plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                        cycle_density_beta_filter_min) + '_betamax' + str(
                        cycle_density_beta_filter_max) + '.png')
                else:
                    plt.savefig(evaluation_dir + 'duty_cycle_density.png')
                if show_phase_diagrams == 1:
                    plt.show()

            if len(results_unstable[0, :]) == 0:
                if results.ndim == 1:
                    min_results_density = results[9]
                    max_results_density = results[9]
                else:
                    min_results_density = results[9, :]
                    max_results_density = results[9, :]
                min_results_density = np.amin(min_results_density)
                max_results_density = np.amax(max_results_density)
                plt.plot(results_stable[9, :], results_stable[14, :], 'bo')
                #plt.hlines(1e-6, min_results_density, max_results_density)
                plt.grid()
                # plt.ylim(0, 1)
                plt.xlim(min_results_density, max_results_density)
                plt.xlabel('ion amount in beam radius')
                plt.ylabel('duty cycle')
                if cycle_density_N_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_N_filter_min) + ' =< N =< ' + str(
                        cycle_density_N_filter_max))
                    plt.savefig(
                        evaluation_dir + 'duty_cycle_density_Nmin' + str(cycle_density_N_filter_min) + '_Nmax' + str(
                            cycle_density_N_filter_max) + '.png')
                if cycle_density_beta_filter == 1:
                    plt.title('Filtered for ' + str(cycle_density_beta_filter_min) + ' =< beta =< ' + str(
                        cycle_density_beta_filter_max))
                    plt.savefig(evaluation_dir + 'duty_cycle_density_betamin' + str(
                        cycle_density_beta_filter_min) + '_betamax' + str(
                        cycle_density_beta_filter_max) + '.png')
                else:
                    plt.savefig(evaluation_dir + 'duty_cycle_density.png')
                if show_phase_diagrams == 1:
                    plt.show()
            plt.clf()

end = time.time()
print('time elapsed: ' + str(np.round(end - begin)) + ' seconds')