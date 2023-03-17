import pandas as pd

import numpy as np
from cantera import CanteraError
from matplotlib import pyplot as plt
import cantera as ct
from Data.Labels import IDT_Label


def show_curve(record_curve, record_time, show_ratio=1.0):
    """Helping method that illustrates the inputted curve in a matplotlib graph.

    Parameters
    ----------
    record_curve: List[float]
        Y-label values.
    record_time: List[float]
        X-label values.
    show_ratio: float
        Ratio of data points being illustrated.

    Returns
    -------
    """
    n = int(len(record_time) * show_ratio)
    plt.figure(figsize=(10, 5))
    # plt.plot(record_time[:n], np.log(record_curve[:n]), 'ob', markersize=1, label='OH')
    plt.plot(np.log10(record_time[:n]), np.log10(record_curve[:n]), 'ob', markersize=1, label='OH')
    # plt.plot(np.log10(record_time[:n]), np.log10(record_curve[:n]), 'ob', markersize=1, label='OH')
    # plt.plot(record_time, self.X_upper_bound, marker='*', ms=10, label='upper_bound')
    plt.legend()
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel('time')
    plt.ylabel("value")
    plt.title("OH concentration")
    plt.show()


def computeIDPbyOHpeak(record_OH_concentration, record_time):
    """IDT is the last time point before the drop of OH concentration.

    Parameters
    ----------
    record_OH_concentration: List[float]
        OH_concentration Y-label values.
    record_time: List[float]
        OH_concentration X-label values.

    Returns
    -------
    float:
        IDT.
    """
    if len(record_OH_concentration) <= 1:
        return 1e-7
    d_OH_concentration = np.diff(record_OH_concentration)
    peak_index = np.argmax(d_OH_concentration < 0)
    return record_time[peak_index]


def computeIDPbyTemperatureChange(record_temperature, record_time):
    """IDP is the time point after which the temperature increases the most.

    Parameters
    ----------
    record_OH_concentration: List[float]
        OH_concentration Y-label values.
    record_time: List[float]
        OH_concentration X-label values.

    Returns
    -------
    float:
        IDT.
    """
    if len(record_temperature) <= 1:
        return 1e-7
    d_temperature_d_t = np.divide(np.diff(record_temperature), np.diff(record_time))
    vol_dTdt = zip(d_temperature_d_t, record_time)
    return max(vol_dTdt)[1]


def computeIDPbyIncreaseOfOH(record_OH_concentration, record_time):
    """IDP can be the time point when OH concentration starts to increase rapidly.
     Source: [An Optimization of a Joint Hydrogen and Syngas Combustion Mechanism] (HongXin Wang's master thesis)

    Parameters
    ----------
    record_OH_concentration: List[float]
        OH_concentration Y-label values.
    record_time: List[float]
        OH_concentration X-label values.

    Returns
    -------
    float:
        IDT.
    """
    if len(record_OH_concentration) <= 1:
        return 1e-7
    d_OH_concentration_d_time = np.diff(record_OH_concentration) / np.diff(record_time)
    max_d_OH_Index = np.argmax(d_OH_concentration_d_time)
    IDT_OH_increase = -record_OH_concentration[max_d_OH_Index] / d_OH_concentration_d_time[max_d_OH_Index] + \
                      record_time[max_d_OH_Index]
    return IDT_OH_increase


def compute_IDP_error(idp_Label: IDT_Label, gas, average_rate=0.5, save_path="CANTERAIDT.csv", save=False):
    """Kernel computation module of the program.

    Parameters
    ----------
    idp_Label: IDT_Label object
        Experimental data.
    gas: Gas object
        Tested mechanism.
    average_rate: float
        Hyper-parameter defining the weight of average term in the IDT error:
        IDT error = average_rate * average_disagreement + (1-average_rate) * max_disagreement

    Returns
    -------
    float:
        IDT error.
    """
    # # place holder error
    # error = np.sum(self.learnable_parameters.X[index])

    IDT_results = np.zeros((idp_Label.GroNum, idp_Label.PointNumMax, 7))  # used to save IDP results
    for group_index in range(idp_Label.GroNum):
        for point_index in range(idp_Label.PointNumMax):
            # test whether this group actually has this point?
            if idp_Label.T5[group_index, point_index] > 0 and idp_Label.IDTrun[
                group_index, point_index] > 0:
                # configure temperature, pressure and species (via text input)
                configuration_text = idp_Label.FuelName[0] + ':' + str(
                    idp_Label.FuelMF[group_index, point_index, 0])
                if idp_Label.FuelNum > 1:
                    for species_index_minus_1 in range(idp_Label.FuelNum - 1):
                        configuration_text = configuration_text + ', ' + idp_Label.FuelName[
                            species_index_minus_1 + 1] + ':' + str(
                            idp_Label.FuelMF[group_index, point_index, species_index_minus_1 + 1])
                gas.TPX = idp_Label.T5[group_index, point_index], idp_Label.p5[
                    group_index, point_index], configuration_text
                reac = ct.IdealGasReactor(contents=gas, name="Batch Reactor")
                # reac = ct.IdealGasReactor(contents=gas, name="ST")
                env = ct.Reservoir(ct.Solution('air.xml'))
                wall = ct.Wall(reac, env, A=1.0, velocity=0)
                netw = ct.ReactorNet([reac])

                # Todo: set tolerance by hard-coding
                # netw.atol = 1e-11
                # netw.rtol = 1e-14
                # Todo: remove hard-coding

                # cache for recording
                record_temperature = []
                record_pressure = []
                record_volume = []
                record_time = []
                record_OH_concentration = []
                # run the simulation for objective time

                while netw.time < idp_Label.IDTrun[group_index, point_index]:
                    # record the time, temperature, pressure and volume for each time step
                    record_time.append(netw.time)
                    record_temperature.append(reac.T)
                    record_pressure.append(reac.thermo.P)
                    record_volume.append(reac.volume)
                    SpPeak = float(reac.thermo[idp_Label.PeakName].Y)
                    record_OH_concentration.append(SpPeak)
                    try:
                        netw.step()
                    except CanteraError:
                        print(
                            "It is a CanteraError during combustion simulation! This simulation process ends but the gene is not discarded.")
                        break
                    except RuntimeError:
                        print(
                            "It is a RuntimeError during combustion simulation! This simulation process ends but the gene is not discarded.")
                        break

                record_temperature = np.array(record_temperature)
                record_pressure = np.array(record_pressure)
                record_volume = np.array(record_volume)
                record_time = np.array(record_time)
                record_OH_concentration = np.array(record_OH_concentration)  # OH

                IDP_by_temperature_increase = computeIDPbyTemperatureChange(record_temperature, record_time)
                IDP_by_OH_peak = computeIDPbyOHpeak(record_OH_concentration, record_time)
                IDP_by_of_OH_increase = computeIDPbyIncreaseOfOH(record_OH_concentration, record_time)

                # save results (column 3 and 4 are empty)
                # Todo: add all records to queue
                IDT_results[group_index, point_index, 0] = idp_Label.T5[group_index, point_index]
                IDT_results[group_index, point_index, 1] = group_index + 1
                IDT_results[group_index, point_index, 2] = point_index + 1
                IDT_results[group_index, point_index, 5] = IDP_by_of_OH_increase
                IDT_results[group_index, point_index, 6] = IDP_by_OH_peak

                if idp_Label.IDTmethod[group_index][point_index] == 'IDTOHinc':
                    idp_Label.results.tarM[group_index, point_index] = IDP_by_of_OH_increase
                elif idp_Label.IDTmethod[group_index][point_index] == 'IDTOHpeak':
                    idp_Label.results.tarM[group_index, point_index] = IDP_by_OH_peak
                    # show_curve(record_temperature, record_time)
                else:
                    raise Exception("Unknown IDP method!")

    # calculate error
    if save:
        np.savetxt(save_path, np.reshape(idp_Label.results.tarM, (-1, 1)), delimiter=",")
    error = np.log10(idp_Label.results.tarM) - np.log10(idp_Label.ExpIDT)
    # print(f"simulation: {idp_Label.results.tarM}, ground truth: {idp_Label.ExpIDT}")
    error = error ** 2
    uncertainty = ((np.log10(1 + idp_Label.ExpUn) - np.log10(1 - idp_Label.ExpUn)) / 2) ** 2.0
    error /= uncertainty
    idp_Label.results.Etar = error
    idp_Label.results.Egro = np.average(error, axis=1)
    # Todo: remove this
    # average_rate = 1.0
    error = np.average(error) * average_rate + np.max(error) * (1 - average_rate)

    return error


def compute_PFR_error(pfr_Label, gas):
    """Kernel computation module of the program.

    Parameters
    ----------
    pfr_Label: PFR_Label object
        Experimental data.
    gas: Gas object
        Tested mechanism.
    Returns
    -------
    float:
        PFR error.
    """
    PFR_results = np.zeros((pfr_Label.GroNum, pfr_Label.PointNumMax, 5))  # used to save PFR results
    for group_index in range(pfr_Label.GroNum):
        for point_index in range(pfr_Label.PointNumMax):
            # test whether this group actually has this point?
            if pfr_Label.T5[group_index, point_index] > 0 and pfr_Label.Runtime[
                group_index, point_index] > 0:
                # configure species (via text input)
                FuelSet = pfr_Label.FuelName[0] + ':' + str(pfr_Label.FuelMF[group_index, point_index, 0])
                if pfr_Label.FuelNum > 1:
                    for II in range(pfr_Label.FuelNum - 1):
                        FuelSet = FuelSet + ', ' + pfr_Label.FuelName[II + 1] + ':' + str(
                            pfr_Label.FuelMF[group_index, point_index, II + 1])
                gas.TPX = pfr_Label.T5[group_index, point_index], pfr_Label.p5[
                    group_index, point_index], FuelSet
                reactor = ct.IdealGasConstPressureReactor(contents=gas, energy='off')
                reactor_net = ct.ReactorNet([reactor])
                n_steps = pfr_Label.timesteps

                duration_of_each_step = pfr_Label.Runtime[group_index, point_index] / n_steps
                t1 = (np.arange(n_steps) + 1) * duration_of_each_step
                # cache for recording
                record_CH3OH_concentration = []
                record_C02_concentration = []
                record_time = []
                states = ct.SolutionArray(reactor.thermo)

                # perform simulation
                for n1, t_i in enumerate(t1):
                    record_time.append(t_i)
                    reactor_net.rtol = 1.0e-5
                    reactor_net.atol = 1.0e-28
                    SpPeak_1 = float(reactor.thermo[pfr_Label.PeakName_1].X)
                    SpPeak_2 = float(reactor.thermo[pfr_Label.PeakName_2].X)
                    record_CH3OH_concentration.append(SpPeak_1)
                    record_C02_concentration.append(SpPeak_2)
                    reactor_net.advance(t_i)
                    states.append(reactor.thermo.state)

                record_CH3OH_concentration = np.array(record_CH3OH_concentration)
                record_C02_concentration = np.array(record_C02_concentration)
                record_time = np.array(record_time)
                min_CH3OH_concentration = min(record_CH3OH_concentration)
                max_record_C02_concentration = max(record_C02_concentration)

                # calculate error

                if pfr_Label.SpecieName1[group_index] == pfr_Label.PeakName_1:  # specie is CH3OH
                    a = np.where(record_CH3OH_concentration >= pfr_Label.Expdata[group_index, point_index])[0]
                    b = np.max(a) if a.size > 0 else 0
                    pfr_Label.t[
                        group_index, point_index] = duration_of_each_step * b  # last time point when simulated CH3OH concentration is higher than the experienced data

                if pfr_Label.SpecieName1[group_index] == pfr_Label.PeakName_2:  # specie is CO2
                    a = np.where(record_C02_concentration <= pfr_Label.Expdata[group_index, point_index])[0]
                    b = np.max(a) if a.size > 0 else 0
                    pfr_Label.t[
                        group_index, point_index] = duration_of_each_step * b  # last time point when simulated CH3OH concentration is higher than the experienced data

                PFR_results[group_index, point_index, 0] = pfr_Label.T5[group_index, point_index]
                PFR_results[group_index, point_index, 1] = group_index + 1
                PFR_results[group_index, point_index, 2] = point_index + 1
                # SimIDT[I, J, 3] = IDTSpPeak
                PFR_results[group_index, point_index, 3] = min_CH3OH_concentration
                PFR_results[group_index, point_index, 4] = max_record_C02_concentration

    pfr_Label.results.tarM = pfr_Label.t
    error = pfr_Label.t - pfr_Label.Exptime
    error = error ** 2
    error /= (pfr_Label.Exptime * pfr_Label.ExpUn) ** 2

    pfr_Label.results.Etar = error
    pfr_Label.results.Egro = np.average(error, axis=1)
    error = np.average(error)
    return error


def compute_sensitivity(idp_Label: IDT_Label, gas):
    """Kernel computation module of the program.

    Parameters
    ----------
    idp_Label: IDT_Label object
        Experimental data.
    gas: Gas object
        Tested mechanism.
    Returns
    -------
    List[float]:
        Reactions sensitivities.
    """
    n_reactions = len(gas.reactions())
    n_cases = 0
    average_sensitivities = np.zeros((n_reactions,))
    for group_index in range(idp_Label.GroNum):
        for point_index in range(idp_Label.PointNumMax):
            n_cases += 1
            print(f"Sensitivity analyze for case {n_cases}.")
            # test whether this group actually has this point?
            if idp_Label.T5[group_index, point_index] > 0 and idp_Label.IDTrun[
                group_index, point_index] > 0:
                # configure temperature, pressure and species (via text input)
                configuration_text = idp_Label.FuelName[0] + ':' + str(
                    idp_Label.FuelMF[group_index, point_index, 0])
                if idp_Label.FuelNum > 1:
                    for species_index_minus_1 in range(idp_Label.FuelNum - 1):
                        configuration_text = configuration_text + ', ' + idp_Label.FuelName[
                            species_index_minus_1 + 1] + ':' + str(
                            idp_Label.FuelMF[group_index, point_index, species_index_minus_1 + 1])
                gas.TPX = idp_Label.T5[group_index, point_index], idp_Label.p5[
                    group_index, point_index], configuration_text
                reactor = ct.IdealGasReactor(contents=gas, name="ST")
                env = ct.Reservoir(ct.Solution('air.xml'))
                wall = ct.Wall(reactor, env, A=1.0, velocity=0)
                reactor_net = ct.ReactorNet([reactor])

                for i in range(n_reactions):
                    reactor.add_sensitivity_reaction(i)

                reaction_sensitivities_wrt_OH_record = []
                for i in range(n_reactions):
                    reaction_sensitivities_wrt_OH_record.append([])

                while reactor_net.time < idp_Label.IDTrun[group_index, point_index]:
                    try:
                        reactor_net.step()
                    except CanteraError:
                        print(
                            "It is a CanteraError during combustion simulation for sensitivity analysis! This computation ends and potentially gives incomplete sensitivity curve.")
                        break
                    except RuntimeError:
                        print(
                            "It is a RuntimeError during combustion simulation for sensitivity analysis! This computation ends and potentially gives incomplete sensitivity curve.")
                        break
                    for i in range(n_reactions):
                        s = reactor_net.sensitivity('OH', i)  # sensitivity of OH to reaction 2
                        reaction_sensitivities_wrt_OH_record[i].append(s)

                reaction_sensitivities_wrt_OH = [np.max(np.abs(np.array(s))) for s in
                                                 reaction_sensitivities_wrt_OH_record]
                average_sensitivities += np.array(reaction_sensitivities_wrt_OH)

    average_sensitivities = average_sensitivities / n_cases

    for i in range(n_reactions):
        print(f"Average sensitivity of reaction{i}: {average_sensitivities[i]}")

    return average_sensitivities


def binary_permutation(N, length):
    """Returns a matrix, each row is has exactly N 1s and length length.
    """
    # return a list of lists of length 'length' that contains N 1s
    assert N <= length, "Please set N <= length!"
    if N == 0:
        return [[0 for _ in range(length)]]
    if N == length:
        return [[1 for _ in range(N)]]

    results = []
    # set first bit to 0
    sub_problem_0_result = binary_permutation(N, length - 1)
    for r in sub_problem_0_result:
        new_result = [0]
        new_result.extend(r)
        results.append(new_result)

    # set first bit to 1
    sub_problem_1_result = binary_permutation(N - 1, length - 1)
    for r in sub_problem_1_result:
        new_result = [1]
        new_result.extend(r)
        results.append(new_result)

    return results


def equation_equal(e1, e2):
    """Judges whether two reaction equations e1 and e2 are equal.
    """
    # "=>" or "<=>"
    e1s = e1.split(" <=> ")
    e2s = e2.split(" <=> ")
    if len(e1s) == 1 or len(e2s) == 1:
        return e1 == e2
    elif (e1s[0] == e2s[0] and e1s[1] == e2s[1]) or (e1s[0] == e2s[1] and e1s[1] == e2s[0]):
        return True
    else:
        return False
