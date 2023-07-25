import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
from scipy import special, stats

# poisson distribution fitting definition - odd/even : odd number events of switching events

def f_odd(tau, n, tR, g0, g1, u0, u1):
    '''Part of the NV photon statistic. Function in the integral from formula S4 in shields 2015 prl spin to charge conversion paper'''
    func = g1 * np.exp((g0 - g1) * tau - g0 * tR) * special.iv(0, 2 * np.sqrt(
        g1 * g0 * tau * (tR - tau))) * stats.poisson.pmf(n, u1 * tau + u0 * (tR - tau))
    return func


def f_even(tau, n, tR, g0, g1, u0, u1):
    '''Part of the NV photon statistic. Function in the integral from formula S5 in shields 2015 prl spin to charge conversion paper'''
    func = float(np.sqrt(g1 * g0 * tau / (tR - tau)) * np.exp((g0 - g1) * tau - g0 * tR) * special.iv(1, 2 * np.sqrt(
        g1 * g0 * tau * (tR - tau))) * stats.poisson.pmf(n, u1 * tau + u0 * (tR - tau)))
    return func


def prob_photon_num(n, tR, g0, g1, u0, u1):
    """
    Part of the NV photon statistic. Integration of f_odd and f_even like
    in the formulas S4/S5 in shields 2015 prl spin to charge conversion paper'''
    :param n:
    :param tR:
    :param g0:
    :param g1:
    :param u0:
    :param u1:
    :return:
    """
    # for NV- g0,g1,u0,u1 stand for g0-,g-0,gamma0,gamma, respectively - in S4
    # for NV0 switch 0 and 1
    # tR : readout time of NV spin,
    # tau : occupying time in charge state

    p_odd, err_odd = sp.integrate.quad(f_odd, 0, tR, args=(n, tR, g0, g1, u0, u1))

    p_even, err_even = sp.integrate.quad(f_even, 0, tR, args=(n, tR, g0, g1, u0, u1))

    p_even += np.exp(-g1 * tR) * stats.poisson.pmf(n,u1 * tR)
    return p_odd + p_even


def prob_total(n, tR, g0, g1, u0, u1):
    '''NV photon statistic as seen in Luke Hacquebards "charge state dynamics" pra paper from 2018 (Formula A2)'''
    return g1 / (g0 + g1) * prob_photon_num(n, tR, g1, g0, u1, u0) + g0 / (g0 + g1) * prob_photon_num(n, tR, g0, g1, u0,
                                                                                                      u1)
def prob_single_poisson_from_prob_total(n, tR, g0, g1, u0, u1):

    return g1 / (g0 + g1)*prob_photon_num(n, tR, g1, g0, u1, u0), g0 / (g0 + g1) *prob_photon_num(n, tR, g0, g1, u0, u1)

def prob_single_poisson_from_prob_total_equal(n, tR, g0, g1, u0, u1):

    return prob_photon_num(n, tR, g1, g0, u1, u0), prob_photon_num(n, tR, g0, g1, u0, u1)

def prob_totalN(n, tR, g0, g1, u0, u1):
    '''prob_total function with the ability to process array input'''
    onebyone = np.vectorize(prob_total)

    return onebyone(n, tR, g0, g1, u0, u1)


def prob_total_ionFidelity(n, tR, g0, g1, u0, u1, FI):
    '''NV photon statistic as seen in Luke Hacquebards "charge state dynamics" pra paper from 2018 (Formula A2)'''
    return (1 - FI) * prob_photon_num(n, tR, g1, g0, u1, u0) + FI * prob_photon_num(n, tR, g0, g1, u0, u1)

def prob_single_total_ionFidelity(n, tR, g0, g1, u0, u1, FI):
    return (1 - FI) * prob_photon_num(n, tR, g1, g0, u1, u0), FI * prob_photon_num(n, tR, g0, g1, u0, u1)

def prob_single_total_ionFidelity_equal(n, tR, g0, g1, u0, u1, FI):
    return  prob_photon_num(n, tR, g1, g0, u1, u0),  prob_photon_num(n, tR, g0, g1, u0, u1)



def prob_totalN_ionFidelity(n, tR, g0, g1, u0, u1, FI):
    '''prob_total function with the ability to process array input'''
    onebyone = np.vectorize(prob_total_ionFidelity)

    return onebyone(n, tR, g0, g1, u0, u1, FI)

def calculate_separate_probability_distributions(n, tR, g0, g1, u0, u1):
    """
    get P(O|+, odd), P(O|+, even), P(O|-, odd), P(O|+, even)
    from the formulas of shields 2015 spin to charge conversion paper'''
    :param n:
    :param tR:
    :param g0:
    :param g1:
    :param u0:
    :param u1:
    :return: p_odd_A_array: P(O|+, odd)
             p_even_A_array: P(O|+, even)
             p_odd_B_array: P(O|-, odd)
             p_even_B_array: P(O|-, even)
    """
    # for NV- g0,g1,u0,u1 stand for g0-,g-0,gamma0,gamma, respectively - in S4
    # tR : readout time of NV spin,
    # tau : occupying time in charge state

    p_odd_A_array = []
    p_even_A_array = []
    p_odd_B_array = []
    p_even_B_array = []
    for i in n:
        # A part can correspond to +
        p_odd_A, err_odd_A = sp.integrate.quad(f_odd, 0, tR, args=(i, tR, g0, g1, u0, u1))
        p_even_A, err_even_A = sp.integrate.quad(f_even, 0, tR, args=(i, tR, g0, g1, u0, u1))
        p_even_A += np.exp(-g1 * tR) * stats.poisson.pmf(i,u1 * tR)
        p_odd_A_array.append(p_odd_A)
        p_even_A_array.append(p_even_A)

        # B part can correspind to -
        p_odd_B, err_odd_B = sp.integrate.quad(f_odd, 0, tR, args=(i, tR, g1, g0, u1, u0))
        p_even_B, err_even_B = sp.integrate.quad(f_even, 0, tR, args=(i, tR, g1, g0, u1, u0))
        p_even_B += np.exp(-g0 * tR) * stats.poisson.pmf(i,u0 * tR)
        p_odd_B_array.append(p_odd_B)
        p_even_B_array.append(p_even_B)

    return p_odd_A_array, p_even_A_array, p_odd_B_array, p_even_B_array

def fit_initial_state_model_no_steady_state(trace, tR, g0, g1, u0, u1, w_1): 
    """
    This function fits P(O) = w_1*P(O|+) + (1-w_1)*P(O|+)
    So it can just be used after tR, g0, g1, u0, u1 are known through a steady state fit. 
    Return: 
        fit_value: w_1 
    """
    hist_y , hist_x = np.histogram(trace,bins=range(int(np.min(trace)),int(np.max(trace))),density=True)

    p_odd_A_array, p_even_A_array, p_odd_B_array, p_even_B_array = calculate_separate_probability_distributions(hist_x, tR, g0, g1, u0, u1)

    y_data = hist_y[hist_y!= 0]

    def fit_error(prmtr): 

        error = []
        function = []
        for n in range(len(y_data)):
            error.append(y_data[n] - (prmtr[0]*(p_even_A_array[n] + p_odd_A_array[n]) + (1-prmtr[0])*(p_odd_B_array[n] + p_even_B_array[n])))

        return error


    prmtr_guess = [w_1]

    pfit = sp.optimize.least_squares(fit_error, prmtr_guess, bounds=([0.], [1])).x


    return  pfit[0]

def prob_double_poisson(k, tR, u0, u1, FI):
    """
    Here we generate a two poisson function probability of two Poissonian distribution with mu = uo and u1 and ask for
    probability to have k.
    :param k:
    :param tR:
    :param u0:
    :param u1:
    :param FI:
    :return:
    """

    mu1 = u1 * tR
    mu0 = u0 * tR
    return FI * stats.poisson.pmf(k,mu1) + (1 - FI) * stats.poisson.pmf(k,mu0)

def prob_single_poission_from_double_result(k, tR, u0, u1, FI):
    mu1 = u1 * tR
    mu0 = u0 * tR
    return FI * stats.poisson.pmf(k,mu1), (1 - FI) * stats.poisson.pmf(k,mu0)


def prob_double_poissonN(n, tR, u0, u1, FI):
    """
    Vectorized version of prob_double_poisson
    :param n: all possible parameters.
    :param tR: integration time
    :param u0: photon level 0
    :param u1: proton level 1
    :param FI: occurence of state 1
    :return: Pmf of various n.
    """
    onebyone = np.vectorize(prob_double_poisson)
    return onebyone(n, tR, u0, u1, FI)


def fitPhotonStat(histdata_x, histdata_y, tR, g0_guess, g1_guess, u0_guess, u1_guess):
    '''fits the probability distribution to the hisogram data set'''

    def fitfunction(n, g0, g1, u0, u1):
        return prob_totalN(n, tR, g0, g1, u0, u1)

    fitmodel = fitfunction
    weights = np.sqrt(histdata_y)
    popt, pcov = sp.optimize.curve_fit(fitmodel, histdata_x, histdata_y, bounds=(0, np.inf),
                                       p0=(g0_guess, g1_guess, u0_guess, u1_guess),
                                       sigma=weights)  # former upper bounds: [20e3,60e3,40e3,600e3]
    # print(fitfunction(1,1,1,1,1))
    return popt, pcov


def fitPhotonStat_ini(histdata_x, histdata_y, tR, g0_guess, g1_guess, u0_guess, u1_guess, FI_guess):
    '''fits the probability distribution to the hisogram data set'''

    def fitfunction(n, g0, g1, u0, u1, FI):
        return prob_totalN_ionFidelity(n, tR, g0, g1, u0, u1, FI)

    fitmodel = fitfunction
    weights = histdata_y
    popt, pcov = sp.optimize.curve_fit(fitmodel, histdata_x, histdata_y, bounds=(0.0, np.inf),
                                       p0=(g0_guess, g1_guess, u0_guess, u1_guess, FI_guess),
                                       sigma=weights)  # former upper bounds: [20e3,60e3,40e3,600e3]
    # print(fitfunction(1,1,1,1,1))
    return popt, pcov


def fit_double_poisson(histdata_x, histdata_y, tR, u0_guess, u1_guess, FI_guess):
    def fitfunction(n, u0, u1, FI):
        return prob_double_poissonN(n, tR, u0, u1, FI)

    fitmodel = fitfunction
    weights = histdata_y
    popt, pcov = sp.optimize.curve_fit(fitmodel, histdata_x, histdata_y, bounds=(0, np.inf),
                                       p0=(u0_guess, u1_guess, FI_guess),
                                       sigma=weights)  # former upper bounds: [20e3,60e3,40e3,600e3]
    return popt, pcov