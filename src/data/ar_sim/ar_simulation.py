import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from src.data.mimic_iii.utils import RandomFourierFeaturesFunction

logger = logging.getLogger(__name__)


def simulate_data(n, T, p, h, tau, gamma_X, gamma_Y, num_traj, noise_X=0.1, noise_Y=0.001, n_patient_types=3,
                  coeff_seed=4242, data_seed=10, mode='factual', cf_treatment_mode='random_trajectories',
                  treatment_sequence=None):
    """
    Autoregressive data simulation
    :param n: Number of patients
    :param T: length of time series
    :param p: Number of exogenous covariates
    :param h: Lag order
    :param tau: Projection horizon
    :param gamma_X: confounding for X_{t} -> A_{t}
    :param gamma_A: confounding for A_{t-1} -> A_{t}
    :param gamma_Y: effect of X_{t} and A_{t} on Y_{t}
    :param num_traj: number of trajectories for tau step ahead pred
    :param noise_X: init noise for X
    :param noise_Y: init noise for Y
    :param mode: factual / counterfactual_treatment_seq / counterfactual_one_step
    :param cf_treatment_mode: random_trajectories / fixed_sequence; only relevant for mode='counterfactual_treatment_seq'
    :param treatment_sequence: fixed treatment sequence; only relevant for mode='counterfactual_treatment_seq'
    :return: X, A, Y, active_entries, patient_types
    """
    logger.info(f'Generating {mode} AR timeseries with {n} observations')

    burn = 1  # Essential for initial state of encoder
    assert tau > 1  # tau + 1 in original notation
    assert h > 0
    if cf_treatment_mode == 'fixed_treatment':
        num_traj = 1

    # Coefficients
    np.random.seed(coeff_seed)
    lambda_coefs = np.random.normal(loc=0, scale=0.5, size=(p, h))
    w_coefs = np.empty(shape=(p, h))
    for i in range(h):
        #w_coefs[:, i] = np.random.normal(loc=0.0, scale=(1 / h), size=p)
        #w_coefs[:, i] = np.random.normal(loc=i/h, scale=(1 / h), size=p)
        w_coefs[:, i] = np.random.normal(loc=h + i / h, scale=(1 / h), size=p)
    w_bar_coefs = np.empty(shape=(n_patient_types, h + 1))  # Lag h + current time-step
    for patient_type in range(n_patient_types):
        for i in range(h + 1):
            # w_bar_coefs[patient_type, i] = np.random.normal(loc=1 - (h - i) / h, scale=1 / h, size=1)
            w_bar_coefs[patient_type, i] = np.random.normal(loc=1 - (h - i) / h, scale=1.0, size=1)

    # treatment_dep = RandomFourierFeaturesFunction(p, 0.01, 20.0)
    # outcome_dep = RandomFourierFeaturesFunction(p, 0.01, 20.0)

    np.random.seed(data_seed)
    # Noise for potential outcomes

    def noise(s, sd):
        return np.random.normal(loc=0, scale=sd, size=s)

    # Initialize Factual data
    X = np.random.normal(loc=0, scale=0.1, size=(n, p, T + h + burn))
    A = np.zeros((n, T + h + burn), dtype=int)
    A_scores = np.zeros((n, T + h + burn))
    Y = np.random.normal(loc=0, scale=0.1, size=(n, T + h + burn))
    patient_types = np.zeros((n, n_patient_types))
    active_entries = np.ones((n, T + h + burn))

    # sampled_noise_X =

    # Initialize One-step Counterfactuals
    if mode == 'counterfactual_one_step':
        X_cf_one = np.empty((n * T * 2, p, T + h + burn))
        A_cf_one = np.empty((n * T * 2, T + h + burn))
        Y_cf_one = np.empty((n * T * 2, T + h + burn))
        patient_types_cf_one = np.empty((n * T * 2, n_patient_types))
        active_entries_cf_one = np.zeros((n * T * 2, T + h + burn))

    # Initialize tau-step-ahead counterfactuals - random trajectories
    if mode == 'counterfactual_treatment_seq':
        X_cf_tau = np.empty((n * (T - tau) * num_traj, p, T + h + burn))
        A_cf_tau = np.empty((n * (T - tau) * num_traj, T + h + burn))
        Y_cf_tau = np.empty((n * (T - tau) * num_traj, T + h + burn))
        patient_types_cf_tau = np.empty((n * (T - tau) * num_traj, n_patient_types))
        active_entries_cf_tau = np.zeros((n * (T - tau) * num_traj, T + h + burn))

    # Sample data until counterfactual prediction timeframe
    for pat in tqdm(range(n)):
        # Sampling type of patient -> Will define the effect of treatment
        patient_type = np.random.randint(0, n_patient_types, (1, ))
        patient_types[pat, patient_type] = 1.0

        # Simulate until time of potential outcome prediction
        for t in range(h, T + h + burn):
            # Covariates
            for j in range(p):
                X[pat, j, t] = np.tanh((1 / h) * np.dot(lambda_coefs[j, :], X[pat, j, (t - h):t]) + \
                               (1 / h) * np.dot(w_coefs[j, :], A[pat, (t - h):t]) + noise(1, noise_X))
            # Treatment
            # A[pat, h - 1] is a constant, so not a hidden confounder!
            # prob = gamma_A * (0.5 - np.mean(A[pat, (h - 1):t])) + gamma_X * np.mean(X[pat, :, t])
            #prob = 0.5 * (0.5 - np.mean(A[pat, (h - 1):t])) + 0.5 * np.mean(X[pat, :, t]) - 0.5
            prob = gamma_Y * Y[pat, t] + gamma_X * np.mean(X[pat, :, t]) - 1.0
            print(prob)
            A_scores[pat, t] = prob
            if expit(prob) > np.random.random():
                A[pat, t] = 1
            # Y[pat, t] = gamma_X * (A[pat, t] * np.mean(X[pat, :, (t - h)]) * Y[pat, t - 1] +
            #                        (1.0 - A[pat, t]) * np.mean(X[pat, :, t]) * Y[pat, t - 1]) \
            #     + gamma_Y * np.dot(w_bar_coefs[patient_type, :], A[pat, (t - h):(t + 1)]) + noise(1, noise_Y)
            Y[pat, t] = (gamma_X * (A[pat, t] * np.mean(X[pat, :, (t - h):t]) )) \
                        + (gamma_Y * np.dot(w_bar_coefs[patient_type, :], A[pat, (t - h):(t + 1)]) + noise(1, noise_Y))

        # ============ One-step Counterfactuals ============
        if mode == 'counterfactual_one_step':
            X_cf_pat = np.repeat(X[pat:pat + 1, :], T * 2, 0)
            A_cf_pat = np.repeat(A[pat:pat + 1, :], T * 2, 0)
            Y_cf_pat = np.repeat(Y[pat:pat + 1, :], T * 2, 0)
            patient_types_pat = np.repeat(patient_types[pat:pat + 1, :], T * 2, 0)
            active_entries_cf_pat = np.zeros((T * 2, T + h + burn))
            active_entries_cf_pat[:, :(h + burn)] = 1.0

            i = 0
            for t in range(h + burn, T + h + burn):

                for treatment_option in [0, 1]:  # A = [0, 1]
                    A_cf_pat[i, t] = treatment_option
                    active_entries_cf_pat[i, :(t + 1)] = 1.0

                    # Covariates
                    for j in range(p):
                        X_cf_pat[i, j, t] = np.tanh((1 / h) * np.dot(lambda_coefs[j, :], X_cf_pat[i, j, (t - h):t]) + \
                                            (1 / h) * np.dot(w_coefs[j, :], A_cf_pat[i, (t - h):t]) + noise(1, noise_X))
                    # Outcomes
                    # Y_cf_pat[i, t] = gamma_X * (A_cf_pat[i, t] * np.mean(X_cf_pat[i, :, (t - h)]) * Y_cf_pat[i, t - 1] +
                    #                             (1.0 - A_cf_pat[i, t]) * np.mean(X_cf_pat[i, :, t]) * Y_cf_pat[i, t - 1]) \
                    #     + gamma_Y * np.dot(w_bar_coefs[patient_type, :], A_cf_pat[i, (t - h):(t + 1)]) + noise(1, noise_Y)
                    Y_cf_pat[i, t] = (gamma_X * (A_cf_pat[i, t] * np.mean(X_cf_pat[i, :, (t - h):t])   )) \
                                     + (gamma_Y * np.dot(w_bar_coefs[patient_type, :],
                                                        A_cf_pat[i, (t - h):(t + 1)]) + noise(1, noise_Y))

                    i += 1

            # Zeroing non-active values
            Y_cf_pat[active_entries_cf_pat == 0.0] = 0.0
            X_cf_pat[np.repeat(active_entries_cf_pat[:, None, :], p, 1) == 0.0] = 0.0
            A_cf_pat[active_entries_cf_pat == 0.0] = 0

            Y_cf_one[pat * T * 2: (pat + 1) * T * 2, :] = Y_cf_pat
            X_cf_one[pat * T * 2: (pat + 1) * T * 2, :, :] = X_cf_pat
            A_cf_one[pat * T * 2: (pat + 1) * T * 2, :] = A_cf_pat
            patient_types_cf_one[pat * T * 2: (pat + 1) * T * 2, :] = patient_types_pat
            active_entries_cf_one[pat * T * 2: (pat + 1) * T * 2, :] = active_entries_cf_pat

        # ============ tau-step-ahead counterfactuals - random trajectories ============
        if mode == 'counterfactual_treatment_seq':
            X_cf_pat = np.repeat(X[pat:pat + 1, :], (T - tau) * num_traj, 0)
            A_cf_pat = np.repeat(A[pat:pat + 1, :], (T - tau) * num_traj, 0)
            Y_cf_pat = np.repeat(Y[pat:pat + 1, :], (T - tau) * num_traj, 0)
            patient_types_pat = np.repeat(patient_types[pat:pat + 1, :], (T - tau) * num_traj, 0)
            active_entries_cf_pat = np.zeros(((T - tau) * num_traj, T + h + burn))
            active_entries_cf_pat[:, :(h + burn)] = 1.0
            i = 0
            for t in range(h + burn + 1, T + h + burn - tau + 1):
                if cf_treatment_mode == 'random_trajectories':
                    treatment_sequence = np.random.randint(0, 2, size=(num_traj, tau))
                elif cf_treatment_mode == 'fixed_treatment':
                    treatment_sequence = np.reshape(treatment_sequence, newshape=(num_traj, tau))
                for sequence in treatment_sequence:
                    active_entries_cf_pat[i, :(t + tau)] = 1.0
                    for tt in range(tau):
                        A_cf_pat[i, t + tt] = sequence[tt]

                        # Covariates
                        for j in range(p):
                            X_cf_pat[i, j, t + tt] = np.tanh((1 / h) * np.dot(lambda_coefs[j, :],
                                                                      X_cf_pat[i, j, (t - h + tt):(t + tt)]) + \
                                                     (1 / h) * np.dot(w_coefs[j, :], A_cf_pat[i, (t - h + tt):(t + tt)]) \
                                + noise(1, noise_X))

                        # Outcomes
                        # Y_cf_pat[i, t + tt] = gamma_X * (A_cf_pat[i, t + tt] *
                        #                                  np.mean(X_cf_pat[i, :, (t + tt - h)]) * Y_cf_pat[i, t + tt - 1] +
                        #                                  (1.0 - A_cf_pat[i, t + tt]) * np.mean(X_cf_pat[i, :, (t + tt)]) *
                        #                                  Y_cf_pat[i, t + tt - 1])  \
                        #     + gamma_Y * np.dot(w_bar_coefs[patient_type, :], A_cf_pat[i, (t - h + tt):(t + 1 + tt)]) + \
                        #     noise(1, noise_Y)
                        Y_cf_pat[i, t + tt] = (gamma_X * (A_cf_pat[i, t + tt] *
                                                         np.mean(X_cf_pat[i, :, (t + tt - h):(t + tt)]) )) \
                                              + (gamma_Y * np.dot(w_bar_coefs[patient_type, :],
                                                                 A_cf_pat[i, (t - h + tt):(t + 1 + tt)])+ \
                                              noise(1, noise_Y))

                    i += 1

            # Zeroing non-active values
            Y_cf_pat[active_entries_cf_pat == 0.0] = 0.0
            X_cf_pat[np.repeat(active_entries_cf_pat[:, None, :], p, 1) == 0.0] = 0.0
            A_cf_pat[active_entries_cf_pat == 0.0] = 0

            Y_cf_tau[pat * (T - tau) * num_traj: (pat + 1) * (T - tau) * num_traj, :] = Y_cf_pat
            X_cf_tau[pat * (T - tau) * num_traj: (pat + 1) * (T - tau) * num_traj, :, :] = X_cf_pat
            A_cf_tau[pat * (T - tau) * num_traj: (pat + 1) * (T - tau) * num_traj, :] = A_cf_pat
            patient_types_cf_tau[pat * (T - tau) * num_traj: (pat + 1) * (T - tau) * num_traj, :] = patient_types_pat
            active_entries_cf_tau[pat * (T - tau) * num_traj: (pat + 1) * (T - tau) * num_traj, :] = active_entries_cf_pat

    # Cutting burn time and lag
    start_from = h + burn - 1

    if mode == 'factual':
        for i in range(active_entries.shape[0]):
            seq_len = np.random.randint(start_from + 1, high=active_entries.shape[1])
            active_entries[i, seq_len:] = 0.0
        X, A, Y, active_entries = X[:, :, start_from:], A[:, start_from:], Y[:, start_from:], active_entries[:, start_from:]
        return X, A, Y, active_entries, patient_types

    elif mode == 'counterfactual_one_step':
        X_cf_one, A_cf_one, Y_cf_one, active_entries_cf_one = X_cf_one[:, :, start_from:], A_cf_one[:, start_from:], \
            Y_cf_one[:, start_from:], active_entries_cf_one[:, start_from:]
        return X_cf_one, A_cf_one, Y_cf_one, active_entries_cf_one, patient_types_cf_one

    elif mode == 'counterfactual_treatment_seq':
        X_cf_tau, A_cf_tau, Y_cf_tau, active_entries_cf_tau = X_cf_tau[:, :, start_from:], A_cf_tau[:, start_from:], \
            Y_cf_tau[:, start_from:], active_entries_cf_tau[:, start_from:]
        return X_cf_tau, A_cf_tau, Y_cf_tau, active_entries_cf_tau, patient_types_cf_tau
    else:
        raise NotImplementedError()


# if __name__ == "__main__":
#     simulate_data(n=10000, T=40, p=5, h=2, tau=3, gamma_X=0.15, gamma_A=10., gamma_Y=0.5, num_traj=10)
#     simulate_data(n=1000, T=40, p=5, h=2, tau=3, gamma_X=0.15, gamma_A=10., gamma_Y=0.5, num_traj=10,
#                   mode='counterfactual_one_step')
#     simulate_data(n=1000, T=40, p=5, h=2, tau=3, gamma_X=0.15, gamma_A=10., gamma_Y=0.5, num_traj=10,
#                   mode='counterfactual_treatment_seq')