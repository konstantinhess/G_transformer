import pandas as pd
from pandas.core.algorithms import isin
import numpy as np
import torch
from copy import deepcopy
import logging

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.data.mimic_iii.semi_synthetic_dataset import MIMIC3SyntheticDataset
from src.data.dataset_collection import SyntheticDatasetCollection
from src.data.ar_sim.ar_simulation import simulate_data

logger = logging.getLogger(__name__)


class SyntheticARDataset(MIMIC3SyntheticDataset):
    def __init__(self, X: np.array, A: np.array, Y: np.array, active_entries: np.array, patient_types: np.array,
                 subset_name: str):
        active_entries = active_entries[:, :, None]
        self.subset_name = subset_name

        user_sizes = np.squeeze(active_entries.sum(1))
        treatments = A[:, :, None].astype(float)
        self.vitals_unscaled = np.swapaxes(X, 1, 2)
        static_features = patient_types
        self.outcomes_unscaled = Y[:, :, None]

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals_unscaled': self.vitals_unscaled[:, 1:, :],
            'next_vitals_unscaled': self.vitals_unscaled[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'unscaled_outputs': self.outcomes_unscaled[:, 1:, :],
            'prev_unscaled_outputs': self.outcomes_unscaled[:, :-1, :],
        }

        self.processed = False  # Need for normalisation of newly generated outcomes
        self.processed_sequential = False
        self.processed_autoregressive = False

        self.norm_const = 1.0

    def get_scaling_params(self):
        logger.info('Performing normalisation.')
        scaling_params = {
            'output_means': self.outcomes_unscaled.mean(),
            'output_stds': self.outcomes_unscaled.std(),
            'vitals_means': self.vitals_unscaled.mean((0, 1)),
            'vitals_stds': self.vitals_unscaled.mean((0, 1))
        }
        return scaling_params

    def process_data(self, scaling_params):
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')

            self.data['outputs'] = \
                (self.data['unscaled_outputs'] - scaling_params['output_means']) / scaling_params['output_stds']
            self.data['prev_outputs'] = \
                (self.data['prev_unscaled_outputs'] - scaling_params['output_means']) / scaling_params['output_stds']

            self.data['vitals'] = \
                (self.data['vitals_unscaled'] - scaling_params['vitals_means']) / scaling_params['vitals_stds']
            self.data['next_vitals'] = \
                (self.data['next_vitals_unscaled'] - scaling_params['vitals_means']) / scaling_params['vitals_stds']

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            self.scaling_params = scaling_params
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')

        return self.data


class SyntheticARDatasetCollection(SyntheticDatasetCollection):
    def __init__(self, num_patients: dict, max_seq_length: int, p: int, lag: int, seed: int, projection_horizon: int,
                 gamma_X: float, gamma_Y: float, cf_seq_mode: str, n_treatments_seq: int, treatment_sequence: list,
                 **kwargs):
        """
        Args:
            :param num_patients: Number of patients
            :max_seq_length: length of time series
            :p: Number of exogenous covariates
            :lag: Lag order
            :seed: Seed for reproducibility
            :projection_horizon: Projection horizon for multistep ahead prediction
            :gamma_X: confounding for X_{t} -> A_{t}
            :gamma_Y: effect of X_{t} and A_{t} on Y_{t}
            :cf_seq_mode: mode for counterfactual treatment sequence
            :n_treatments_seq: number of random trajectories for multistep ahead prediction for 'random_trajectories'
            :treatment_sequence: treatment sequence for 'fixed_treatment_sequence'
        """

        super(SyntheticARDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)

        X, A, Y, active_entries, patient_types = simulate_data(n=num_patients['train'], T=max_seq_length, p=p, h=lag,
                                                               tau=projection_horizon,
                                                               gamma_X=gamma_X, gamma_Y=gamma_Y,
                                                               num_traj=n_treatments_seq,
                                                               data_seed=seed, coeff_seed=seed)
        self.train_f = SyntheticARDataset(X, A, Y, active_entries, patient_types, 'train')
        import matplotlib.pyplot as plt
        plt.plot(Y[0, :])
        plt.show()
        X, A, Y, active_entries, patient_types = simulate_data(n=num_patients['val'], T=max_seq_length, p=p, h=lag,
                                                               tau=projection_horizon,
                                                               gamma_X=gamma_X, gamma_Y=gamma_Y,
                                                               num_traj=n_treatments_seq,
                                                               data_seed=2 * seed, coeff_seed=seed)
        self.val_f = SyntheticARDataset(X, A, Y, active_entries, patient_types, 'val')

        X, A, Y, active_entries, patient_types = simulate_data(n=num_patients['test'], T=max_seq_length, p=p, h=lag,
                                                               tau=projection_horizon,
                                                               gamma_X=gamma_X, gamma_Y=gamma_Y,
                                                               num_traj=n_treatments_seq,
                                                               mode='counterfactual_one_step', data_seed=3 * seed,
                                                               coeff_seed=seed)
        self.test_cf_one_step = SyntheticARDataset(X, A, Y, active_entries, patient_types, 'test')

        X, A, Y, active_entries, patient_types = simulate_data(n=num_patients['test'], T=max_seq_length, p=p, h=lag,
                                                               tau=projection_horizon,
                                                               gamma_X=gamma_X, gamma_Y=gamma_Y,
                                                               num_traj=n_treatments_seq,
                                                               mode='counterfactual_treatment_seq', data_seed=4 * seed,
                                                               coeff_seed=seed, cf_treatment_mode=cf_seq_mode,
                                                               treatment_sequence=treatment_sequence)
        self.test_cf_treatment_seq = SyntheticARDataset(X, A, Y, active_entries, patient_types, 'test')

        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = True
        self.train_scaling_params = self.train_f.get_scaling_params()