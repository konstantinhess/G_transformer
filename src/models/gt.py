from ray import ray_constants
from torch_ema import ExponentialMovingAverage

from src.models.utils_transformer import AbsolutePositionalEncoding, RelativePositionalEncoding
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns

from src.models.utils_transformer import TransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import OutcomeHead
from src.models.time_varying_model import TimeVaryingCausalModel


logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb

########################################################################################################################


class GT(TimeVaryingCausalModel):
    """
    Pytorch-Lightning implementation of G-Transformer (CT)
    """

    model_type = 'gt'  # multi-input model
    possible_model_types = {'gt'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if projection_horizon is not None:
            self.projection_horizon = projection_horizon
        elif args.model.gt.projection_horizon is not None:
            self.projection_horizon = args.model.gt.projection_horizon
        elif self.dataset_collection is not None:
            self.projection_horizon = args.dataset.projection_horizon
        else:
            raise MissingMandatoryValue()

        self.treatment_sequence = torch.tensor(args.dataset.treatment_sequence)[: self.projection_horizon+1, :]
        if self.dataset_collection is not None:
            self.max_projection = args.dataset.projection_horizon
        else:
            self.max_projection = self.projection_horizon

        assert self.projection_horizon <= self.max_projection

        # Used in hparam tuning
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args)
        self.tuning_criterion = 'rmse'
        self.save_hyperparameters(args)


    def _init_specific(self, args: DictConfig):
        """
        Initialization of specific sub-network
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            sub_args = args.model.gt
            self.max_seq_length = sub_args.max_seq_length
            self.hr_size = sub_args.hr_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate

            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads

            if self.seq_hidden_units is None or self.hr_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            # Init of positional encodings
            self.self_positional_encoding = self.self_positional_encoding_k = self.self_positional_encoding_v = None
            if sub_args.self_positional_encoding.absolute:
                self.self_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.self_positional_encoding.trainable)
            else:
                # Relative positional encoding is shared across heads
                self.self_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)
                self.self_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)

            self.cross_positional_encoding = self.cross_positional_encoding_k = self.cross_positional_encoding_v = None
            if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
                self.cross_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.cross_positional_encoding.trainable)
            elif 'cross_positional_encoding' in sub_args:
                # Relative positional encoding is shared across heads
                self.cross_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)
                self.cross_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)

            self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
            self.vitals_input_transformation = \
                nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
            self.vitals_input_transformation = nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
            self.outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
            self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

            self.n_inputs = 3 if self.has_vitals else 2  # prev_outcomes and prev_treatments

            self.transformer_blocks = nn.ModuleList(
                [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                                      self.dropout_rate,
                                      self.dropout_rate if sub_args.attn_dropout else 0.0,
                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                      self_positional_encoding_v=self.self_positional_encoding_v,
                                      n_inputs=self.n_inputs,
                                      disable_cross_attention=sub_args.disable_cross_attention,
                                      isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])
            self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
            self.output_dropout = nn.Dropout(self.dropout_rate)

            # G-computation heads: nested expectations
            self.G_comp_heads = nn.ModuleList(
                [OutcomeHead(self.seq_hidden_units, self.hr_size,
                             self.fc_hidden_units, self.dim_treatments,
                             self.dim_outcome) for _ in range(self.projection_horizon+1)])

            # self.last_layer_norm = LayerNorm(self.seq_hidden_units)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")


    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        for block in self.transformer_blocks:

            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        hr = nn.ELU()(self.hr_output_transformation(output))
        return hr


    def forward(self, batch):

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries'].clone()

        batch_size = prev_treatments.size(0)
        time_dim = prev_treatments.size(1)

        if self.training:

            # 1) train all hidden states on factual data
            if self.projection_horizon == 0:
                hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
                pred_factuals = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
                pseudo_outcomes = pred_pseudos = None

                return pred_factuals, pred_pseudos, pseudo_outcomes, active_entries


            else:
                # 2) G-computation formula: iterate over all time steps
                # 2a) Initialize
                pseudo_outcomes_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1,
                                                         self.projection_horizon+1, self.dim_outcome), device=self.device)
                pred_pseudos_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1,
                                                      self.projection_horizon+1, self.dim_outcome), device=self.device)
                active_entries_all_steps = torch.zeros((batch_size, time_dim-self.projection_horizon-1, 1), device=self.device)

                for t in range(1, time_dim-self.projection_horizon):
                    current_active_entries = batch['active_entries'].clone()
                    current_active_entries[:, int(t + self.projection_horizon):] = 0.0
                    active_entries_all_steps[:, t-1,:] = current_active_entries[:, t+self.projection_horizon-1,:]

                    # 2b) Generate pseudo outcomes
                    with torch.no_grad():
                        indexes_cf = (torch.arange(0, time_dim, device=self.device) >= t-1)*(
                                torch.arange(0, time_dim, device=self.device) < t+self.projection_horizon)
                        curr_treatments_cf = curr_treatments.clone()
                        curr_treatments_cf[:,indexes_cf,:] = self.treatment_sequence.to(self.device)
                        prev_treatments_cf = torch.concat((prev_treatments[:, :1, :], curr_treatments_cf[:, :-1, :]), dim=1)

                        hr_cf = self.build_hr(prev_treatments_cf, vitals, prev_outputs, static_features, current_active_entries)
                        pseudo_outcomes = torch.zeros((batch_size, self.projection_horizon+1, self.dim_outcome), device=self.device)

                        for i in range(self.projection_horizon, 0, -1):
                            pseudo_outcome = self.G_comp_heads[i].build_outcome(hr_cf, curr_treatments_cf)[:, t+i-1, :]
                            pseudo_outcomes[:, i-1, :] = pseudo_outcome
                        pseudo_outcomes[:, -1, :] = batch['outputs'][:, t+self.projection_horizon-1, :]
                        # Store pseudo outcomes
                        pseudo_outcomes_all_steps[:, t-1, :, :] = pseudo_outcomes

                    # 2c) Predict pseudo outcomes with G-computation heads
                    hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, current_active_entries)
                    pred_pseudos = torch.zeros((batch_size, self.projection_horizon + 1, self.dim_outcome), device=self.device)
                    for i in range(self.projection_horizon, -1, -1):
                        pred_pseudo = self.G_comp_heads[i].build_outcome(hr, curr_treatments)[:, t+i-1, :]
                        pred_pseudos[:, i, :] = pred_pseudo
                    # Store predicted pseudo outcomes
                    pred_pseudos_all_steps[:, t-1, :, :] = pred_pseudos

                return None, pred_pseudos_all_steps, pseudo_outcomes_all_steps, active_entries_all_steps

        # 3) Prediction: only use the first head ("=outermost expectation")
        else:
            fixed_split = batch['sequence_lengths'] - self.max_projection if self.projection_horizon > 0 else batch['sequence_lengths']
            for i in range(len(active_entries)):
                active_entries[i, int(fixed_split[i] + self.projection_horizon):] = 0.0

            hr = self.build_hr(prev_treatments, vitals, prev_outputs, static_features, active_entries)
            if self.projection_horizon > 0:
                pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)
                index_pred = (torch.arange(0, time_dim, device=self.device) == fixed_split[..., None] - 1)
                pred_outcomes = pred_outcomes[index_pred]
            else:
                pred_outcomes = self.G_comp_heads[0].build_outcome(hr, curr_treatments)

            return pred_outcomes, hr

    def training_step(self, batch, batch_ind, optimizer_idx=None):

        for par in self.parameters():
            par.requires_grad = True

        pred_factuals, pred_pseudos, pseudo_outcomes, active_entries_all_steps = self(batch)

        if self.projection_horizon > 0:

            active_entries_all_steps = active_entries_all_steps.unsqueeze(-2)
            mse_gcomp = F.mse_loss(pred_pseudos, pseudo_outcomes, reduction='none')
            mse_gcomp = (mse_gcomp * active_entries_all_steps).sum(dim=(0,1)) / (active_entries_all_steps.sum(dim=(0,1)) * self.dim_outcome)

            for i in range(mse_gcomp.shape[0]):
                self.log(f'{self.model_type}_mse_'+str(i), mse_gcomp[i].mean(),
                         on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

            loss = mse_gcomp.mean()

        else:
            mse_factual = F.mse_loss(pred_factuals, batch['outputs'], reduction='none')
            mse_factual = (mse_factual * batch['active_entries']).sum() / (batch['active_entries'].sum() * self.dim_outcome)
            loss = mse_factual

        self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

        return loss


    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        outcome_pred, hr = self(batch)
        return outcome_pred.cpu(), hr.cpu()


    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))] # call predict_step(...), which returns predictions and hr
        return outcome_pred.numpy()



    def get_normalised_n_step_rmses(self, dataset: Dataset):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')

        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        # Only evaluate RMSE on final outcome
        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs'][:, (self.projection_horizon-1)]) ** 2)

        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs'][:, (self.projection_horizon-1)]) ** 2)

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])
        mse = mse[not_nan]

        mse = mse.mean() # mean across batch
        rmse_normalised = np.sqrt(mse) / dataset.norm_const

        if percentage:
            rmse_normalised *= 100.0

        return rmse_normalised


    def configure_optimizers(self):
        # one optimizer
        optimizer = self._get_optimizer(list(self.named_parameters()))

        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers(optimizer)

        return optimizer

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        """
        Vizualizes attention scores
        :param dataset: dataset
        :param index: index of an instance
        :param artifacts_path: Path for saving
        """
        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += ['cross_attention_vo', 'cross_attention_ov', 'cross_attention_vt', 'cross_attention_tv',
                         'self_attention_v']
        self._visualize(fig_keys, dataset, index, artifacts_path)


    # for ray tuning
    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.num_heads = new_args['num_heads']
        sub_args.projection_horizon = 0

        if 'seq_hidden_units' in new_args:  # Only relevant for encoder: seq_hidden_units should be divisible by num_heads
            # seq_hidden_units should even number - required for fixed positional encoding
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
            comon_multiplier = np.lcm.reduce([sub_args.num_heads, 2]).item()
            if sub_args.seq_hidden_units % comon_multiplier != 0:
                sub_args.seq_hidden_units = sub_args.seq_hidden_units + \
                                            (comon_multiplier - sub_args.seq_hidden_units % comon_multiplier)
            print(f'Factual seq_hidden_units of {model_type}: {sub_args.seq_hidden_units}.')

        sub_args.hr_size = int(input_size * new_args['hr_size'])

        sub_args.fc_hidden_units = int(sub_args.hr_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']


