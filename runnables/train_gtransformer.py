import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import FilteringMlFlowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for GT (G-Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False) # turn of strict mode
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True) # custom resolver: add interpolation expression for within config files
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True)) # convert config object to yaml-formatted string. Resolve interpolation expressions. Log yaml

    # Initialisation of data
    seed_everything(args.exp.seed) # global seed
    dataset_collection = instantiate(args.dataset, _recursive_=True) # Instantiate dataset dynamically from dataset config
    dataset_collection.process_data_multi() # prepare data set
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    gt_callbacks = []

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}_FINAL'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri, run_name='0') # exclude submodels from logging
        gt_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== 1-step ahead prediction ===========================================
    args.model.gt.projection_horizon = 0
    gtmodel = instantiate(args.model.gt, args, dataset_collection, _recursive_=False)  # initialize g-transformer
    if args.model.gt.tune_hparams:
        gtmodel.finetune(resources_per_trial=args.model.gt.resources_per_trial)

    gtmodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                              callbacks=gt_callbacks, terminate_on_nan=True,
                              gradient_clip_val=args.model.gt.max_grad_norm)
    gtmodel_trainer.fit(gtmodel)

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    gtmodel_trainer.test(gtmodel, test_dataloaders=val_dataloader)
    # gtmodel.visualize(dataset_collection.val_f, index=0, artifacts_path=artifacts_path)
    val_rmse_orig, val_rmse_all = gtmodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = gtmodel.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step,
            one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = gtmodel.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)
    # ============================== multi-step ahead prediction ===========================================

    for t in [1]: #2,3,4,5]: #range(1, args.dataset.projection_horizon+1): #range(2, args.dataset.projection_horizon + 1):
        seed_everything(args.exp.seed)  # global seed -> if training breaks for some reason, start again with same seed
        test_rmses = {}
        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig
        }
        if args.exp.logging:
            mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                                               tracking_uri=args.exp.mlflow_uri, run_name=str(t))
        # ============================== Train ===========================================
        args.model.gt.projection_horizon = t
        gtmodel = instantiate(args.model.gt, args, dataset_collection, _recursive_=False)
        gtmodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                  callbacks=gt_callbacks, terminate_on_nan=True,
                                  gradient_clip_val=args.model.gt.max_grad_norm)
        gtmodel_trainer.fit(gtmodel)

        # ============================== Test ===========================================
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
            test_rmse = gtmodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
            test_rmse = gtmodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
        test_rmses = {f'{t+1}-step': test_rmse}
        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()

