G-Transformer
==============================

G-transformer for conditional average potential outcome estimation over time.

### Setup
Please set up a virtual environment and install the libraries as given in the requirements file.
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow
To start an experiments server, run: 

`mlflow server --port=3333`

Connect via ssh to access the MlFlow UI:

`ssh -N -f -L localhost:3333:localhost:3333 <username>@<server-link>`

Then, one can go to the local browser <http://localhost:3333>.

## Experiments

The main training script `config/config.yaml` is run automatically for all models and datasets.
___
The training `<script>` for each different models specified by:

**CRN**: `runnables/train_enc_dec.py`

**TE-CDE**: `runnables/train_enc_dec.py`

**CT**: `runnables/train_multi.py`

**RMSNs**: `runnables/train_rmsn.py`

**G-Net**: `runnables/train_gnet.py`

**GT**: `runnables/train_gtransformer.py`

___

The `<backbone>` is specified by:

**CRN**: `crn`

**TE-CDE**: `tecde`

**CT**: `ct`

**RMSNs**: `rmsn`

**G-Net**: `gnet`

**GT**: `gt`
___

The `<hyperparameter>` configuration for each model is specified by:

**CRN**: `backbone/crn_hparams='HPARAMS'`

**TE-CDE**: `backbone/tecde_hparams='HPARAMS'`

**CT**: `backbone/ct_hparams='HPARAMS'`

**RMSNs**: `backbone/rmsn_hparams='HPARAMS'`

**G-Net**: `backbone/gnet_hparams='HPARAMS'`

**GT**: `backbone/gt_hparams='HPARAMS'`

`HPARAMS` is either one of:
`cancer_sim_hparams_grid.yaml` / `cancer_sim_tuned.yaml`,

for tuning the hyperparameters / reproducing our results on tuned hyperparameters for synthetic data, or:

`mimi3_synthetic_hparams_grid.yaml`, `mimic3_synthetic_tuned.yaml`

for tuning the hyperparameters / reproducing our results on tuned hyperparameters for semi-synthetic data.

___

The `<dataset>` is specified by:

**Synthetic**: `cancer_sim`

**Semi-synthetic**: `mimic3_synthetic`

Please use the following commands to run the experiments. 
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 <script> +dataset=<dataset> +backbone=<backbone> +<hyperparameter> exp.seed=101 exp.logging=True 
```

## Example usage
To run our GT with optimized hyperparameters on synthetic data on random seeds 101--105, use the command:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_gtransformer.py --multirun +dataset=cancer_sim +backbone=gt +backbone/gt_hparams='cancer_sim_tuned' exp.seed=101,102,103,104,105
```

Note that, before running semi-synthetic experiments, the MIMIC-III-extract dataset ([all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract)) needs to be placed in `data/processed/`.

