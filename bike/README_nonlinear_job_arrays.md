# Nonlinear approximate-posterior job-array update

This directory contains v4 job-array scripts for adding a nonlinear approximate posterior to the synthetic simulation, Diamonds, and Bike Sharing experiments.

The nonlinear posterior is a random feature last layer Gibbs posterior.  The raw covariates are mapped to random Fourier features built only from the fitting split, and the Gibbs posterior is Gaussian over the last layer weights.  The split logic is unchanged. Thresholds are chosen on `Dthr`, projection constraints are computed on `Dproj`, and all reported metrics are evaluated on `Dtest`.

## New metrics

The per-replicate `metrics.csv` files now include:

- `posterior_model` and `posterior_model_code` (`0=linear`, `1=rff`, `2=relu`, `3=tanh`);
- `rf_dim`, `rf_scale`, and `rf_include_linear`;
- `feature_dim_input` and `feature_dim_posterior`;
- `time_feature_map`;
- for the synthetic nonlinear DGP, `nonlinear_signal`.

The usual calibration and accuracy metrics are unchanged: global key-threshold miscoverage, worst-group multi-threshold violation, predictive MSE, GroupTemp diagnostics, KL/ESS diagnostics, timing breakdowns, and optional tail curves.

## Synthetic nonlinear simulation

Run only the nonlinear simulation suite:

```bash
SUITE=nonlinear MAX_CONCURRENT=50 bash submit_sim_neurips_pipeline_v4_nonlinear.sh
```

Quick smoke test:

```bash
SUITE=nonlinear FAST=1 R_NONLINEAR=1 S_POOL=500 N_PARTICLES=100 S_TUNE=100 S_GROUP_TUNE=50 RF_DIM=32 \
  MAX_CONCURRENT=4 bash submit_sim_neurips_pipeline_v4_nonlinear.sh
```

The nonlinear suite runs a nonlinear response DGP with a misspecified linear posterior and one or more random-feature posteriors.  The default random-feature dimension is `RF_DIM=128`; if `FAST=0` and `RF_DIM` is not 256, an additional `rff256` setting is included.

Outputs include:

```text
sim_neurips_array_nonlinear/nonlinear_summary.csv
sim_neurips_array_nonlinear/<setting>/metrics.csv
sim_neurips_array_nonlinear/figures/
```

## Diamonds nonlinear random-feature runs

Run the default Diamonds settings plus nonlinear random-feature settings:

```bash
INCLUDE_NONLINEAR=1 MAX_CONCURRENT=20 bash submit_diamonds_neurips_pipeline_v4_nonlinear.sh
```

By default this runs 4 settings over 20 splits:

```text
diamonds_blind
diamonds_with_cut_drop1_cutprior1
diamonds_blind_rff256
diamonds_with_cut_rff256
```

You can change the random feature dimension:

```bash
INCLUDE_NONLINEAR=1 RF_DIM=128 bash submit_diamonds_neurips_pipeline_v4_nonlinear.sh
```

## Bike Sharing nonlinear random feature runs

Run the default Bike settings plus nonlinear random feature settings:

```bash
INCLUDE_NONLINEAR=1 NO_K_ABLATION=1 MAX_CONCURRENT=20 bash submit_bike_neurips_pipeline_v4_nonlinear.sh
```

By default, this runs 4 settings over 20 splits:

```text
bike_blind
bike_with_group_drop1_groupPrior1
bike_blind_rff256
bike_with_group_rff256
```

Set `NO_K_ABLATION=0` to also run the K-sensitivity settings.

## Manual count and plan commands

Simulation:

```bash
python run_sim_neurips_array_v4_nonlinear.py --mode count --suite nonlinear --R_nonlinear 20
python run_sim_neurips_array_v4_nonlinear.py --mode plan  --suite nonlinear --R_nonlinear 20
```

Diamonds:

```bash
python run_diamonds_neurips_array_v4_nonlinear.py --mode count --R 20 --include_nonlinear
python run_diamonds_neurips_array_v4_nonlinear.py --mode plan  --R 20 --include_nonlinear
```

Bike:

```bash
python run_bike_neurips_experiment_array_v4_nonlinear.py --mode count --R 20 --include_nonlinear --no_k_ablation
python run_bike_neurips_experiment_array_v4_nonlinear.py --mode plan  --R 20 --include_nonlinear --no_k_ablation
```
