# CIP NeurIPS figure scripts

These scripts redraw the paper figures from the aggregated `metrics.csv` and
summary files produced by the job-array pipelines.  They keep the existing
method color/marker/line-style convention:

- `q0`: blue circle, solid line
- `TempTune`: orange square, dashed line
- `GroupTemp`: green triangle, dash-dot line
- `CIP-Global`: vermillion diamond, dotted line
- `CIP-Group`: purple pentagon, solid line

All PDFs are saved with Type-3-font-safe Matplotlib settings.

## Simulation

```bash
python draw_sim_figures_neurips.py --outdir sim_neurips_array
```

Main outputs:

```text
sim_neurips_array/figures_neurips/paper_summary_2x2.pdf
sim_neurips_array/figures_neurips/global_tail_curve.pdf
sim_neurips_array/figures_neurips/groupwise_key_miscoverage.pdf
sim_neurips_array/figures_neurips/worst_group_violation.pdf
sim_neurips_array/figures_neurips/calibration_accuracy_frontier.pdf
sim_neurips_array/figures_neurips/ablations/*.pdf
```

If `nonlinear_summary.csv` exists, the script also writes:

```text
sim_neurips_array/figures_neurips/nonlinear/nonlinear_wv_mse_by_posterior.pdf
sim_neurips_array/figures_neurips/nonlinear/nonlinear_tail_curve_grid.pdf
```

To skip nonlinear figures:

```bash
python draw_sim_figures_neurips.py --outdir sim_neurips_array --no_nonlinear
```

## Diamonds

```bash
python draw_diamonds_figures_neurips.py \
  --outdir diamonds_neurips_array \
  --run_label diamonds_blind
```

The script writes the usual per-run 2x2 figure and individual panels under:

```text
diamonds_neurips_array/diamonds_blind/figures_neurips/
```

If `diamonds_summary_all.csv` contains RFF runs, the script also writes:

```text
diamonds_neurips_array/figures_neurips/nonlinear_comparison/diamonds_linear_vs_rff_summary.pdf
diamonds_neurips_array/figures_neurips/nonlinear_comparison/diamonds_linear_vs_rff_tail_curves.pdf
```

To draw figures for the nonlinear RFF run itself:

```bash
python draw_diamonds_figures_neurips.py \
  --outdir diamonds_neurips_array \
  --run_label diamonds_blind_rff256
```

## Bike Sharing

```bash
python draw_bike_figures_neurips.py \
  --outdir bike_neurips_array \
  --run_label bike_with_group_drop1_groupPrior1
```

If `bike_summary_all.csv` contains RFF runs, the script also writes:

```text
bike_neurips_array/figures_neurips/nonlinear_comparison/bike_linear_vs_rff_summary.pdf
bike_neurips_array/figures_neurips/nonlinear_comparison/bike_linear_vs_rff_tail_curves.pdf
```

To draw figures for the nonlinear RFF run itself:

```bash
python draw_bike_figures_neurips.py \
  --outdir bike_neurips_array \
  --run_label bike_with_group_rff256
```

## Optional PNG copies

PDF is the default. To also save PNG copies:

```bash
CIP_SAVE_PNG=1 python draw_bike_figures_neurips.py --outdir bike_neurips_array
```

## Font check

After compiling the paper, check:

```bash
pdffonts paper.pdf | grep "Type 3"
```

The command should return nothing.
