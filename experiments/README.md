# Baselines and ablations

Run `python experiments/run_matrix.py --dry-run` to inspect the study. Each
completed training run must publish the evaluation JSON returned by
`evaluate()`. Report at minimum `mae`, `mae_enforced`, `gen_rollout_24_mae`,
`violation_rate_raw`, `violation_rate_enforced`, `latent_std`, and
`latent_cov_offdiag`.

The matrix compares persistence, GRU-only, the full model, and targeted
ablations for EMA, VICReg, and constraint enforcement. Keep the seed and
dataset manifest fixed across rows. A result is not a clinical benchmark:
the current generator is synthetic.
