# masd_final_trisoup_unionmask_clean_100run

Below, each `*_95CI` field reports the 95% confidence interval of the corresponding `MAE reduction (K)`.

1. This 100-run reconfirmation was completed under the fixed scientific mainline and the frozen final reporting configuration. No MSPCE/RCMF/MASD definition, model structure, loss definition, split rule, or new training protocol was introduced. This reconfirmation keeps the MSPCE/RCMF/MASD mainline frozen while using validation-only trisoup candidate selection. The external holdout is excluded from checkpoint and soup selection and is used only for final reporting.
2. On the main test set, the strongest baseline reached MAE 24.6678 K, RMSE 36.7942 K, Pearson 0.9454; the final reported configuration reached MAE 23.9814 K, RMSE 36.2603 K, Pearson 0.9466.
3. Relative to the strongest baseline, the final reported configuration achieved MAE reduction of 0.6864 K on the main test set, 4.2242 K on the hard subgroup, and 0.4039 K on the external holdout.
4. Stable improvement across the main test set, hard subgroup, and external holdout is maintained under the 100-run review. The corresponding MAE-reduction 95% CIs are [0.6121, 0.7607] K, [3.5340, 4.9145] K, and [0.2305, 0.5773] K.
5. The weakest chemistry cluster is `other` and it no longer shows improvement with MAE reduction -0.7327 K.
6. After 100 runs, the current result does not fully support the statement that the line reaches a more stable SCI2 level.
7. The manuscript should use `main_results_table.csv`, `subgroup_results_table.csv`, `cluster_results_table.csv`, and `improvement_table.csv` as the paper-facing tables. It should stop using internal-only field names such as negative `螖MAE`, `CI upper` alone, dashboard-style YES/NO gates, or seed-rate guardrail language as main-text result tables.

STATUS: FAIL
GPU_NAME: NVIDIA GeForce RTX 4070 Laptop GPU
USED_GPU_FOR_TRAINING: True
FINAL_MAINLINE_BASE: main_core_sci2_masd_final
FINAL_REPORTED_CONFIGURATION: masd_final_trisoup_unionmask_clean_100run
NUM_RUNS: 100
PRIMARY_FULLDATA_MAE_BASELINE: 24.6678
PRIMARY_FULLDATA_MAE_FINAL: 23.9814
PRIMARY_FULLDATA_MAE_REDUCTION: 0.6864
PRIMARY_FULLDATA_95CI: [0.6121, 0.7607] K
HARD_SUBGROUP_MAE_BASELINE: 29.3758
HARD_SUBGROUP_MAE_FINAL: 25.1516
HARD_SUBGROUP_MAE_REDUCTION: 4.2242
HARD_SUBGROUP_95CI: [3.5340, 4.9145] K
EXTERNAL_HOLDOUT_MAE_BASELINE: 27.6086
EXTERNAL_HOLDOUT_MAE_FINAL: 27.2047
EXTERNAL_HOLDOUT_MAE_REDUCTION: 0.4039
EXTERNAL_HOLDOUT_95CI: [0.2305, 0.5773] K
WEAKEST_CLUSTER_MAE_REDUCTION: -0.7327
SCI2_STABILITY_LEVEL: NOT_STABLE_ENOUGH
SUMMARY_FILE: D:\111_Molecule Property Prediction\code\tg_clean_v2\outputs\exp\diagnostics\masd_final_trisoup_unionmask_clean_100run\final_summary.md