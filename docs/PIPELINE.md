# Analysis Pipeline

## Workflow

The notebooks are numbered in execution order and exchange restricted derived data through `outputs/working_data/` unless `JMIR_WORKING_DATA_DIR` is configured.

| Step | Notebook | Main purpose | Stable handoff |
|---|---|---|---|
| 1 | `01_build_core_message_preprocessing_clean.ipynb` | Stack cleaned message extracts; normalize subjects; merge governed provider mappings; classify sender, receiver, and message origin | `messages_without_threads.parquet` |
| 2 | `02_threading_diagnostics_sensitivity.ipynb` | Assign conversation threads and compare 15-, 30-, and 60-day gap rules; retain the manuscript threading rule | `messages_with_threads.parquet` |
| 3 | `03_build_visit_encounters_clean.ipynb` | Harmonize clinical, emergency, hospital, and provider encounter extracts | `encounters_unified_clean.parquet` |
| 4 | `04_build_demographics_cci_clean.ipynb` | Construct demographics, age at message, and message-time Charlson Comorbidity Index variables | Three demographics/CCI parquet handoffs |
| 5 | `05_build_exposure_analysis_dataset_clean.ipynb` | Collapse messages to HCW-initiated threads; define outcomes; calculate rolling message and encounter exposures; apply the manuscript analytic period | `analysis_df_all_exposures_with_encounter_covariates.parquet` |
| 6 | `06_analysis_models_clean.ipynb` | Fit primary and sensitivity models and generate Table 3 and Figure S2 | Tables and primary-model exports for notebook 10 |
| 7 | `07_physapp_demographic_cci_interactions_clean.ipynb` | Fit physician/APP subgroup models | Subgroup tables and model exports for notebook 10 |
| 8 | `08_patient_fixed_effects_robustness.ipynb` | Fit exact conditional logistic patient fixed-effects models | Table S7 and conditional sample summaries |
| 9 | `09_generate_table_s9_other_hcw_role_composition.ipynb` | Describe Other HCW-initiated threads by the structured role category of the initiating sender | Table S9 in CSV, HTML, and XLSX formats |

Once the analysis inputs are available, use the prerequisites below. Notebook 09 can run directly from the threaded-message and unified-analysis files.

| Notebook | Prerequisites | Main output |
| --- | --- | --- |
| 06 | 05 | Primary and sensitivity models and primary-model exports |
| 07 | 05 | Subgroup models and subgroup-model exports |
| 08 | 05 | Patient fixed effects and pooled comparisons for Table S7 |
| 09 | 02, 05 | Sender-role composition for Table S9 |
| 10 | 06, 07, including their model export cells | Probability contrasts and final Figures 1 and 2 |
| 11 | 01, 02, 05 | Sender mapping and coverage checks for Table S9 |
| 12 | 05 | Reply windows and quantiles for Table 2 and Figure S2 |
| 13 | 02, 03, 04, 05 | Thread gap sensitivity and cross-thread checks for Table S4 |
| 14 | 02, 05 | Initiating message types and sensitivity models for Tables S10–S11 |
| 15 | 03, 05 | Encounter and initiator sensitivity for Tables S8 and S12 |
| 16 | 02, 05 | Exposure windows and model fit for Tables S13–S14 |
| 17 | 05 | Unadjusted associations for Table S15 |

## Thread construction

Notebook 02 sorts messages chronologically within patient, normalized subject, appointment department, and message-type grouping fields. A new thread is started when the elapsed time from the preceding message exceeds the selected gap threshold. The notebook evaluates 15-, 30-, and 60-day thresholds and records aggregate diagnostics. The manuscript analysis uses the finalized threaded handoff created by this notebook.

Thread-level variables include the first message and timestamp, initiating sender group, first subsequent patient reply, and reply-window indicators. The notebook code contains the exact grouping, sorting, gap, and validation operations.

## Exposure construction

Notebook 05 calculates rolling counts using recorded history and the initiating message, without using future messages from the index thread. Message exposure variables are separated into physician/APP, other HCW, all HCW, and system-generated sources. Prior patient-initiated threads are calculated separately. Continuous count variables used in regression models are transformed with `log1p`.

Recent health care activity is represented by encounters in the 7 days before thread initiation and transformed with `log1p`. Encounter construction and the rolling pre-thread merge are documented in notebooks 03 and 05.

## Output controls

All row-level handoffs and generated results are written under the ignored `outputs/` directory by default. Timestamped run directories preserve local audit artifacts without placing them in version control. Public notebook files intentionally contain no executed output.


## Additional analysis handoffs

For the complete manuscript output map, see [Tables and figures](OUTPUT_MAP.md).

- 06 exports aggregate_models/{A,B,C}/ within its analysis_models run.
- 07 exports aggregate_models/{Sex,Race,CCI}/ within its physapp_demographic_cci_interactions run.
- Each model export includes named coefficients, full patient-clustered covariance, a representative profile, settings, and a content manifest.
- 10 selects the latest complete primary/subgroup exports. Its new prediction_contrasts run contains four primary contrasts, reported race/moderate-CCI contrasts, dense probability/CI curves, and final vector PDF/SVG and 600-dpi PNG figures.
- 08 and 11–17 read the restricted handoffs in JMIR_WORKING_DATA_DIR. 11 also reads the provider master from 01; 13 reads three gap-specific message files from 02.
- The additional analyses create timestamped output folders with manifests and completion markers. Notebook 10 also records the two model exports it used.

Use separate output/working-data roots for different input versions to avoid mixing projects during run discovery.
