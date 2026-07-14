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
| 6 | `06_analysis_models_clean.ipynb` | Fit the primary and sensitivity models; create main tables and Figure 1; create Figure S2 and supplementary outputs | Timestamped tables and figures |
| 7 | `07_physapp_demographic_cci_interactions_clean.ipynb` | Fit physician/APP effect-modification models and create Figure 2 | Timestamped tables and figures |
| 8 | `08_patient_fixed_effects_robustness.ipynb` | Fit conditional logistic patient fixed-effects robustness models | Aggregate robustness summaries |
| 9 | `09_generate_table_s9_other_hcw_role_composition.ipynb` | Describe Other HCW-initiated threads by the structured role category of the initiating sender | Table S9 in CSV, HTML, and XLSX formats |

Notebook 09 reads the stable threaded-message and unified-analysis handoffs produced by steps 2 and 5. It does not rerun exposure construction or fit statistical models.

## Thread construction

Notebook 02 sorts messages chronologically within patient, normalized subject, and appointment-department grouping fields. A new thread is started when the elapsed time from the preceding message exceeds the selected gap threshold. The notebook evaluates 15-, 30-, and 60-day thresholds and records aggregate diagnostics. The manuscript analysis uses the finalized threaded handoff created by this notebook.

Thread-level variables include the first message and timestamp, initiating sender group, first subsequent patient reply, and reply-window indicators. The notebook code contains the exact grouping, sorting, gap, and validation operations.

## Exposure construction

Notebook 05 calculates prior rolling counts without using future messages from the index thread. Message exposure variables are separated into physician/APP, other HCW, all HCW, and system-generated sources. Prior patient-initiated threads are calculated separately. Continuous count variables used in regression models are transformed with `log1p`.

Recent health care activity is represented by encounters in the 7 days before thread initiation and transformed with `log1p`. Encounter construction and the rolling pre-thread merge are documented in notebooks 03 and 05.

## Output controls

All row-level handoffs and generated results are written under the ignored `outputs/` directory by default. Timestamped run directories preserve local audit artifacts without placing them in version control. Public notebook files intentionally contain no executed output.
