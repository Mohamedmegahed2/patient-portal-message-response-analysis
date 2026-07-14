# Data Dictionary and Restricted Inputs

This document describes expected file roles and principal variables. It contains no patient-level values. Exact source-column normalization and validation are implemented in the notebooks.

## Restricted source files

The default layout beneath `JMIR_RESTRICTED_DATA_DIR` is documented in `data/README.md`. The source extracts include:

- Eleven cleaned message-type CSV files.
- Governed provider and user-mapping files.
- Visit, provider, emergency, hospital, and diagnosis extracts.
- Patient demographic and birth-year extracts.
- Problem-list history and the Charlson ICD-10 lookup table.

These files may contain protected health information or operational identifiers and must not be placed in the public repository.

## Key identifiers and timestamps

| Variable | Definition |
|---|---|
| `DMRN` | Restricted patient identifier used for longitudinal linkage and patient-level clustering. |
| `MESSAGE_ID` | Restricted unique message identifier. |
| `CREATED_TIME` | Message creation timestamp. |
| `thread_id` | Derived conversation-thread identifier. |
| `thread_first_message_id` | Message identifier for the index message initiating a thread. |
| `thread_first_time` | Timestamp of the index message initiating a thread. |
| `first_patient_reply_time` | Timestamp of the first observed patient reply after thread initiation. |

## Message and thread classifications

| Variable | Definition |
|---|---|
| `SUBJECT_CLEAN` | Normalized subject used in thread grouping. No subject values are included publicly. |
| `Sender_all` / `Receiver_all` | Final analytic sender and receiver classifications. |
| `SENDER_CATEGORY` / `RECEIVER_CATEGORY` | Structured role buckets derived from provider mappings. |
| `SENDER_TYPE` | Detailed provider-role field used for the Table S9 role breakdown. |
| `origin` | Final message-source classification: patient, provider/HCW, or system. |
| `thread_initiator_group` | Thread initiator category used to define all-HCW, physician/APP, and Other HCW analytic samples. |

## Outcomes

| Variable | Definition |
|---|---|
| `patient_replied_7d` | Indicator that a first patient reply occurred from thread initiation through 7 days. |
| `patient_replied_30d` | Sensitivity indicator that a first patient reply occurred from thread initiation through 30 days. |
| `first_patient_reply_time` | Underlying observed first-reply timestamp used to construct reply-window outcomes and Figure S2. |

## Exposure variables

For each exposure period (`30d` and `90d`), notebook 05 creates raw rolling counts and `log1p` transforms. Principal regression variables include:

- `provider_msgs_past_<period>` and `log_provider_msgs_past_<period>`: prior HCW messages.
- `physapp_msgs_past_<period>` and `log_physapp_msgs_past_<period>`: prior physician/APP messages.
- `other_human_msgs_past_<period>` and `log_other_human_msgs_past_<period>`: prior non-physician/APP HCW messages.
- `system_msgs_past_<period>` and `log_system_msgs_past_<period>`: prior system-generated messages.
- `patient_initiated_threads_past_<period>` and `log_patient_initiated_threads_past_<period>`: prior patient-initiated threads.
- `encounters_past_7d` and `log_encounters_past_7d`: recent health care encounters before thread initiation.

## Covariates

| Variable | Definition |
|---|---|
| `Age_at_Message` | Patient age in years at the message timestamp. |
| `Age65` | Age category used in sensitivity/effect-modification analyses. |
| `Gender` | Governed demographic category used by the fitted models. |
| `Race` | Cleaned race category; the regression reference is White. |
| `Ethnicity` | Cleaned ethnicity category. |
| `Charlson_score` | Message-time Charlson Comorbidity Index score. |
| `Charlson_group` | Categorical CCI grouping used in models. |
| `month_year` | Calendar month-year fixed-effect category derived from `thread_first_time`. |

## Stable derived handoffs

| File | Unit |
|---|---|
| `messages_without_threads.parquet` | One row per cleaned message before thread assignment. |
| `messages_with_threads.parquet` | One row per cleaned message with thread assignments. |
| `encounters_unified_clean.parquet` | One row per harmonized encounter. |
| `patient_demographics_clean.parquet` | One row per patient. |
| `message_demographics_clean.parquet` | One row per message with age-at-message variables. |
| `message_level_charlson_cci.parquet` | One row per message with message-time CCI variables. |
| `analysis_df_all_exposures_with_encounter_covariates.parquet` | One row per HCW-initiated analytic thread. |
