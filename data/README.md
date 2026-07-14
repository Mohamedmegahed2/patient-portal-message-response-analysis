# Restricted Local Data Layout

No data files belong in this repository. The `.gitignore` excludes common tabular formats and all files under `data/` other than this documentation.

Set `JMIR_RESTRICTED_DATA_DIR` to an authorized directory outside the repository. By default, the notebooks expect:

```text
<JMIR_RESTRICTED_DATA_DIR>/
|-- Cleaned/
|   |-- App_Cancel_Request_cleaned.csv
|   |-- App_Schedule_Request_cleaned.csv
|   |-- Case_Reminder_cleaned.csv
|   |-- Customer_Service_Request_cleaned.csv
|   |-- E_visit_cleaned.csv
|   |-- Medical_Advice_Request_cleaned.csv
|   |-- Medication_Renewal_Request_cleaned.csv
|   |-- User_Message_cleaned.csv
|   |-- Referral_cleaned.csv
|   |-- System_Message_cleaned.csv
|   |-- Treatment_Team_cleaned.csv
|   `-- Charlson_comorbidity_index.csv
|-- Providers/
|   |-- edited_provider_master.xlsx
|   |-- unmapped_user_ids_updated.csv
|   `-- MissingUsers.txt
`-- raw/
    |-- Demog.txt
    |-- ProblemList.txt
    |-- VisitDetails.txt
    |-- VisitProvider_Details.txt
    `-- added October 25/
        |-- BirthYear.xlsx
        |-- EmergencyVisits.txt
        |-- HospitalVisits(Inpt,Out,Emergency)_updated.txt
        |-- HospitalVisits_DischargeDiagnosis_updated.txt
        |-- ClinicalVisits_ClinicalDx_updated.txt
        `-- ClinicalVisits_BilledDx_updated.txt
```

The source filenames reflect the governed extracts used for this study. Optional environment variables documented in the main README can point to equivalent authorized locations when the directory structure differs.

Derived parquet handoffs are created in `outputs/working_data/` by default. Those files remain restricted and must not be uploaded to GitHub, attached to issues, or included in releases.
