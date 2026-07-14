# Patient Portal Reply Analysis Code

This repository contains the analytic code supporting the manuscript:

> Patient Responsiveness to Health Care Worker Messages in a Patient Portal: Retrospective Observational Study

The repository documents the message preprocessing, conversation-thread construction, encounter and covariate construction, exposure definitions, regression specifications, sensitivity analyses, and figure-generation workflow used for the revised manuscript.

## Data availability and privacy

No patient-level electronic health record (EHR), patient-portal message, provider, encounter, or derived analytic data are included. These data cannot be shared publicly because of patient privacy protections and institutional data-governance requirements.

The code can be inspected without the data. Reproducing the numerical results authorized access to the underlying institutional data using the schemas described in [data/README.md](data/README.md). Do not commit source data, derived row-level data, notebook outputs, or generated results to this repository.

## Repository contents

- `notebooks/`: nine notebooks covering the analytic pipeline and supplementary Table S9.
- `docs/PIPELINE.md`: workflow and handoff files.
- `docs/DATA_DICTIONARY.md`: key restricted inputs and derived variables.
- `docs/MODEL_SPECIFICATIONS.md`: regression and prediction specifications.
- `data/README.md`: authorized local-data layout expected by the notebooks.
- `requirements.txt`: Python dependencies.

## Software environment

The core analysis environment reported in the manuscript is Python 3.11.9 with statsmodels 0.14.5, pandas 2.2.3, and NumPy 1.24.4. Auxiliary I/O, plotting, and notebook dependencies are also pinned in `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Restricted-data configuration

Set `JMIR_RESTRICTED_DATA_DIR` to the authorized local root containing the raw, cleaned-message, and provider source directories. Do not place that directory inside this repository.

PowerShell example:

```powershell
$env:JMIR_RESTRICTED_DATA_DIR = "D:\authorized-data\patient-portal-study"
```

macOS/Linux example:

```bash
export JMIR_RESTRICTED_DATA_DIR="/secure/authorized-data/patient-portal-study"
```

Optional variables:

- `JMIR_REPRO_OUTPUT_DIR`: generated output root; defaults to `outputs/` in the repository.
- `JMIR_WORKING_DATA_DIR`: derived handoff directory; defaults to `outputs/working_data/`.
- `JMIR_RAW_DATA_DIR`: overrides `<JMIR_RESTRICTED_DATA_DIR>/raw`.
- `JMIR_ADDITIONAL_DATA_DIR`: overrides `<raw>/added October 25`.
- `JMIR_CLEANED_MESSAGE_DIR`: overrides `<JMIR_RESTRICTED_DATA_DIR>/Cleaned`.
- `JMIR_PROVIDER_SOURCE_DIR`: overrides `<JMIR_RESTRICTED_DATA_DIR>/Providers`.

The optional overrides accommodate differences in authorized storage layouts without embedding local paths in the code.

## Notebook order

Run the notebooks from the repository root or the `notebooks/` directory:

1. `01_build_core_message_preprocessing_clean.ipynb`
2. `02_threading_diagnostics_sensitivity.ipynb`
3. `03_build_visit_encounters_clean.ipynb`
4. `04_build_demographics_cci_clean.ipynb`
5. `05_build_exposure_analysis_dataset_clean.ipynb`
6. `06_analysis_models_clean.ipynb`
7. `07_physapp_demographic_cci_interactions_clean.ipynb`
8. `08_patient_fixed_effects_robustness.ipynb`
9. `09_generate_table_s9_other_hcw_role_composition.ipynb`

Notebook 09 requires only the stable handoffs created by notebooks 02 and 05. It can therefore generate Table S9 directly without rerunning exposure construction or fitting any models.

All generated handoff datasets and results are written beneath `outputs/` by default. That directory is excluded from version control.

## Reproducibility scope

The public notebooks preserve the analysis logic and model specifications. For public release, saved outputs and execution counts were cleared, machine-specific paths and legacy data fallbacks were removed, and restricted input locations were replaced with environment-variable configuration. No statistical models, samples, coefficients, exposure definitions, interaction levels, or outcome definitions were changed during repository preparation.

## License and citation

The code is released under the [MIT License](LICENSE). Citation metadata are provided in [CITATION.cff](CITATION.cff).
