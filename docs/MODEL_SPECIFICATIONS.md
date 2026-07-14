# Model Specifications

This document summarizes the executable specifications in notebooks 06 through 08. The notebook formulas are authoritative.

## General estimation framework

- Unit of analysis: one HCW-initiated conversation thread.
- Primary outcome: patient reply within 7 days (`patient_replied_7d`).
- Estimator: multivariable logistic regression using statsmodels.
- Uncertainty: robust standard errors clustered by patient (`DMRN`).
- Exposure scales: `log1p` of prior message or thread counts over 30- or 90-day windows.
- Time adjustment: categorical month-year fixed effects.
- Primary encounter adjustment: `log_encounters_past_7d`.
- Common patient covariates: age at message, gender, race, ethnicity, and Charlson group.

## Primary all-HCW model

For exposure period `p` (30 or 90 days), the executable formula is:

```text
patient_replied_7d ~
    log_provider_msgs_past_p
  + log_system_msgs_past_p
  + log_provider_msgs_past_p:log_system_msgs_past_p
  + log_patient_initiated_threads_past_p
  + log_encounters_past_7d
  + Age_at_Message
  + C(Gender)
  + C(Race, Treatment(reference="White"))
  + Ethnicity
  + C(Charlson_group)
  + C(month_year)
```

## Provider-type-specific models

The physician/APP-initiated and Other HCW-initiated samples use the same adjustment framework. Their focal exposure terms are:

```text
log_physapp_msgs_past_p
+ log_other_human_msgs_past_p
+ log_system_msgs_past_p
+ log_physapp_msgs_past_p:log_other_human_msgs_past_p
```

The models additionally include prior patient-initiated threads, recent encounters, age, gender, race, ethnicity, Charlson group, and month-year fixed effects. Only the thread-initiator sample restriction differs.

## Effect modification

Notebook 07 fits physician/APP-initiated thread models that add one interaction at a time between physician/APP message volume and:

- gender;
- race;
- age group (`Age65`); or
- Charlson group.

Each effect-modification model retains the physician/APP-by-Other-HCW exposure interaction and the full common adjustment set. Joint Wald tests evaluate categorical interactions.

## Prediction framework for Figures 1 and 2

Predictions are generated with full `model.predict(pd.DataFrame(...))` rows rather than manually reconstructing the linear predictor. Nonfocal continuous covariates are held at sample medians, and categorical covariates are held at their reference or modal values. Only the displayed focal exposure and interacting exposure levels are varied. Figure 1 is generated in notebook 06 and Figure 2 in notebook 07.

The plotting code exports vector PDF graphics and 300-dpi PNG previews. Presentation settings do not alter the fitted models or underlying predictions.

## Sensitivity and robustness analyses

Notebook 06 includes:

- 90-day exposure-window models;
- a 30-day patient-reply outcome;
- models excluding 2021 index threads;
- original versus encounter-adjusted model-fit comparisons;
- Figure S2, the empirical cumulative distribution of observed first patient replies; and
- absolute predicted-probability contrasts calculated from full prediction rows.

Notebook 08 compares pooled clustered logistic regression with conditional logistic patient fixed-effects models. The conditional models include within-patient-varying message exposures, their interaction, prior patient-initiated threads, recent encounters, Charlson group, and month-year terms. Time-invariant patient characteristics are absorbed by the patient fixed effects and are not separately estimated.
