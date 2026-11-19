# Model Specifications

This document summarizes the executable specifications in notebooks 06 through 17. The executable formulas are defined in the notebooks and the shared analysis code.

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

Primary and subgroup models are fitted in notebooks 06 and 07. They export named coefficients, full patient-clustered covariance, and representative profiles for notebook 10, which generates the final figures and contrasts. Named design-column reconstruction is verified against the fitted models' Patsy predictions. Final artwork is exported as vector PDF/SVG and 600-dpi PNG. The exact profile settings, uncertainty calculations, and exposure support are described below.

## Sensitivity and robustness analyses

Notebook 06 includes:

- 90-day exposure-window models;
- a 30-day patient-reply outcome;
- models excluding 2021 index threads;
- original versus encounter-adjusted model-fit comparisons;
- Figure S2, the empirical cumulative distribution of observed first patient replies; and
- absolute predicted-probability contrasts calculated from full prediction rows.

Notebook 08 compares pooled clustered logistic regression with conditional logistic patient fixed-effects models. The conditional models include within-patient-varying message exposures, their interaction, prior patient-initiated threads, recent encounters, any within-patient-varying Charlson terms, and month-year terms. Time-invariant patient characteristics are absorbed by the patient fixed effects and are not separately estimated.


## Final prediction and sensitivity implementations

Notebook 10 uses full patient-clustered covariance from 06/07. Individual probability limits are logit-scale normal intervals, back-transformed. Differences and changes in differences use paired delta-method gradients.

Primary contrasts hold continuous covariates/other exposures at sample medians and categorical covariates at reference levels. Month–year is the mode: August 2022 for Models 1/3, March 2023 for Model 2. Subgroup predictions hold remaining categorical covariates at modes and continuous covariates at medians, with March 2023. The probabilities describe these representative profiles.

Primary comparisons: HCW 2 versus 8 with system 0 (Model 1); physician/APP 1 versus 5 with other HCW 1 (Model 2); other HCW 2 versus 5 with physician/APP 0 (Model 3); physician/APP 0 versus 3 with other HCW 2 (Model 3). Race/moderate-CCI differences use 1 and 5 physician/APP messages. Subgroup intervals are exploratory and unadjusted for multiple comparisons.

Final Figure 1 ranges: 1–18 (A), 1–13 (B), 0–9 (C), with a common 0%–62% y-axis. Figure 2 uses 1–13 and 20%–70%. Every curve has pointwise 95% bands. Counts include initiation, so zero is excluded from Figure 1A/B and Figure 2 but valid for physician/APP exposure in Figure 1C.

Notebook 08 uses exact conditional likelihood by log-space dynamic programming. It compares full-sample pooled, retained-sample pooled, and conditional estimates. Patients without within-patient outcome variation are excluded. Patient constants are absorbed; varying covariates and month–year effects remain. Covariance uses patient scores, observed information, and G/(G−1). Likelihood and scores are checked against statsmodels ConditionalLogit.

13 reconstructs anchors, outcomes, and prior patient-initiated counts for 15/30/60-day thread gaps, retaining 30-day exposure/7-day reply definitions. The cross-key diagnostic reports possible matches separately from confirmed replies. 14 adds initiating type to Models 2/3 with User Message as reference, pooling types with <100 initiators per sample. Role exclusion removes Administration/Unknown initiators from Model 3 while retaining original exposures.

15 varies encounter adjustment from 7 to 30 days and adds initiator category to Model 1. 16 varies exposures/prior patient-initiated history together across 7/14/30/90 days; fit comparisons use identical samples, plus a full-90-day-capture check. Thirty days remains primary. Lower AIC/BIC does not measure causal duration. 17 estimates one-predictor unadjusted models without other covariates/interactions.
