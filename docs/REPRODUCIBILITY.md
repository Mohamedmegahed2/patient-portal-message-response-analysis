# Environment and verification

Use Python 3.11.9 with [requirements.txt](../requirements.txt). statsmodels is 0.14.5 throughout. Original primary models used NumPy 1.24.4/pandas 2.2.3; additional calculations used NumPy 2.2.6/pandas 2.3.3. Plotting uses matplotlib 3.9.2; exact conditional likelihood uses numba 0.61.2/llvmlite 0.44.0.

Verification included notebook and Python syntax, imports, model exports from fresh primary and subgroup fits, and checks of the thread reconstruction and descriptive counts. The final figure probabilities, confidence intervals, and selected table exports reproduced the saved results.

The verification covered reported outputs and selected steps on the original data. A complete rerun from raw inputs was outside the verification scope. Numerical reproduction requires authorized institutional inputs.

The final conditional implementation replaces the older notebook 08 fitting implementation while retaining the exact estimator. Likelihood/scores are compared with statsmodels; convergence uses score/information diagnostics as well as optimizer results.
