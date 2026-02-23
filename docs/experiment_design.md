# Extended Experiment Design

## Clinical-motivation framing
Synthetic two-arm randomized trial with a binary follow-up endpoint (e.g., remission yes/no) and realistic baseline heterogeneity.

## Data-generating process (DGP)
For each participant:

- Covariates: age, comorbidity burden, baseline severity, biomarker
- Treatment assignment: randomized Bernoulli(0.5)
- Outcome model: logistic model with treatment effect, severity interaction, and mild nonlinearity

Target estimand:

- Marginal treatment log-odds contrast

## Missingness mechanisms
Outcome-observation indicator `R` depends on covariates and treatment in all scenarios; dependence on true outcome `Y` is controlled by `gamma`:

1. **MAR** (`gamma=0.0`)
2. **Moderate MNAR-like** (`gamma=0.6`)
3. **Strong MNAR-like** (`gamma=1.0`)

## Estimators
1. Complete-case (CC)
2. Outcome-model g-computation (OM)
3. Inverse probability weighting (IPW)
4. Augmented inverse probability weighting (AIPW)

## Stress conditions
For non-null scenarios:

- Correct nuisance models
- Outcome model misspecified
- Missingness model misspecified
- Both nuisance models misspecified

Additionally, null-effect scenarios are included for type-I error assessment.

## Sample-size regime
- `n = 350`
- `n = 700`

(Null scenarios evaluated at `n = 700`.)

## Monte Carlo budget
- **Replicates per scenario:** 140
- **Total scenarios:** 27
- **Total simulated datasets:** 3780
- **Method-specific estimates:** 15120

## Performance metrics
- Bias
- RMSE
- Empirical 95% CI coverage (when SE available)
- Type-I error (null scenarios)
- Mean outcome-observation rate

## Expected pattern (hypothesis)
- CC should deteriorate under informative missingness
- IPW should reduce bias but may lose precision
- AIPW should provide strongest robustness when at least one nuisance model is near-correct
- Under strong MNAR-like missingness, all MAR-based approaches should degrade
