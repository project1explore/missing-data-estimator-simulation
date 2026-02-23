# Genesis Study Plan (Reevaluated)

## Why we are restarting
The initial draft was useful as a sketch but not sufficiently rigorous for a reproducible medical-statistics simulation paper. This plan replaces it with a tighter protocol and explicit literature grounding.

## Internet-backed literature check (performed)
Because `web_search` is not configured (missing Brave API key), references were verified via Crossref API fetches:

1. Rubin DB (1976), *Inference and missing data*, Biometrika. DOI: 10.1093/biomet/63.3.581
2. Sterne et al. (2009), *Multiple imputation... potential and pitfalls*, BMJ. DOI: 10.1136/bmj.b2393
3. Seaman & White (2013), *Review of inverse probability weighting for dealing with missing data*, Stat Methods Med Res. DOI: 10.1177/0962280210395740
4. Bang & Robins (2005), *Doubly Robust Estimation in Missing Data and Causal Inference Models*, Biometrics. DOI: 10.1111/j.1541-0420.2005.00377.x

## Research gap (target)
Applied medical analyses often compare complete-case, IPW, and doubly robust estimators, but far fewer practical studies map their **finite-sample behavior** under progressively informative outcome missingness and simultaneous mild model misspecification in trial-like settings.

## Experiment summary
- Trial-like binary endpoint setting with randomized treatment.
- Missing outcomes generated under MAR, moderate MNAR, and strong MNAR.
- Estimators: Complete-Case (CC), Outcome-model g-computation (OM), IPW, AIPW.
- Stress tests: outcome model misspecification and missingness model misspecification.
- Monte Carlo metrics: bias, RMSE, empirical 95% CI coverage, and type-I error (null scenarios).

## Deliverables
1. Reproducible simulation code (pure Python stdlib; no external dependencies).
2. Raw and summarized results (CSV + LaTeX tables).
3. Manuscript in LaTeX, compiled to PDF.
4. Full documentation in this repository.
