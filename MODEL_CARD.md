# Model Card — FairAI Legal System

> **Prototype only. This system is NOT suitable for real sentencing, bail, or any court decisions.**

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | FairAI Legal Risk Scorer (COMPAS Copy + Debiasing) |
| **Model type** | Ordinary Least Squares (OLS) linear regression |
| **Training target** | COMPAS `decile_score` rescaled to 0–100 |
| **Version** | 1.0 (final-review branch) |
| **Date** | November 2026 |
| **Authors** | yogarisi745-star |
| **License** | MIT |
| **Contact** | GitHub: yogarisi745-star/fairai |

---

## Intended Use

- **Primary use:** Academic demonstration of AI bias detection and mitigation in a legal context.
- **Intended users:** Students, researchers, policymakers studying algorithmic fairness.
- **Use case:** Illustrates how an AI risk-score system can carry demographic bias, how SHAP values can decompose that bias, and how counterfactual debiasing partially addresses it.

---

## Out-of-Scope Use

- Real bail, sentencing, parole, or any judicial decisions.
- Any jurisdiction outside Broward County, FL (2013–2014).
- Any population other than adult male/female defendants in the original COMPAS dataset.
- Use as a standalone risk-assessment tool without mandatory human-in-the-loop review.
- Use with race or gender as inputs for any real-world predictions.

---

## Training Data

| Field | Value |
|-------|-------|
| **Source** | ProPublica COMPAS Analysis dataset |
| **URL** | https://github.com/propublica/compas-analysis |
| **Raw rows** | 7,214 |
| **After filters** | 5,278 (ProPublica standard filters + race restriction) |
| **Race restriction** | African-American and Caucasian only (other groups lack statistical power) |
| **Time period** | 2013–2014, Broward County, FL |
| **Target variable (Model A)** | `decile_score` (COMPAS proprietary score, 1–10, rescaled 0–100) |
| **Target variable (Model B)** | `two_year_recid` (binary: reoffended within 2 years) |

### Filters Applied (ProPublica standard)
- `|days_b_screening_arrest| ≤ 30`
- `is_recid ≠ -1`
- `c_charge_degree ≠ 'O'` (ordinance violations excluded)
- `score_text ≠ 'N/A'`
- Race ∈ {African-American, Caucasian}

---

## Features

| Feature | Source | Notes |
|---------|--------|-------|
| `age_at_arrest` | COMPAS `age` | Continuous |
| `prior_felonies` | COMPAS `priors_count` | All priors mapped (no felony/misdemeanor split in COMPAS) |
| `offense_severity` | COMPAS `c_charge_degree` | F→6, M→3 (project's 1-10 scale) |
| `sex_female` | COMPAS `sex` | Binary (Female=1) |
| `race_marginalized` | COMPAS `race` | Binary (African-American=1, Caucasian=0) |

**Note:** `race_marginalized` and `sex_female` are **sensitive/protected attributes**. The counterfactual debiasing step neutralizes these to their reference values (Caucasian/Male) before producing the debiased recommendation.

---

## Performance Metrics (Test Set, n=1,584, seed=42)

All metrics from `results/fairness_report.json` — read dynamically by the app. Key headline numbers:

| Model | Accuracy | FPR Gap (AA−CA) | EOD |
|-------|----------|-----------------|-----|
| COMPAS Reference | 0.668 | +0.143 | 0.215 |
| Model A — COMPAS Copy (with demo) | 0.655 | +0.326 | 0.326 |
| Model A — COMPAS Copy (debiased) | 0.664 | +0.084 | 0.125 |
| Model B — Real Outcome (with demo) | 0.669 | +0.163 | 0.325 |
| Model B — Real Outcome (debiased) | 0.670 | +0.143 | 0.264 |

*Exact numbers may differ slightly — always read from `results/fairness_report.json`.*

### By Race Group (COMPAS Reference)

| Metric | African-American | Caucasian | Gap |
|--------|-----------------|-----------|-----|
| FPR (false positive rate) | Higher | Lower | +0.14 |
| FNR (false negative rate) | Lower | Higher | Negative |
| Selection Rate | Higher | Lower | Positive |

---

## Fairness Evaluation

| Method | EOD | FPR Gap |
|--------|-----|---------|
| Baseline (no mitigation) | 0.32 | +0.16 |
| Counterfactual Neutralization | 0.26 | +0.14 |
| Reweighing | 0.14 | +0.12 |
| ThresholdOptimizer (EqualizedOdds) | 0.06 | ~0 |
| ExponentiatedGradient (EqualizedOdds) | 0.05 | ~0 |

*Figures approximate — see `results/fairness_report.json` for exact values.*

---

## Limitations

1. **COMPAS-copy model:** Model A replicates COMPAS's own score — it is not an independent predictor. Its errors are COMPAS's errors.
2. **Proxy bias:** Counterfactual debiasing (setting race/gender to neutral) does not eliminate proxy bias. Features like `priors_count` and `charge_degree` are correlated with race due to systemic disparities in policing and charging. Removing race from the input does not remove its influence.
3. **Binary race grouping:** Only two race groups are modelled (data limitation). Results do not apply to Hispanic, Asian, Native American, or other groups in the dataset.
4. **Simulated judges:** All judge decisions in the demo are seeded from COMPAS data — no real judicial decisions are used.
5. **Inactive features:** `prior_misdemeanors`, `bail_failure_history`, `community_ties_score`, and `income_level` have zero weight because COMPAS does not provide them. Their conceptual importance is not reflected in the model.
6. **Historical data:** 2013–2014 Broward County, FL. Policing practices, demographics, and recidivism rates vary by time and location.
7. **Binary sex:** Non-binary gender is not in COMPAS. The `gender_Non-binary` weight is a placeholder (2.0, hand-coded) and has no empirical basis.
8. **Statistical power:** The JudicialBiasMonitor requires a minimum of 20 cases per judge for reliable bias detection (see `results/audit_validation.json`).

---

## Ethical Considerations

- **Do not use for real decisions.** This system does not meet any standard for evidentiary use in legal proceedings.
- **COMPAS itself is contested.** The ProPublica analysis found systematic racial disparity in COMPAS scores. This system reproduces that disparity (Model A) and attempts to measure it (Model B + fairness metrics).
- **Recidivism is not destiny.** Risk scores represent statistical patterns in historical data, not predictions about individuals.
- **Feedback loops:** Using risk scores in real bail decisions could create self-fulfilling prophecies (denied bail → job loss → higher recidivism). This system does not model or mitigate feedback loops.
- **Human oversight is mandatory.** The dual HITL design (judge review + oversight board) is the minimum responsible use pattern for any AI-assisted legal tool.

---

## Roadmap (Future Work)

- Replace COMPAS-copy with a model trained on real judicial outcome data from a consenting jurisdictional partner.
- Hyperledger Fabric multi-party blockchain for genuine tamper-evident ledger with audit trails across agencies.
- Authentication and role-based access control (judge vs. oversight board vs. defence counsel).
- User study with legal practitioners to validate interface and interpretation.
- Extend to non-binary race groups with sufficient sample sizes.

---

*Last updated: November 2026. See README.md for deployment and run instructions.*
