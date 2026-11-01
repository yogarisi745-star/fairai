# FairAI Legal System — Fairness Evaluation Report

> Generated: 2026-11-01T14:29:32.012669+00:00  |  Train n=3694  |  Test n=1584

**Honest reporting policy:** If debiasing does not reduce a gap, that is stated explicitly.
All numbers are test-set only (stratified 70/30 split, seed=42).

---

## Overall Metrics by Model Variant

| Model | Accuracy | ROC-AUC | FPR | FNR | Selection Rate | Brier |
|-------|----------|---------|-----|-----|----------------|-------|
| COMPAS Reference (decile>=5) | 0.6679 | 0.7205 | 0.3087 | 0.3584 | 0.4653 | 0.2339 |
| Model A — COMPAS Copy (with race/gender) | 0.6553 | 0.7164 | 0.2884 | 0.4081 | 0.4312 | 0.2206 |
| Model A — COMPAS Copy (debiased) | 0.6635 | 0.7172 | 0.1585 | 0.5369 | 0.3018 | 0.2271 |
| Model B — Real Outcome (with race/gender) | 0.6692 | 0.7212 | 0.2574 | 0.4134 | 0.4122 | 0.2126 |
| Model B — Real Outcome (debiased) | 0.6698 | 0.7230 | 0.2729 | 0.3946 | 0.4293 | 0.2129 |

## Fairness Gaps (African-American vs Caucasian, test set)

| Model | FPR Gap (AA−CA) | FNR Gap (AA−CA) | Dem. Parity Diff | Disparate Impact Ratio | EOD |
|-------|----------------|----------------|-----------------|----------------------|-----|
| COMPAS Reference (decile>=5) | +0.1427 | -0.2277 | +0.2216 | 1.6659 | 0.2277 |
| Model A — COMPAS Copy (with race/gender) | +0.3256 | -0.4307 | +0.4061 | 3.1555 | 0.4307 |
| Model A — COMPAS Copy (debiased) | +0.0839 | -0.2371 | +0.1896 | 2.0064 | 0.2371 |
| Model B — Real Outcome (with race/gender) | +0.1625 | -0.3200 | +0.2720 | 2.0897 | 0.3200 |
| Model B — Real Outcome (debiased) | +0.1433 | -0.2638 | +0.2375 | 1.8267 | 0.2638 |

## Fairlearn Mitigation Method Comparison

| Method | Accuracy | ROC-AUC | FPR Gap | FNR Gap | EOD |
|--------|----------|---------|---------|---------|-----|
| Baseline (no mitigation) | 0.6692 | 0.7212 | +0.1625 | -0.3200 | 0.3200 |
| Counterfactual Neutralization | 0.6698 | 0.7230 | +0.1433 | -0.2638 | 0.2638 |
| Reweighing | 0.6578 | 0.7081 | -0.1392 | +0.0219 | 0.1392 |
| ThresholdOptimizer (EqualizedOdds) | 0.6484 | 0.7212 | -0.0273 | -0.0608 | 0.0608 |
| ExponentiatedGradient (EqualizedOdds) | 0.6534 | 0.6857 | -0.0351 | -0.0546 | 0.0546 |

## Plain-Language Findings

1. COMPAS shows a false-positive-rate gap of +0.143 (African-American minus Caucasian): African-American defendants are incorrectly labeled high-risk at a higher rate than Caucasian defendants.

2. Counterfactual debiasing (Model B) reduces the FPR gap from +0.163 to +0.143. However, the gap does not reach zero because race correlates with other features (proxy bias).

3. Among all mitigation methods, 'ExponentiatedGradient (EqualizedOdds)' achieves the lowest equalized-odds difference (0.0546) at accuracy 0.6534. Best trade-off does not necessarily mean perfect fairness — examine FPR and FNR gaps separately.

4. Fairness-accuracy trade-off: the best mitigation method reduces equalized-odds difference by 0.2654 at a cost of 0.0158 accuracy points (1.58%). This trade-off is inherent and cannot be eliminated.

5. Limitations: race and sex are binary here due to COMPAS data constraints. Counterfactual debiasing cannot eliminate proxy bias through correlated legal-merit features. These results apply to Broward County, FL (2013-2014) only and should not be extrapolated to other jurisdictions or time periods.

---

## Limitations

- Race is restricted to African-American / Caucasian (COMPAS data limitation).
- The COMPAS-copy model (Model A) replicates COMPAS's own score; it is not a new model.
- Counterfactual debiasing cannot eliminate proxy bias through correlated features.
- Simulated judges in the system are labelled as such and use seeded COMPAS data only.
- Results apply to Broward County, FL (2013-2014). Do not extrapolate.
- **This prototype is NOT suitable for real sentencing or bail decisions.**