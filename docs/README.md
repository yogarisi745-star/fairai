# docs/ — Application Screenshots

This folder contains screenshots of the live FairAI Legal System application.

## Screenshots to Include

### 1. `tab1_shap_assessment.png`
**View 1 — Accused Assessment & SHAP Debiasing**
Capture after loading a Marginalized/Female benchmark case and clicking "Run Algorithmic Scoring & SHAP Debiasing".
Shows: risk scores, bias gap metric, SHAP horizontal bar chart, verdict comparison badge.

### 2. `tab4_bias_audit.png`
**View 4 — Judicial Oversight Board (HITL-2)**
Capture after committing enough cases (20+) via View 2 to trigger the statistical audit.
Shows: Disparate Impact Ratio, Demographic Parity Gap, p-value, FLAGGED or EQUITABLE status.

### 3. `view5_fairness_report.png`
**View 5 — Model Fairness Report**
Capture after generating `results/fairness_report.json` and deploying.
Shows: overall metrics table, FPR/FNR bar chart by race group, calibration curve, accuracy-vs-EOD scatter.

## How to Capture Screenshots (Manual)

1. Start the app locally:
   ```bash
   streamlit run app.py
   ```
2. Open http://localhost:8501 in a browser.
3. Navigate to each view and take a screenshot (F12 → Device toolbar → Full-page screenshot in Chrome, or use system screenshot tool).
4. Save as PNG files in this `docs/` folder with the filenames above.
5. Commit and push.

## Automated Screenshot Generation (Advanced — requires network access)

Install Playwright:
```bash
pip install playwright
playwright install chromium
```

Then run a script like:
```python
from playwright.sync_api import sync_playwright
import subprocess, time

proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.headless=true"])
time.sleep(8)  # wait for server
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:8501")
    page.screenshot(path="docs/tab1_shap_assessment.png", full_page=True)
    browser.close()
proc.terminate()
```

Note: Playwright requires network access and is not available in the Streamlit Community Cloud
build environment. Screenshots must be committed manually from a local machine.
