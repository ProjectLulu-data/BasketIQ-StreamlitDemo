# BasketIQ — Streamlit Dashboard

Deployment package for the Group 2 CI3249 Advanced Data Mining project.

This repository contains the Streamlit dashboard, its runtime dependencies, and the saved analysis tables and visualizations used by the dashboard. It does not retrain the analytical models when the site is opened.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Streamlit Community Cloud

Select this repository, branch `main`, and main file path `app.py` at https://share.streamlit.io/.

This smaller deployment copy omits the large transaction-level source tables. Use the complete CI3249 submission package for the full project code and data-processing work.
