"""Cricket X-Ray: discovering T20 cricket player performance profiles with K-Means and PCA.

Modules (in pipeline order):
    data_loader    read, validate, clean and merge the batting, bowling and fielding CSVs
    preprocessing  minimum-matches filter, engineered per-innings / per-match features
    modeling       clipping, imputation, scaling, K-Means, evaluation, PCA, interpretation
    charts         Matplotlib figures styled with the dashboard's design tokens

The Streamlit dashboard that uses them is app.py at the project root.
"""
