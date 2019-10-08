# Run tests and reproduce all results

pytest -p no:warnings

python3 analysis_ols.py

python3 analysis_ridge.py

python3 analysis_lasso.py

python3 analysis_terrain.py
