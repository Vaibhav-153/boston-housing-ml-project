# Boston Housing Regression

This project trains and serves regression models on the historical Boston Housing dataset. The main goal is to practice a clean machine-learning workflow: reproducible data loading, leakage-safe preprocessing, model comparison, held-out evaluation, model serialization, API inference, testing, CI, and Docker deployment.

This is a learning project. The dataset is historical and should not be used as a modern property-pricing or decision-making system.

## Problem statement

Given neighborhood-level housing attributes, predict `MEDV`, the historical median value of owner-occupied homes in thousands of US dollars.

The project compares a simple baseline with linear and nonlinear regression models, selects the model with the lowest cross-validated RMSE on the training split, and evaluates the selected model once on a held-out test set.

## Features

- reproducible dataset download and validation;
- fixed 80/20 train/test split;
- median imputation inside sklearn Pipelines;
- feature scaling for linear models;
- baseline comparison with `DummyRegressor`;
- 5-fold cross-validation on the training split only;
- Linear Regression, Ridge, and Random Forest candidates;
- MAE, RMSE, and R2 evaluation;
- one serialized preprocessing + model pipeline;
- Flask web form and JSON prediction API;
- input validation for missing, extra, non-numeric, and non-finite values;
- Pytest tests and Ruff linting;
- GitHub Actions CI;
- Render and Docker deployment files.

## Dataset

The project uses the historical Boston Housing dataset from StatLib. The download script fetches the same numeric dataset distributed by Keras and verifies the downloaded file with SHA-256 before converting it to `data/HousingData.csv`.

The original dataset contains 506 rows, 13 input attributes, and the `MEDV` target. The model intentionally excludes the historical `B` variable because it was constructed from racial-composition information. The remaining 12 variables are used for training.

The dataset is small, old, and geographically limited. It is useful for regression practice, not for current housing valuation.

## Technologies used

- Python
- pandas and NumPy
- scikit-learn
- Flask and Gunicorn
- Joblib
- Pytest
- Ruff
- GitHub Actions
- Docker
- Render

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/                 # generated model files
├── assets/                    # repository images/static project assets
├── data/                      # downloaded dataset
├── docs/
│   ├── INTERVIEW_GUIDE.md
│   ├── MODEL_CARD.md
│   └── PROJECT_DOCUMENTATION.md
├── reports/                   # generated evaluation metrics
├── scripts/
│   └── download_data.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── train.py
├── templates/
│   └── home.html
├── tests/
│   ├── test_app.py
│   ├── test_download_data.py
│   └── test_training.py
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── LICENSE
├── Makefile
├── portfolio.json
├── Procfile
├── pyproject.toml
├── render.yaml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

Generated model, metadata, dataset, and metric files are ignored by Git so they can be reproduced from the source code.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vaibhav-153/boston-housing-ml-project.git
cd boston-housing-ml-project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Download the dataset

```bash
python scripts/download_data.py
```

The script creates:

```text
data/HousingData.csv
```

## Train the model

```bash
python -m src.train
```

Training creates:

```text
artifacts/model.joblib
artifacts/model_metadata.json
reports/metrics.json
```

## Model and evaluation method

The split and model-selection process is fixed so repeated runs use the same evaluation design:

1. Split the dataset into 80% training data and 20% test data using `random_state=42`.
2. Keep the test set untouched during model selection.
3. Compare four candidates with shuffled 5-fold cross-validation on the training split.
4. Select the candidate with the lowest mean CV RMSE.
5. Fit that pipeline on the full training split.
6. Evaluate it once on the held-out test set using MAE, RMSE, and R2.

The candidate set is intentionally small:

| Model | Preprocessing |
| --- | --- |
| Dummy median | Median imputation |
| Linear Regression | Median imputation + standardization |
| Ridge Regression | Median imputation + standardization |
| Random Forest | Median imputation |

Preprocessing is part of each sklearn `Pipeline`, so imputation and scaling are fitted only on training data during cross-validation and final training.

## Results

The training command prints the selected model, cross-validation RMSE for every candidate, and the held-out MAE, RMSE, and R2. The same values are saved to `reports/metrics.json`.

Generated reports are intentionally ignored by Git, so this README does not claim a score that cannot be verified from the repository itself. Run the commands below to reproduce the current result from the pinned dataset source and fixed random state:

```bash
python scripts/download_data.py
python -m src.train
cat reports/metrics.json
```

This keeps the reported result tied to the exact code and dataset used for the run.

## Run the web app

After training:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The home page provides a simple prediction form.

## Prediction API

Endpoint:

```http
POST /predict_api
Content-Type: application/json
```

Example request:

```json
{
  "data": {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296,
    "PTRATIO": 15.3,
    "LSTAT": 4.98
  }
}
```

Example response:

```json
{
  "predicted_median_value_thousands_usd": 25.123
}
```

## Health check

```http
GET /health
```

The endpoint returns HTTP `200` only when the model is loaded. If the model artifact is missing, it returns HTTP `503` with `model_ready: false`. This prevents a deployment from reporting itself healthy when predictions cannot run.

## Tests and linting

Run the complete local check:

```bash
make check
```

Or run the commands separately:

```bash
ruff check .
python -m pytest -q
```

CI runs the same lint and test checks on pushes and pull requests.

## Docker

Build the image:

```bash
docker build -t boston-housing-regression .
```

The Docker build downloads the dataset and trains the model so the resulting image is ready to serve predictions.

Run it locally:

```bash
docker run --rm -p 8000:8000 boston-housing-regression
```

Then open:

```text
http://127.0.0.1:8000
```

The container uses `PORT` when the deployment platform provides it and falls back to port `8000` for local runs.

## Limitations

- The dataset contains only 506 historical observations.
- It describes Boston-area data from the 1970s and does not represent current housing markets.
- A random train/test split does not test generalization across time or geography.
- The candidate model set is deliberately small and does not include extensive hyperparameter tuning.
- The project is for ML engineering practice, not appraisal, lending, investment, or housing-policy decisions.

## Future improvements

- add a small, documented hyperparameter search while keeping the held-out test set untouched;
- report cross-validation standard deviation in addition to mean RMSE;
- add model explainability for the final estimator where it is appropriate;
- add an integration test that builds and starts the Docker image in CI;
- move to a more suitable modern housing dataset for a production-style version of the project.

## Author

Vaibhav Admane  
GitHub: https://github.com/Vaibhav-153
