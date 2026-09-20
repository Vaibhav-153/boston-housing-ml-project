# Boston Housing Regression - End-to-End ML Practice Project

An end-to-end regression project that demonstrates a reproducible machine-learning workflow: data validation, leakage-safe preprocessing, baseline comparison, cross-validated model selection, held-out evaluation, model serialization, Flask inference, tests, CI, and Docker deployment.

The project uses the historical Boston Housing dataset strictly as an **ML engineering practice dataset** rather than as a modern real-estate valuation product.

## Technology stack

- **Python / pandas / NumPy**
- **scikit-learn**
- **Flask + Gunicorn**
- **Joblib**
- **Pytest**
- **Ruff**
- **GitHub Actions**
- **Docker**

## ML workflow

```text
HousingData.csv
      |
      v
Schema validation
      |
      v
Train / test split
      |
      v
Training-only preprocessing pipeline
      |
      +--> median imputation
      +--> scaling where required
      |
      v
5-fold CV model comparison
      |
      +--> Dummy median baseline
      +--> Linear Regression
      +--> Ridge Regression
      +--> Random Forest Regression
      |
      v
Select lowest CV RMSE
      |
      v
Final fit on training data
      |
      v
Held-out test evaluation
      |
      +--> MAE
      +--> RMSE
      +--> R2
      |
      v
Serialized sklearn Pipeline
      |
      v
Flask API / HTML form
```

## Methodology improvements implemented

### Leakage-safe preprocessing

Imputation and scaling live inside sklearn Pipelines. Preprocessing parameters are learned only from training folds during cross-validation and from the training split during final fitting.

### No global IQR deletion

The implementation does not repeatedly delete observations using thresholds computed from the entire dataset. Rare but valid observations remain in evaluation unless they violate an explicit data-quality rule.

### Baseline comparison

A median `DummyRegressor` provides a naive reference. Learned models are useful only when they outperform this baseline under the same validation design.

### Model selection

Candidate models are ranked by mean 5-fold cross-validated RMSE using only the training split. The held-out test set is evaluated once after model selection.

### Single inference artifact

The complete preprocessing + estimator pipeline is serialized as one artifact. The Flask app therefore cannot accidentally use a scaler or feature order different from training.

## Responsible feature handling

The historical Boston dataset includes an original variable conventionally named `B`, constructed from racial-composition information. This implementation **excludes `B` from the model feature set** and documents the dataset's ethical limitations in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Reproducible execution

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python -m src.train
pytest -q
ruff check .
python app.py
```

Training writes:

```text
artifacts/model.joblib
artifacts/model_metadata.json
reports/metrics.json
```

## API

### Health check

```http
GET /health
```

### Prediction

```http
POST /predict_api
Content-Type: application/json
```

Example:

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

Response:

```json
{
  "predicted_median_value_thousands_usd": 25.123
}
```

## Input validation

The inference service rejects:

- missing features;
- unexpected fields;
- non-numeric values;
- NaN/infinite values;
- invalid `CHAS` values;
- requests made before a model artifact is available.

## Automated tests

The test suite covers:

- health endpoint;
- valid API prediction;
- missing feature rejection;
- unexpected feature rejection;
- invalid `CHAS` rejection;
- training-pipeline construction.

CI runs tests and Ruff on every push and pull request.

## Docker

```bash
docker build -t boston-housing-regression .
docker run --rm -p 8000:8000 boston-housing-regression
```

## Repository structure

```text
.
├── README.md
├── LICENSE
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── Procfile
├── render.yaml
├── portfolio.json
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
│   └── test_training.py
├── artifacts/
├── reports/
└── docs/
    ├── PROJECT_DOCUMENTATION.md
    ├── MODEL_CARD.md
    └── INTERVIEW_GUIDE.md
```

## Limitations

- The dataset is historical and small.
- A random split does not establish temporal or geographic generalization.
- The project is not a modern property-pricing model.
- Predictions are educational outputs, not appraisal, lending, investment, or policy advice.

## Portfolio role

**Machine Learning Engineering practice project**

The main value of the repository is the end-to-end engineering workflow around a regression problem.

## Author

**Vaibhav Admane**  
GitHub: [Vaibhav-153](https://github.com/Vaibhav-153)
