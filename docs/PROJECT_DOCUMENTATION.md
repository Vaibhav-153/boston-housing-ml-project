# Project Documentation

## Original implementation risks addressed

### Preprocessing leakage
The refreshed training workflow moves median imputation and standardization inside Pipelines so they are fitted within the training process rather than on the complete dataset.

### Global outlier deletion
The workflow keeps rare observations unless they violate a documented validity rule. It does not repeatedly delete records based on full-dataset IQR thresholds.

### No baseline
A median DummyRegressor establishes the error level achievable without learning feature relationships.

### Separate model/scaler artifacts
The inference service loads one serialized sklearn Pipeline, preserving feature preprocessing and ordering.

## Model candidates

- Dummy median baseline;
- Linear Regression;
- Ridge;
- Random Forest.

The goal is not to maximize model count. The set compares a naive baseline, linear parametric models, regularization, and one nonlinear ensemble.

## Evaluation

MAE describes typical absolute error. RMSE penalizes large errors more strongly. R2 describes explained variance relative to a constant-mean baseline. All three are reported together.

## Serving architecture

```text
HTTP request
   |
   v
Flask validation
   |
   v
ordered pandas DataFrame
   |
   v
serialized sklearn Pipeline
   |
   v
numeric prediction
   |
   v
JSON or HTML response
```
