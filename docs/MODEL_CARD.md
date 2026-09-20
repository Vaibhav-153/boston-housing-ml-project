# Model Card

## Model purpose

Educational regression and deployment practice using the historical Boston Housing dataset.

## Intended use

- learning regression methodology;
- comparing a naive baseline with learned models;
- practicing sklearn Pipelines;
- practicing model serialization and API serving;
- practicing tests, CI, and Docker.

## Excluded use

The model is not intended for real-estate appraisal, mortgage/lending decisions, investment decisions, housing policy, or modern market valuation.

## Feature policy

The original historical dataset includes `B`, a variable constructed from racial-composition information. The implementation excludes that variable from the model feature list.

## Validation

- 80/20 train/test split with fixed random state;
- 5-fold shuffled CV on the training split;
- model selection by mean CV RMSE;
- final MAE, RMSE, and R2 on the untouched test split.

## Generalization limits

The dataset is small and historical. Random splitting does not prove transfer across time, geography, or modern housing markets.
