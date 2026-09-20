# Interview Guide

## Why put imputation and scaling in a Pipeline?
It guarantees that preprocessing is fitted only on the training data within each validation fold and that the same transformations are applied at inference.

## Why include a DummyRegressor?
Without a naive baseline, a model's RMSE has no context. A learned model should outperform a simple constant predictor under the same validation design.

## Why use RMSE for model selection but still report MAE and R2?
RMSE provides a consistent selection objective with stronger penalty for large errors. MAE remains easier to interpret, while R2 provides a relative fit measure.

## Why exclude `B`?
The variable is historically constructed from racial-composition information and is inappropriate to present as a neutral modern pricing feature in a portfolio model.

## Why is this not a production valuation system?
The data is historical, small, and not representative of modern transactions. The project demonstrates ML engineering rather than market-valid pricing performance.
