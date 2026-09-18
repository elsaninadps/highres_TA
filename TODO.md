Assemble data
- [ ] `uv run python train_quantile_ensemble.py --stage assemble --config quantile_ensemble_config.yaml`

Uncertainty and evaluation
- [ ] for each member, predict the validation (depends on seed - data saved in models/<model>/splits) and test folds (same for all folds) - see schematic
- [ ] use validation RMSE to tune the uncertainty estimates - either STD or (Q90 - Q10) / 2.56 - roughly equal to STD for normal distribution. Use this to decide if STD or scaled interquantile approach is better
- [ ] compare test yhat and scaled uncertainty estimates with test uncertainty
- [ ] compare output against BATS and HOT

Inference
- [ ] Do predictions for full period. 

Reporting
- [ ] explain train test split 
- [ ] uncertainty estimation approach

NOTE: data is currently from 1993 onward only. Truncated early. need to update to 1982... 

