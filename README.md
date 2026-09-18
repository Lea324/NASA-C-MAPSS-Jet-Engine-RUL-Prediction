This project compares machine learning methods for predicting the Remaining Useful Life (RUL) of aircraft engines using NASA C-MAPSS data (FD001–-FD004), which vary in operating conditions and fault modes (HPC and fan degradation). 
Six approaches were evaluated: linear regression, Ridge regression, random forest, XGBoost and CNN.
Linear regression served as the baseline. 
Ridge regression alpha = 0.291 achieved a 0.95% lower MAE (25.96) and trained 6.8 times faster than the baseline.
Random Forest reached MAE 19.85 but required 2270 seconds, making it impractical.
XGBoost captured non-linear degradation well, with optimal over/under-penalty (5,1) yielding predicted MAE ~17.28 while reducing overestimation by 17%.
The CNN performed best, achieving OOF MAE 14.0043 and 590.16s training time via 10-fold GroupKFold. Its errors were nearly balanced 51% overestimation, 49% underestimation).
Overall, the CNN offered the best accuracy-cost trade-off.
