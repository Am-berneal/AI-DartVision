# AI DartVision - Evaluation Report

All numbers below were produced by `run_pipeline.py` on synthetic, geometry-labelled data.
They measure how well each ML model reproduces the deterministic scoring rules - not real-world player skill.

## Data validation

```
{'rows_in': 15750, 'dropped_non_numeric': 0, 'dropped_out_of_range': 0, 'dropped_duplicates': 0, 'rows_out': 15750}
```

## Scorecard - held-out stratified test set

| model               | feature_set   |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   score_accuracy |   score_mae |   boundary_accuracy |   interior_accuracy |   train_seconds |
|:--------------------|:--------------|-----------:|------------------:|---------------:|-----------:|-----------------:|------------:|--------------------:|--------------------:|----------------:|
| Geometry baseline   | rules         |     1      |            1      |         1      |     1      |           1      |      0      |              1      |              1      |            0    |
| Logistic Regression | raw           |     0.701  |            0.6966 |         0.701  |     0.6863 |           0.701  |      5.9238 |              0.7299 |              0.6841 |            1.19 |
| Decision Tree       | raw           |     0.9352 |            0.9359 |         0.9352 |     0.9347 |           0.9352 |      1.039  |              0.8844 |              0.9648 |            0.29 |
| Random Forest       | raw           |     0.9514 |            0.9521 |         0.9514 |     0.9505 |           0.9514 |      0.786  |              0.9154 |              0.9724 |            4.46 |
| KNN                 | raw           |     0.9498 |            0.9512 |         0.9498 |     0.9486 |           0.9498 |      0.8543 |              0.9137 |              0.9709 |            0.07 |
| SVM                 | raw           |     0.9035 |            0.9144 |         0.9035 |     0.9004 |           0.9035 |      1.8797 |              0.9034 |              0.9036 |            0.67 |
| Logistic Regression | engineered    |     0.7483 |            0.7452 |         0.7483 |     0.7411 |           0.7483 |      4.9352 |              0.7455 |              0.7499 |            1.56 |
| Decision Tree       | engineered    |     0.9975 |            0.9975 |         0.9975 |     0.9975 |           0.9975 |      0.0603 |              0.9931 |              1      |            0.11 |
| Random Forest       | engineered    |     0.9968 |            0.9969 |         0.9968 |     0.9968 |           0.9968 |      0.0724 |              0.9914 |              1      |            0.79 |
| KNN                 | engineered    |     0.9537 |            0.955  |         0.9537 |     0.9532 |           0.9537 |      0.8397 |              0.9085 |              0.9799 |            0.02 |
| SVM                 | engineered    |     0.9384 |            0.9432 |         0.9384 |     0.9373 |           0.9384 |      1.1876 |              0.9085 |              0.9558 |            0.27 |

## Scorecard - uniform board coverage (natural class imbalance)

| model               | feature_set   |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   score_accuracy |   score_mae |   boundary_accuracy |   interior_accuracy |   train_seconds |
|:--------------------|:--------------|-----------:|------------------:|---------------:|-----------:|-----------------:|------------:|--------------------:|--------------------:|----------------:|
| Geometry baseline   | rules         |     1      |            1      |         1      |     1      |           1      |      0      |              1      |              1      |            0    |
| Logistic Regression | raw           |     0.3692 |            0.3834 |         0.7023 |     0.3458 |           0.3692 |     12.1268 |              0.4085 |              0.3606 |            1.19 |
| Decision Tree       | raw           |     0.852  |            0.6996 |         0.9363 |     0.7758 |           0.852  |      2.2922 |              0.6228 |              0.902  |            0.29 |
| Random Forest       | raw           |     0.8482 |            0.6873 |         0.9511 |     0.7681 |           0.8482 |      2.583  |              0.6217 |              0.8977 |            4.46 |
| KNN                 | raw           |     0.832  |            0.6498 |         0.9441 |     0.7355 |           0.832  |      2.9544 |              0.5926 |              0.8843 |            0.07 |
| SVM                 | raw           |     0.7862 |            0.5717 |         0.9072 |     0.6309 |           0.7862 |      4.1284 |              0.5792 |              0.8314 |            0.67 |
| Logistic Regression | engineered    |     0.6356 |            0.5201 |         0.7457 |     0.5076 |           0.6356 |      6.9772 |              0.433  |              0.6798 |            1.56 |
| Decision Tree       | engineered    |     0.996  |            0.9728 |         0.9973 |     0.9835 |           0.996  |      0.0794 |              0.9777 |              1      |            0.11 |
| Random Forest       | engineered    |     0.9792 |            0.9193 |         0.9957 |     0.9521 |           0.9792 |      0.427  |              0.9219 |              0.9917 |            0.79 |
| KNN                 | engineered    |     0.865  |            0.6925 |         0.9492 |     0.7773 |           0.865  |      2.306  |              0.5926 |              0.9245 |            0.02 |
| SVM                 | engineered    |     0.8528 |            0.6388 |         0.9358 |     0.7137 |           0.8528 |      2.801  |              0.6083 |              0.9062 |            0.27 |

**Best ML model:** Decision Tree with engineered features (macro F1 = 0.9975).

`boundary_accuracy` = accuracy on throws within ~1.7 mm of a wire; this is where ML models disagree with the rules.

## Real data - DeepDarts (IEEE DataPort)

Source: W. McNally et al., "DeepDarts Dataset," IEEE DataPort, 2021, doi: 10.21227/05e7-xs69. Paper: "DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera," CVPR Sports Workshop 2021.

16535 unique real darts in 5702 visits over 51 sessions (2020-02-04 to 2021-02-23). Dart tips were converted to board coordinates with a homography through the 4 labelled calibration points, then scored with the geometry rules. Held-out test: 4146 darts from 11 unseen sessions.

Average 11.43 points per dart, 33.2 per 3-dart visit, miss rate 14.7%, treble rate 6.8%.

### Trained on synthetic data, tested on real throws

| model               | feature_set   | trained_on   |   accuracy |   macro_f1 |   boundary_accuracy |
|:--------------------|:--------------|:-------------|-----------:|-----------:|--------------------:|
| Geometry baseline   | rules         | rules        |     1      |     1      |              1      |
| Logistic Regression | raw           | synthetic    |     0.4636 |     0.4228 |              0.5062 |
| Decision Tree       | raw           | synthetic    |     0.897  |     0.8343 |              0.6938 |
| Random Forest       | raw           | synthetic    |     0.9011 |     0.8308 |              0.6771 |
| KNN                 | raw           | synthetic    |     0.8789 |     0.8023 |              0.6312 |
| SVM                 | raw           | synthetic    |     0.8003 |     0.6854 |              0.6125 |
| Logistic Regression | engineered    | synthetic    |     0.6679 |     0.5706 |              0.525  |
| Decision Tree       | engineered    | synthetic    |     0.9986 |     0.9953 |              0.9875 |
| Random Forest       | engineered    | synthetic    |     0.9855 |     0.968  |              0.9396 |
| KNN                 | engineered    | synthetic    |     0.9011 |     0.8352 |              0.6417 |
| SVM                 | engineered    | synthetic    |     0.8649 |     0.7666 |              0.6208 |

### Trained on real throws, tested on real throws (held-out sessions)

| model               | feature_set   | trained_on   |   accuracy |   macro_f1 |   boundary_accuracy |
|:--------------------|:--------------|:-------------|-----------:|-----------:|--------------------:|
| Geometry baseline   | rules         | rules        |     1      |     1      |              1      |
| Logistic Regression | raw           | real         |     0.335  |     0.1377 |              0.25   |
| Decision Tree       | raw           | real         |     0.9436 |     0.882  |              0.7292 |
| Random Forest       | raw           | real         |     0.9706 |     0.9263 |              0.8167 |
| KNN                 | raw           | real         |     0.9759 |     0.9451 |              0.8167 |
| SVM                 | raw           | real         |     0.9192 |     0.6212 |              0.6812 |
| Logistic Regression | engineered    | real         |     0.9165 |     0.6162 |              0.6729 |
| Decision Tree       | engineered    | real         |     0.9937 |     0.9824 |              0.9521 |
| Random Forest       | engineered    | real         |     0.9952 |     0.9859 |              0.9625 |
| KNN                 | engineered    | real         |     0.9761 |     0.9441 |              0.8188 |
| SVM                 | engineered    | real         |     0.9185 |     0.6367 |              0.6667 |

## Simulated player summary

| player                 | aim   |   throws |   hit_rate_target |   avg_score |   three_dart_avg |   mean_error_mm |   spread_x_mm |   spread_y_mm |   bias_x_mm |   bias_y_mm |   miss_rate | bias       |
|:-----------------------|:------|---------:|------------------:|------------:|-----------------:|----------------:|--------------:|--------------:|------------:|------------:|------------:|:-----------|
| Player A (pro)         | T20   |      300 |             0.46  |      38.28  |           114.84 |           7.035 |         5.276 |         5.872 |       0.347 |       0.196 |           0 | centred    |
| Player B (club)        | T20   |      300 |             0.287 |      28.933 |            86.8  |          13.402 |        10.603 |        10.731 |       2.291 |      -1.366 |           0 | right      |
| Player C (low & left)  | T20   |      300 |             0.23  |      27.08  |            81.24 |          13.52  |         9.164 |         9.165 |      -4.56  |      -6.756 |           0 | low & left |
| Player D (beginner)    | T20   |      300 |             0.063 |      14.71  |            44.13 |          28.356 |        22.935 |        21.295 |       0.908 |      -5.882 |           0 | low        |
| Player E (bull hunter) | IB    |      300 |             0.207 |      26.267 |            78.8  |          12.031 |         9.632 |         9.45  |       0.799 |       1.008 |           0 | centred    |

## Interpretation

* The geometry baseline scores 100% by construction - it is the rule set that produced the labels.
* Engineered polar features (r, sin θ, cos θ) help linear models most, because ring membership is a function of r.
* Remaining ML errors are concentrated at region boundaries (see confusion matrices in `figures/`).
* Player statistics come from simulated throwers; meaningful player analysis requires recorded real throws.