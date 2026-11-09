# American Express Default Prediction: Advanced Business Analytics (Assignment 4)

A credit-default prediction pipeline built on the Kaggle [American Express – Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) dataset. The notebook (`Advance_Assignment_4_Business_Analytics.ipynb`) covers data sampling, cleaning, feature selection, class balancing, model training (neural network, gradient-boosted ensemble, random forest) and model explainability (LIME and SHAP).

## Objective

Predict whether a credit card customer will default on their balance (`target` = 1) using anonymized, behavior-based monthly statement data, and explain which features drive the predictions.

## Dataset

| Item | Detail |
|---|---|
| Source | Kaggle Amex Default Prediction (`train_data.csv`, `train_labels.csv`) |
| Raw size | 5,531,451 monthly statement rows |
| Unique customers | 323,413 (after keeping the latest statement per customer) |
| Raw features | 190 columns, anonymized and grouped by prefix |
| Target | `target` (1 = default, 0 = no default) |

Feature prefixes: **D_** Delinquency, **S_** Spend, **P_** Payment, **B_** Balance, **R_** Risk.

The data is not included in this repository. Download it from Kaggle and update the file paths in the notebook (they currently point to Google Drive under `/content/drive/MyDrive/...`).

## Workflow

1. **Data loading and sampling**
   - Row count checked with PySpark.
   - The CSV is read in 1M-row chunks with pandas, and a 10% random sample of each chunk is kept. This reduces memory use.
   - The sample is saved as `my_dataframe.csv`.
2. **Data cleaning**
   - Known categorical columns (`B_30`, `B_38`, `D_114`, `D_116`, `D_117`, `D_120`, `D_126`, `D_63`, `D_64`, `D_66`, `D_68`) are cast to `category`.
   - Rows are sorted by `customer_ID` and `S_2` (date), and the **most recent statement per customer** is kept (`groupby().tail(1)`).
   - 30 columns with more than 50% missing values are dropped (190 → 160 columns).
   - Labels are merged in, and `customer_ID` and `S_2` are dropped.
   - Remaining nulls are imputed with the **mean** (numeric columns) or the **mode** (categorical columns).
3. **Encoding and feature selection**
   - Categorical features are one-hot encoded (`get_dummies`, `drop_first=True`), giving 182 columns.
   - A Random Forest is fit, and the **top 48 features by importance** are kept.
   - The most important feature is `P_2` (importance ≈ 0.077). Next are `B_9`, `D_44`, `D_48`, `B_3` and `D_61`.
4. **Exploratory data analysis**
   - Target class distribution is shown as a pie chart.
   - Density histograms of each selected numeric feature are split by `target`. Count plots are drawn for categorical features.
5. **Class balancing**
   - **SMOTE** (`random_state=42`) oversamples the minority class.
   - An 80/20 stratified train/test split follows (the train set has 193,580 positives and 193,580 negatives).
6. **Modeling**
   - **Neural network** (Keras): Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid). It uses the Adam optimizer, binary cross-entropy, batch size 1024 and up to 20 epochs with early stopping.
   - **LightGBM + XGBoost ensemble**: each has 100 estimators. The blend weight is chosen by grid search on test AUC.
   - **Random Forest**: 100 trees.
7. **Explainability**
   - **LIME** is used for single-prediction explanations (neural network and LightGBM).
   - **SHAP** is used for global feature importance (`DeepExplainer` for the neural network, and `Explainer` for LightGBM and XGBoost).
8. **Model persistence**: `lgb_model.pkl` and `xgb_model.pkl` are saved with `joblib`.

## Results

Metrics are computed on the SMOTE-balanced hold-out test set.

| Model | Accuracy | AUC |
|---|---|---|
| Neural Network | 0.8748 | ~0.93+ (validation) |
| LightGBM + XGBoost ensemble (0.2 / 0.8 weights) | 0.9060 | **0.9724** |
| Random Forest | **0.9082** | n/a |

The best blend weight was LightGBM = 0.2 and XGBoost = 0.8.

> The test set was resampled with SMOTE before splitting, so synthetic samples appear in both train and test. Accuracy and AUC are therefore likely optimistic compared with performance on real, imbalanced data. For a stricter evaluation, split first and apply SMOTE only to the training set. Also consider the competition's metric, which combines normalized Gini and top-4% default capture rate.

## Tech Stack

- **Data**: pandas, polars, PySpark, NumPy
- **ML**: scikit-learn, imbalanced-learn (SMOTE), LightGBM, XGBoost, CatBoost (installed, not used)
- **Deep learning**: TensorFlow / Keras
- **Explainability**: LIME, SHAP
- **Visualization**: matplotlib, seaborn

Versions seen in the notebook: scikit-learn 1.6.1, TensorFlow 2.18.0, LightGBM 4.5.0. The notebook was run in Google Colab.

## Getting Started

```bash
pip install pandas polars numpy matplotlib seaborn scikit-learn imbalanced-learn \
            lightgbm xgboost catboost tensorflow pyspark lime shap joblib
```

1. Download the competition data from Kaggle and place `train_data.csv` and `train_labels.csv` in a local folder.
2. Update the file paths in the notebook (cells 3, 5, 7 and 15).
3. Run the notebook top to bottom.

### Known issues to fix before re-running

- Cell 5 has an incomplete line (`df=`). It should be `df = pd.concat(sample_data)`.
- The Keras imports (`Sequential`, `Dense`, `Dropout`, `EarlyStopping`) and the `lime` and `shap` imports are missing from the import cell.
- Early stopping monitors training `accuracy` instead of `val_loss`, which is less robust.
- `DataFrame.applymap` is deprecated. Use `DataFrame.map`.
- The Random Forest used for feature selection is trained on the full dataset, including rows later used for evaluation. This leaks some information into feature selection.

## Repository Structure

```
.
├── Advance_Assignment_4_Business_Analytics.ipynb   # Main analysis notebook
├── lgb_model.pkl                                   # Saved LightGBM model (generated)
├── xgb_model.pkl                                   # Saved XGBoost model (generated)
├── my_dataframe.csv                                # 10% sampled training data (generated)
└── README.md
```

## Possible Improvements

- Aggregate the full statement history per customer (mean, last, min, max, deltas) instead of using only the latest statement.
- Balance with class weights, or apply SMOTE only after the train/test split.
- Tune hyperparameters (Optuna) and add CatBoost to the ensemble.
- Evaluate with the Amex competition metric and use time-aware cross-validation.

## Acknowledgements

- Gordon, R. (2022). *Prototype for Amex* [Notebook]. Kaggle. https://www.kaggle.com/code/robgordon2077/prototype-for-amex
- Tiwari, K. (2022). *Base level data engineering and file load* [Notebook]. Kaggle. https://www.kaggle.com/code/kshitiztiwari/base-level-data-engineering-and-file-load
- American Express – Default Prediction, Kaggle competition.
