# Modeling Module (`matching/modeling/`)

This module is responsible for training, evaluating, and managing machine learning models used in entity matching. It includes functions to prepare data splits, train classification models, and assess their performance using standard metrics.

---

## Files and Functions

### `train_eval.py`

- `train_and_evaluate(train_df: pd.DataFrame) -> model`  
  Trains a logistic regression model on the provided training DataFrame. It splits the data into train and test sets, fits the model, predicts outcomes, calculates evaluation metrics including classification report and AUC, prints results, and returns the trained model instance.

---

_For more details on model training workflows, see the source file in this directory._