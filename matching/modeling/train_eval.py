# matching/modeling/train_eval.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from matching.config.match_config import FEATURE_COLS

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)

    print(classification_report(y_test, preds))
    print("AUC:", auc)

    metrics = {"classification_report": report, "auc": auc}
    return model, metrics