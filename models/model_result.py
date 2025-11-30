import joblib
import pandas as pd

model_path = "models/model.pkl"
model = joblib.load(model_path)

print(f"Model type: {type(model)}")
print("\nModel parameters:")
print(model.get_params())

if hasattr(model, "coef_"):
    coef_series = pd.Series(model.coef_[0])
    print("\nFeature importance (coefficients):")
    print(coef_series.sort_values(ascending=False))
else:
    print("\nModel has no coefficients to display.")