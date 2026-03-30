import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("dataset/alzhdataset.csv")

data["Alzheimer"] = data["CDR"].apply(lambda x: 0 if x == 0 else 1)

# Feature engineering
data["Brain_Ratio"] = data["nWBV"] / data["eTIV"]
data["Cognitive_Index"] = data["MMSE"] / data["Age"]

X = data[["Age","Educ","SES","MMSE","eTIV","nWBV","ASF",
          "Brain_Ratio","Cognitive_Index"]]
y = data["Alzheimer"]

# Impute
imputer = KNNImputer()
X = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(n_estimators=500, learning_rate=0.03)

model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")