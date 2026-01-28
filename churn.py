import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

# ================= LOAD DATA =================
df = pd.read_csv("churn_data.csv")

# ================= ENCODE CATEGORICAL DATA =================
label_encoder = LabelEncoder()
categorical_cols = ["Gender", "Contract", "PaymentMethod", "Churn"]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ================= FEATURES & TARGET =================
X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

# ================= TRAIN-TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TRAIN MODEL =================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================= PREDICTIONS =================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ================= METRICS =================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================= CONFUSION MATRIX (PLOTLY) =================
cm = confusion_matrix(y_test, y_pred)

cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted No", "Predicted Yes"],
    y=["Actual No", "Actual Yes"],
    colorscale="Blues"
)

cm_fig.update_layout(
    title="Churn Prediction Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)

cm_fig.show()

# ================= FEATURE IMPORTANCE (PLOTLY) =================
importance = pd.Series(
    model.coef_[0],
    index=X.columns
).sort_values()

imp_fig = px.bar(
    importance,
    orientation="h",
    title="Feature Importance (Logistic Regression)"
)

imp_fig.show()

# ================= CHURN PROBABILITY OUTPUT =================
output = X_test.copy()
output["Actual Churn"] = y_test.values
output["Predicted Churn"] = y_pred
output["Churn Probability"] = y_prob

print("\nChurn Prediction Output:\n")
print(output)