import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load dataset
data = pd.read_csv('dataset.csv')

# Feature engineering and preprocessing
X = data.drop(columns=['loan_default'])  # Replace 'loan_default' with your target column
y = data['loan_default']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')
print("Model saved as 'trained_model.pkl'")

# Save the scaler
joblib.dump(scaler, 'scale.pkl')
print("Scaler saved as 'scale.pkl'")

# Save selected features
selected_features = list(X.columns)
joblib.dump(selected_features, 'selected_features.pkl')
print("Selected features saved as 'selected_features.pkl'")

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

evaluation_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_pred_proba)
}

# Save evaluation metrics
with open('evaluation_metrics.json', 'w') as f:
    json.dump(evaluation_metrics, f, indent=4)
print("Evaluation metrics saved as 'evaluation_metrics.json'")

# Save train and test datasets
X_train['loan_default'] = y_train
X_test['loan_default'] = y_test

X_train.to_csv('model_train.csv', index=False)
print("Training dataset saved as 'model_train.csv'")

X_test.to_csv('model_test.csv', index=False)
print("Testing dataset saved as 'model_test.csv'")
