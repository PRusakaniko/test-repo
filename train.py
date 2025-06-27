import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- The Contract ---
# This script expects the input data to be available at /data/input/
# and will write its outputs to /output/
# The Aethelred worker will handle mapping these directories.

# Define output paths
output_path = Path("/output")
model_path = output_path / "model.joblib"
results_path = output_path / "results.json"

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# --- Model Training Logic ---
print("Executing client training script...")

# 1. Load Data
# This path is where the Aethelred worker will make the data available.
data_path = "/data/input/loan_applications.csv" 
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} records from {data_path}")

# 2. Feature Engineering & Model Selection
features = ['credit_score', 'income', 'loan_amount', 'age']
target = 'approved'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.4f}")

# 5. Write results to the specified output file (part of the contract)
results = {
    "accuracy": accuracy,
    "num_records_trained": len(X_train)
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

# 6. Save Model Artifact to the specified output file (part of the contract)
joblib.dump(model, model_path)
print(f"Model artifact saved to {model_path}")

print("Client script execution finished successfully.")
